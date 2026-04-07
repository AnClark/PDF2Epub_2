"""
转换流水线：PDF → Document → EPUB / Typst

流程（v3 三阶段，支持并行 OCR）：
  阶段 0 — 渲染：逐页将 PDF 渲染为 JPEG 字节缓存（串行，速度快）。
  阶段 1 — OCR：使用 ThreadPoolExecutor 并行对各页执行 Tesseract 识别；
           同时向 HeaderFooterFilter 收集页眉/页脚候选文本。
           workers=1 等价于完全串行，不引入额外开销。
  阶段 2 — 版面分析：利用缓存数据重建元素列表（标题/段落/图片），
           HeaderFooterFilter 已 finalize()，过滤精度更高。
  后处理 — 跨页段落合并：检测上一页末尾未完结的段落，与下一页首段合并。
  导出   — 调用对应导出器输出 EPUB / Typst 文件。
"""

import io
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import List, Optional

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .document_model import (
    Chapter, Document, DocumentElement, HeadingLevel, ImageBlock, TextBlock
)
from .exporters.epub_exporter import EPUBExporter
from .exporters.typst_exporter import TypstExporter
from .layout_analyzer import HeaderFooterFilter, LayoutAnalyzer, SENTENCE_END_RE
from .ocr_processor import OCRProcessor, OCRResult
from .pdf_renderer import PDFRenderer

_console = Console()

# ── 断点续传 ─────────────────────────────────────────────────────────────────
_CKPT_VERSION = "pdf2ebook-v1"
_CKPT_SAVE_INTERVAL = 5  # 每完成 N 页 OCR 保存一次断点

# ── 渲染缓存 ─────────────────────────────────────────────────────────────────
_RENDER_CACHE_VERSION = "pdf2ebook-render-v1"

# ── OCR 缓存 ──────────────────────────────────────────────────────────────────
_OCR_CACHE_VERSION = "pdf2ebook-ocr-v1"


class OcrCache:
    """OCR 识别结果缓存容器。

    将所有页面的 OCRResult 持久化到独立文件，与断点文件和渲染缓存分开存储。
    缓存键：PDF 修改时间 + lang + 总页数，三者任一变化则缓存失效。
    全流程成功后缓存文件**保留**，供下次运行复用。
    """

    def __init__(self, total: int, pdf_mtime: float, lang: str) -> None:
        self.version: str = _OCR_CACHE_VERSION
        self.pdf_mtime: float = pdf_mtime
        self.lang: str = lang
        self.total: int = total
        self.ocr_results: List[Optional["OCRResult"]] = [None] * total

    # ── 序列化 ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """原子写入：先写 .tmp，再 os.replace，防止中断导致文件损坏。"""
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "OcrCache":
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, OcrCache) or obj.version != _OCR_CACHE_VERSION:
            raise ValueError("OCR 缓存文件格式版本不兼容")
        return obj

    # ── 状态查询 ────────────────────────────────────────────────────────
    def is_complete(self) -> bool:
        """所有页面均已缓存时返回 True。"""
        return all(r is not None for r in self.ocr_results)

    def completed_count(self) -> int:
        return sum(1 for r in self.ocr_results if r is not None)

    def pending_pages(self) -> List[int]:
        """返回尚未缓存的页码列表。"""
        return [i for i, r in enumerate(self.ocr_results) if r is None]


class RenderCache:
    """渲染结果缓存容器。

    将所有页面的 JPEG 字节缓存持久化到独立文件，与断点文件分开存储。
    缓存键：PDF 修改时间 + DPI + 总页数，三者任一变化则缓存失效。
    全流程成功后缓存文件**保留**，供下次运行复用。
    """

    def __init__(self, total: int, pdf_mtime: float, dpi: int) -> None:
        self.version: str = _RENDER_CACHE_VERSION
        self.pdf_mtime: float = pdf_mtime
        self.dpi: int = dpi
        self.total: int = total
        self.page_jpeg_bytes: List[Optional[bytes]] = [None] * total

    # ── 序列化 ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """原子写入：先写 .tmp，再 os.replace，防止中断导致文件损坏。"""
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "RenderCache":
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, RenderCache) or obj.version != _RENDER_CACHE_VERSION:
            raise ValueError("渲染缓存文件格式版本不兼容")
        return obj

    # ── 状态查询 ────────────────────────────────────────────────────────
    def is_complete(self) -> bool:
        """所有页面均已缓存时返回 True。"""
        return all(b is not None for b in self.page_jpeg_bytes)

    def pending_pages(self) -> List[int]:
        """返回尚未缓存的页码列表。"""
        return [i for i, b in enumerate(self.page_jpeg_bytes) if b is None]


class Checkpoint:
    """断点续传数据容器。

    存储格式：pickle（使用原子写入防止中断时损坏）。
    包含：所有页面的 JPEG 字节缓存 + OCR 识别结果。
    """

    def __init__(self, total: int, pdf_mtime: float) -> None:
        self.version: str = _CKPT_VERSION
        self.pdf_mtime: float = pdf_mtime
        self.total: int = total
        # None 表示该页尚未完成对应阶段
        self.page_jpeg_bytes: List[Optional[bytes]] = [None] * total
        self.ocr_results: List[Optional["OCRResult"]] = [None] * total

    # ── 序列化 ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """原子写入：先写 .tmp，再 os.replace，防止中断导致文件损坏。"""
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301  — 读取本工具自己写的文件，可信
        if not isinstance(obj, Checkpoint) or obj.version != _CKPT_VERSION:
            raise ValueError("断点文件格式版本不兼容")
        return obj

    # ── 状态查询 ────────────────────────────────────────────────────────
    def completed_ocr_count(self) -> int:
        return sum(1 for r in self.ocr_results if r is not None)

    def pending_render_pages(self) -> List[int]:
        """返回尚未渲染的页码列表。"""
        return [i for i, b in enumerate(self.page_jpeg_bytes) if b is None]

    def pending_ocr_pages(self) -> List[int]:
        """返回尚未完成 OCR 的页码列表。"""
        return [i for i, r in enumerate(self.ocr_results) if r is None]


def _ocr_worker(page_num: int, jpeg_bytes: bytes, lang: str):
    """线程 Worker：对单页 JPEG 字节执行 OCR，返回 (page_num, OCRResult)。

    在独立线程中创建 OCRProcessor 实例以避免共享状态；
    pytesseract 调用外部 tesseract 进程，天然释放 GIL，适合多线程并行。
    """
    img = Image.open(io.BytesIO(jpeg_bytes))
    result = OCRProcessor(lang=lang).process(img)
    return page_num, result


def _ends_sentence(text: str) -> bool:
    """判断文本是否以句末标点结尾（用于跨页段落合并）。"""
    return bool(SENTENCE_END_RE.search(text.rstrip()))


def _merge_cross_page_paragraphs(
    page_elements: List[List[DocumentElement]],
) -> List[DocumentElement]:
    """
    合并跨页未完结的段落。

    规则：若第 N 页最后一个元素是文本段落（非标题），且不以句末标点结尾，
    同时第 N+1 页第一个元素也是文本段落（非标题），则两者合并为一段。
    遇到图片则中断合并链。
    """
    result: List[DocumentElement] = []
    pending: Optional[TextBlock] = None  # 跨页等待合并的段落

    for page_elems in page_elements:
        processed: List[DocumentElement] = []

        for i, elem in enumerate(page_elems):
            # 尝试把上一页的 pending 与本页第一个段落合并
            if i == 0 and pending is not None:
                if isinstance(elem, TextBlock) and not elem.is_heading:
                    pending.text = pending.text + elem.text  # 不加空格：中文无词间距
                    processed.append(pending)
                    pending = None
                    continue
                else:
                    # 首个元素不是段落（是标题或图片），不合并
                    result.append(pending)
                    pending = None

            processed.append(elem)

        # 空页处理
        if not processed and pending is not None:
            result.append(pending)
            pending = None

        # 检查本页末尾：最后一个元素若为未完结段落，暂存等待下页合并
        # 条件：最后元素是 TextBlock（非标题）且不以句末标点结尾
        if (
            processed
            and isinstance(processed[-1], TextBlock)
            and not processed[-1].is_heading
            and not _ends_sentence(processed[-1].text)
        ):
            pending = processed.pop()

        result.extend(processed)

    # 末页可能有残留的 pending（最后一页末段无后续页）
    if pending is not None:
        result.append(pending)

    return result


class Pipeline:
    def __init__(
        self,
        pdf_path: str,
        output_dir: str = ".",
        output_format: str = "both",
        lang: str = "chi_sim",
        dpi: int = 300,
        title: Optional[str] = None,
        author: Optional[str] = None,
        verbose: bool = False,
        workers: int = 1,
        fresh: bool = False,
        cache_render: bool = False,
        cache_ocr: bool = False,
    ) -> None:
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.output_format = output_format.lower()
        self.lang = lang
        self.dpi = dpi
        self.verbose = verbose
        self.workers = max(1, workers)
        self.fresh = fresh
        self.cache_render = cache_render
        self.cache_ocr = cache_ocr

        stem = Path(pdf_path).stem
        self.title = title or stem
        self.author = author or ""
        self.typst_path = os.path.join(output_dir, f"{stem}.typ")
        self.epub_path = os.path.join(output_dir, f"{stem}.epub")

    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        _console.print(Panel(
            f"[bold cyan]PDF2Ebook[/]  →  [white]{self.pdf_path}[/]\n"
            f"输出目录：[dim]{self.output_dir}[/]  |  格式：[yellow]{self.output_format}[/]  "
            f"|  DPI：{self.dpi}  |  OCR 线程：[yellow]{self.workers}[/]",
            title="[bold]任务配置[/]",
            border_style="blue",
        ))

        document = self._build_document()

        elem_count = sum(len(c.elements) for c in document.chapters)
        ch_count = len(document.chapters)

        # ── 统计表格 ────────────────────────────────────────────────────
        tbl = Table(show_header=False, box=None, pad_edge=False)
        tbl.add_column(style="dim", width=16)
        tbl.add_column(style="bold green")
        tbl.add_row("识别章节数", str(ch_count))
        tbl.add_row("识别元素数", str(elem_count))
        _console.print(Panel(tbl, title="[bold]识别结果[/]", border_style="green"))

        # ── 导出 ────────────────────────────────────────────────────────
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            export_task = progress.add_task("[bold]导出文件[/]", total=2 if self.output_format == "both" else 1)

            if self.output_format in ("typst", "both"):
                progress.update(export_task, description=f"[cyan]导出 Typst[/] → [dim]{self.typst_path}[/]")
                TypstExporter().export(document, self.typst_path)
                progress.advance(export_task)

            if self.output_format in ("epub", "both"):
                progress.update(export_task, description=f"[cyan]导出 EPUB[/]  → [dim]{self.epub_path}[/]")
                EPUBExporter().export(document, self.epub_path)
                progress.advance(export_task)

            progress.update(export_task, description="[green]导出完成[/]")

        _console.print("[bold green]✓ 全部完成！[/]")

    # ──────────────────────────────────────────────────────────────────────
    def _build_document(self) -> Document:
        os.makedirs(self.output_dir, exist_ok=True)
        ckpt_path = self._checkpoint_path()
        ckpt = self._load_or_create_checkpoint(ckpt_path)
        total = ckpt.total
        analyzer = LayoutAnalyzer()

        if self.verbose:
            _console.print(
                f"[dim]  PDF 共 {total} 页，DPI={self.dpi}，"
                f"语言={self.lang}，OCR 线程={self.workers}[/]"
            )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=36),
            MofNCompleteColumn(),
            TextColumn("[dim]页[/]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_console,
            transient=False,
        )

        with progress:
            # ── 阶段 0：渲染未完成的页面 ─────────────────────────────────
            pending_render = ckpt.pending_render_pages()
            if pending_render:
                render_task = progress.add_task(
                    "[cyan]阶段 0[/]  渲染页面",
                    total=total,
                    completed=total - len(pending_render),
                )
                try:
                    # 尝试从渲染缓存中补充已渲染的页面
                    if self.cache_render:
                        render_cache = self._load_render_cache()
                        if render_cache is not None:
                            hit_pages = []
                            for page_num in list(pending_render):
                                cached_bytes = render_cache.page_jpeg_bytes[page_num]
                                if cached_bytes is not None:
                                    ckpt.page_jpeg_bytes[page_num] = cached_bytes
                                    hit_pages.append(page_num)
                            if hit_pages:
                                pending_render = ckpt.pending_render_pages()
                                progress.update(
                                    render_task,
                                    completed=total - len(pending_render),
                                )
                                if self.verbose:
                                    _console.print(
                                        f"[dim]  渲染缓存命中 {len(hit_pages)} 页，"
                                        f"仍需渲染 {len(pending_render)} 页[/]"
                                    )
                    else:
                        render_cache = None

                    save_counter = 0
                    if pending_render:
                        with PDFRenderer(self.pdf_path, dpi=self.dpi) as renderer:
                            for page_num in pending_render:
                                img = renderer.render_page(page_num)
                                buf = io.BytesIO()
                                img.convert("RGB").save(buf, format="JPEG", quality=82)
                                jpeg_bytes = buf.getvalue()
                                ckpt.page_jpeg_bytes[page_num] = jpeg_bytes
                                # 同步写入渲染缓存对象（延迟持久化）
                                if self.cache_render:
                                    if render_cache is None:
                                        render_cache = RenderCache(
                                            total,
                                            os.path.getmtime(self.pdf_path),
                                            self.dpi,
                                        )
                                    render_cache.page_jpeg_bytes[page_num] = jpeg_bytes
                                progress.advance(render_task)
                                save_counter += 1
                                if save_counter >= _CKPT_SAVE_INTERVAL:
                                    ckpt.save(ckpt_path)
                                    save_counter = 0
                    ckpt.save(ckpt_path)
                    # 渲染阶段全部完成后持久化渲染缓存
                    if self.cache_render and render_cache is not None:
                        render_cache.save(self._render_cache_path())
                        if self.verbose:
                            _console.print(
                                f"[dim]  渲染缓存已保存至 {self._render_cache_path()}[/]"
                            )
                except KeyboardInterrupt:
                    _console.print("\n[yellow]正在保存断点…[/]")
                    ckpt.save(ckpt_path)
                    # 中断时也保存已完成部分的渲染缓存
                    if self.cache_render and render_cache is not None:
                        with suppress(OSError):
                            render_cache.save(self._render_cache_path())
                    done = sum(1 for b in ckpt.page_jpeg_bytes if b is not None)
                    _console.print(
                        f"[yellow]断点已保存至 [bold]{ckpt_path}[/bold]，"
                        f"已渲染 {done}/{total} 页。\n"
                        f"下次运行相同命令可自动继续。[/]"
                    )
                    raise

            # ── 阶段 1：并行 OCR（跳过已完成的页面）────────────────────
            pending_ocr = ckpt.pending_ocr_pages()
            already_done = ckpt.completed_ocr_count()
            workers_label = f" × {self.workers} 线程" if self.workers > 1 else ""
            ocr_task = progress.add_task(
                f"[cyan]阶段 1[/]  OCR 识别{workers_label}",
                total=total,
                completed=already_done,
            )

            if pending_ocr:
                try:
                    # 尝试从 OCR 缓存中补充已识别的页面
                    if self.cache_ocr:
                        ocr_cache = self._load_ocr_cache()
                        if ocr_cache is not None:
                            hit_pages = []
                            for page_num in list(pending_ocr):
                                cached_result = ocr_cache.ocr_results[page_num]
                                if cached_result is not None:
                                    ckpt.ocr_results[page_num] = cached_result
                                    hit_pages.append(page_num)
                            if hit_pages:
                                pending_ocr = ckpt.pending_ocr_pages()
                                already_done = ckpt.completed_ocr_count()
                                progress.update(ocr_task, completed=already_done)
                                if self.verbose:
                                    _console.print(
                                        f"[dim]  OCR 缓存命中 {len(hit_pages)} 页，"
                                        f"仍需识别 {len(pending_ocr)} 页[/]"
                                    )
                    else:
                        ocr_cache = None

                    if pending_ocr:
                        with ThreadPoolExecutor(max_workers=self.workers) as pool:
                            futures = {
                                pool.submit(
                                    _ocr_worker, i, ckpt.page_jpeg_bytes[i], self.lang
                                ): i
                                for i in pending_ocr
                            }
                            save_counter = 0
                            for fut in as_completed(futures):
                                page_num, result = fut.result()
                                ckpt.ocr_results[page_num] = result
                                # 同步写入 OCR 缓存对象（延迟持久化）
                                if self.cache_ocr:
                                    if ocr_cache is None:
                                        ocr_cache = OcrCache(
                                            total,
                                            os.path.getmtime(self.pdf_path),
                                            self.lang,
                                        )
                                    ocr_cache.ocr_results[page_num] = result
                                progress.advance(ocr_task)
                                save_counter += 1
                                if save_counter >= _CKPT_SAVE_INTERVAL:
                                    ckpt.save(ckpt_path)
                                    save_counter = 0
                    # 全部完成后再保存一次，确保最后几页也持久化
                    ckpt.save(ckpt_path)
                    # OCR 阶段全部完成后持久化 OCR 缓存
                    if self.cache_ocr and ocr_cache is not None:
                        ocr_cache.save(self._ocr_cache_path())
                        if self.verbose:
                            _console.print(
                                f"[dim]  OCR 缓存已保存至 {self._ocr_cache_path()}[/]"
                            )
                except KeyboardInterrupt:
                    _console.print("\n[yellow]正在保存断点…[/]")
                    ckpt.save(ckpt_path)
                    # 中断时也保存已完成部分的 OCR 缓存
                    if self.cache_ocr and ocr_cache is not None:
                        with suppress(OSError):
                            ocr_cache.save(self._ocr_cache_path())
                    done = ckpt.completed_ocr_count()
                    _console.print(
                        f"[yellow]断点已保存至 [bold]{ckpt_path}[/bold]，"
                        f"已完成 {done}/{total} 页。\n"
                        f"下次运行相同命令可自动继续。[/]"
                    )
                    raise

            # ── 阶段 2：版面分析 ─────────────────────────────────────────
            hf_filter = HeaderFooterFilter()
            for result in ckpt.ocr_results:
                hf_filter.collect(result)  # type: ignore[arg-type]
            hf_filter.finalize()
            if self.verbose:
                _console.print(
                    f"[dim]  检测到 {len(hf_filter._excluded)} 条页眉/页脚模板[/]"
                )

            analyze_task = progress.add_task(
                "[cyan]阶段 2[/]  版面分析", total=total
            )
            raw_page_elements: List[List[DocumentElement]] = []

            for page_num in range(total):
                page_image = Image.open(io.BytesIO(ckpt.page_jpeg_bytes[page_num]))  # type: ignore[arg-type]
                elements = analyzer.analyze_page(
                    ckpt.ocr_results[page_num], page_image, hf_filter  # type: ignore[arg-type]
                )
                raw_page_elements.append(elements)
                progress.advance(analyze_task)

            progress.add_task("[cyan]后处理[/]  跨页段落合并", total=None)

        # ── 按 H1 标题切分章节 ────────────────────────────────────────────
        merged = _merge_cross_page_paragraphs(raw_page_elements)

        document = Document(title=self.title, author=self.author)
        current_chapter = Chapter(title="")
        document.chapters.append(current_chapter)

        for elem in merged:
            if (
                isinstance(elem, TextBlock)
                and elem.is_heading
                and elem.heading_level == HeadingLevel.H1
            ):
                current_chapter = Chapter(title=elem.text)
                document.chapters.append(current_chapter)

            current_chapter.elements.append(elem)

        document.chapters = [c for c in document.chapters if c.elements]
        if not document.chapters:
            document.chapters = [Chapter(title=self.title)]

        # 全流程成功完成，删除断点文件
        with suppress(OSError):
            os.unlink(ckpt_path)

        return document

    # ──────────────────────────────────────────────────────────────────────
    def _checkpoint_path(self) -> str:
        """断点文件路径：<output_dir>/.<stem>.ckpt"""
        stem = Path(self.pdf_path).stem
        return os.path.join(self.output_dir, f".{stem}.ckpt")

    def _render_cache_path(self) -> str:
        """渲染缓存文件路径：<output_dir>/.<stem>.render.ckpt"""
        stem = Path(self.pdf_path).stem
        return os.path.join(self.output_dir, f".{stem}.render.ckpt")

    def _ocr_cache_path(self) -> str:
        """OCR 缓存文件路径：<output_dir>/.<stem>.ocr.ckpt"""
        stem = Path(self.pdf_path).stem
        return os.path.join(self.output_dir, f".{stem}.ocr.ckpt")

    def _load_ocr_cache(self) -> "Optional[OcrCache]":
        """尝试加载 OCR 缓存；缓存不存在、版本不符或参数不一致时返回 None。"""
        cache_path = self._ocr_cache_path()
        if not os.path.exists(cache_path):
            return None
        try:
            cache = OcrCache.load(cache_path)
            pdf_mtime = os.path.getmtime(self.pdf_path)
            if abs(cache.pdf_mtime - pdf_mtime) > 1.0:
                raise ValueError("PDF 文件已被修改（mtime 变化）")
            if cache.lang != self.lang:
                raise ValueError(f"语言不符：缓存 {cache.lang!r} ≠ 当前 {self.lang!r}")
            with PDFRenderer(self.pdf_path, self.dpi) as r:
                current_total = len(r)
            if cache.total != current_total:
                raise ValueError(
                    f"缓存记录 {cache.total} 页 ≠ 当前 PDF {current_total} 页"
                )
            done = cache.completed_count()
            _console.print(
                f"[yellow]发现 OCR 缓存[/]：已缓存 [bold]{done}[/bold]/{cache.total} 页，"
                f"语言={cache.lang!r}。\n"
                f"[dim]（缓存文件：{cache_path}）[/]"
            )
            return cache
        except Exception as exc:
            _console.print(f"[red]OCR 缓存无效（{exc}），已忽略。[/]")
            with suppress(OSError):
                os.unlink(cache_path)
            return None

    def _load_render_cache(self) -> "Optional[RenderCache]":
        """尝试加载渲染缓存；缓存不存在、版本不符或参数不一致时返回 None。"""
        cache_path = self._render_cache_path()
        if not os.path.exists(cache_path):
            return None
        try:
            cache = RenderCache.load(cache_path)
            pdf_mtime = os.path.getmtime(self.pdf_path)
            if abs(cache.pdf_mtime - pdf_mtime) > 1.0:
                raise ValueError("PDF 文件已被修改（mtime 变化）")
            if cache.dpi != self.dpi:
                raise ValueError(f"DPI 不符：缓存 {cache.dpi} ≠ 当前 {self.dpi}")
            with PDFRenderer(self.pdf_path, self.dpi) as r:
                current_total = len(r)
            if cache.total != current_total:
                raise ValueError(
                    f"缓存记录 {cache.total} 页 ≠ 当前 PDF {current_total} 页"
                )
            done = sum(1 for b in cache.page_jpeg_bytes if b is not None)
            _console.print(
                f"[yellow]发现渲染缓存[/]：已缓存 [bold]{done}[/bold]/{cache.total} 页，"
                f"DPI={cache.dpi}。\n"
                f"[dim]（缓存文件：{cache_path}）[/]"
            )
            return cache
        except Exception as exc:
            _console.print(f"[red]渲染缓存无效（{exc}），已忽略。[/]")
            with suppress(OSError):
                os.unlink(cache_path)
            return None

    def _load_or_create_checkpoint(self, ckpt_path: str) -> Checkpoint:
        """加载已有断点，或在以下情况下新建：--fresh、文件不存在、文件损坏。"""
        pdf_mtime = os.path.getmtime(self.pdf_path)

        if self.fresh:
            with suppress(OSError):
                os.unlink(ckpt_path)
            if self.cache_render:
                with suppress(OSError):
                    os.unlink(self._render_cache_path())
            if self.cache_ocr:
                with suppress(OSError):
                    os.unlink(self._ocr_cache_path())
            if self.verbose:
                _console.print("[dim]  --fresh：跳过断点，从头开始。[/]")
        elif os.path.exists(ckpt_path):
            try:
                ckpt = Checkpoint.load(ckpt_path)
                # 验证：PDF 修改时间和页数必须一致
                if abs(ckpt.pdf_mtime - pdf_mtime) > 1.0:
                    raise ValueError("PDF 文件已被修改（mtime 变化）")
                with PDFRenderer(self.pdf_path, self.dpi) as r:
                    current_total = len(r)
                if ckpt.total != current_total:
                    raise ValueError(
                        f"断点记录 {ckpt.total} 页 ≠ 当前 PDF {current_total} 页"
                    )
                done = ckpt.completed_ocr_count()
                _console.print(
                    f"[yellow]发现断点[/]：已完成 [bold]{done}[/bold]/{ckpt.total} 页 OCR，继续处理。\n"
                    f"[dim]（断点文件：{ckpt_path}）[/]"
                )
                return ckpt
            except Exception as exc:
                _console.print(
                    f"[red]断点文件无效（{exc}），已忽略，从头开始。[/]"
                )
                with suppress(OSError):
                    os.unlink(ckpt_path)

        with PDFRenderer(self.pdf_path, self.dpi) as r:
            total = len(r)
        return Checkpoint(total, pdf_mtime)
