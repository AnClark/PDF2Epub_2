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
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ) -> None:
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.output_format = output_format.lower()
        self.lang = lang
        self.dpi = dpi
        self.verbose = verbose
        self.workers = max(1, workers)

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
        analyzer = LayoutAnalyzer()

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
            # ── 阶段 0：串行渲染所有页面为 JPEG 字节缓存 ─────────────────
            with PDFRenderer(self.pdf_path, dpi=self.dpi) as renderer:
                total = len(renderer)

            if self.verbose:
                _console.print(
                    f"[dim]  PDF 共 {total} 页，DPI={self.dpi}，"
                    f"语言={self.lang}，OCR 线程={self.workers}[/]"
                )

            render_task = progress.add_task("[cyan]阶段 0[/]  渲染页面", total=total)
            page_jpeg_bytes: List[bytes] = []

            with PDFRenderer(self.pdf_path, dpi=self.dpi) as renderer:
                for _page_num, page_image in renderer.iter_pages():
                    buf = io.BytesIO()
                    page_image.convert("RGB").save(buf, format="JPEG", quality=82)
                    page_jpeg_bytes.append(buf.getvalue())
                    progress.advance(render_task)

            # ── 阶段 1：并行 OCR ──────────────────────────────────────────
            workers_label = f" × {self.workers} 线程" if self.workers > 1 else ""
            ocr_task = progress.add_task(
                f"[cyan]阶段 1[/]  OCR 识别{workers_label}", total=total
            )

            ocr_results: List[Optional[OCRResult]] = [None] * total

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(_ocr_worker, i, jpeg, self.lang): i
                    for i, jpeg in enumerate(page_jpeg_bytes)
                }
                for fut in as_completed(futures):
                    page_num, result = fut.result()
                    ocr_results[page_num] = result
                    progress.advance(ocr_task)

            # 收集页眉/页脚候选并 finalize
            hf_filter = HeaderFooterFilter()
            for result in ocr_results:
                hf_filter.collect(result)  # type: ignore[arg-type]
            hf_filter.finalize()
            if self.verbose:
                _console.print(
                    f"[dim]  检测到 {len(hf_filter._excluded)} 条页眉/页脚模板[/]"
                )

            # ── 阶段 2：串行版面分析 ──────────────────────────────────────
            analyze_task = progress.add_task(
                "[cyan]阶段 2[/]  版面分析", total=total
            )
            raw_page_elements: List[List[DocumentElement]] = []

            for _page_num, (ocr_result, jpeg_bytes) in enumerate(
                zip(ocr_results, page_jpeg_bytes)
            ):
                page_image = Image.open(io.BytesIO(jpeg_bytes))
                elements = analyzer.analyze_page(
                    ocr_result, page_image, hf_filter  # type: ignore[arg-type]
                )
                raw_page_elements.append(elements)
                progress.advance(analyze_task)

            # ── 后处理：跨页段落合并 ──────────────────────────────────────
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

        return document
