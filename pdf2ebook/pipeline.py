"""
转换流水线：PDF → Document → EPUB / Typst

流程（v2 两遍扫描）：
  第一遍 — OCR：逐页渲染并识别文字，缓存 OCRResult 和 JPEG 图像字节；
           同时向 HeaderFooterFilter 收集页眉/页脚候选文本。
  第二遍 — 版面分析：利用缓存数据重建元素列表（标题/段落/图片），
           HeaderFooterFilter 已 finalize()，过滤精度更高。
  后处理 — 跨页段落合并：检测上一页末尾未完结的段落，与下一页首段合并。
  导出   — 调用对应导出器输出 EPUB / Typst 文件。
"""

import io
import os
import sys
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
    ) -> None:
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.output_format = output_format.lower()
        self.lang = lang
        self.dpi = dpi
        self.verbose = verbose

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
            f"输出目录：[dim]{self.output_dir}[/]  |  格式：[yellow]{self.output_format}[/]  |  DPI：{self.dpi}",
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
        ocr = OCRProcessor(lang=self.lang)
        analyzer = LayoutAnalyzer()

        # 构造通用进度条样式
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
            # ── 第一遍：OCR + 缓存页面图像字节 + 收集页眉/页脚候选 ────────
            with PDFRenderer(self.pdf_path, dpi=self.dpi) as renderer:
                total = len(renderer)

            if self.verbose:
                _console.print(f"[dim]  PDF 共 {total} 页，DPI={self.dpi}，语言={self.lang}[/]")

            ocr_task = progress.add_task(
                "[cyan]第一遍[/]  OCR 识别", total=total
            )

            ocr_results: List[OCRResult] = []
            page_jpeg_bytes: List[bytes] = []
            hf_filter = HeaderFooterFilter()

            with PDFRenderer(self.pdf_path, dpi=self.dpi) as renderer:
                for _page_num, page_image in renderer.iter_pages():
                    result = ocr.process(page_image)
                    ocr_results.append(result)
                    hf_filter.collect(result)

                    buf = io.BytesIO()
                    page_image.convert("RGB").save(buf, format="JPEG", quality=82)
                    page_jpeg_bytes.append(buf.getvalue())

                    progress.advance(ocr_task)

            hf_filter.finalize()
            if self.verbose:
                _console.print(
                    f"[dim]  检测到 {len(hf_filter._excluded)} 条页眉/页脚模板[/]"
                )

            # ── 第二遍：版面分析 ─────────────────────────────────────────
            analyze_task = progress.add_task(
                "[cyan]第二遍[/]  版面分析", total=total
            )

            raw_page_elements: List[List[DocumentElement]] = []

            for _page_num, (ocr_result, jpeg_bytes) in enumerate(
                zip(ocr_results, page_jpeg_bytes)
            ):
                page_image = Image.open(io.BytesIO(jpeg_bytes))
                elements = analyzer.analyze_page(ocr_result, page_image, hf_filter)
                raw_page_elements.append(elements)
                progress.advance(analyze_task)

            # ── 跨页段落合并（无 I/O，无需进度条）────────────────────────
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
