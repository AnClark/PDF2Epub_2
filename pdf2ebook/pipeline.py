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

from .document_model import (
    Chapter, Document, DocumentElement, HeadingLevel, ImageBlock, TextBlock
)
from .exporters.epub_exporter import EPUBExporter
from .exporters.typst_exporter import TypstExporter
from .layout_analyzer import HeaderFooterFilter, LayoutAnalyzer, SENTENCE_END_RE
from .ocr_processor import OCRProcessor, OCRResult
from .pdf_renderer import PDFRenderer


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

        self._log(f"开始处理：{self.pdf_path}")
        document = self._build_document()
        self._log(f"共识别 {sum(len(c.elements) for c in document.chapters)} 个元素，"
                  f"{len(document.chapters)} 个章节")

        if self.output_format in ("typst", "both"):
            self._log(f"导出 Typst → {self.typst_path}")
            TypstExporter().export(document, self.typst_path)

        if self.output_format in ("epub", "both"):
            self._log(f"导出 EPUB  → {self.epub_path}")
            EPUBExporter().export(document, self.epub_path)

        self._log("完成。")

    # ──────────────────────────────────────────────────────────────────────
    def _build_document(self) -> Document:
        ocr = OCRProcessor(lang=self.lang)
        analyzer = LayoutAnalyzer()

        # ── 第一遍：OCR + 缓存页面图像字节 + 收集页眉/页脚候选 ────────────
        self._log("第一遍：OCR 识别 …")
        ocr_results: List[OCRResult] = []
        page_jpeg_bytes: List[bytes] = []
        hf_filter = HeaderFooterFilter()

        with PDFRenderer(self.pdf_path, dpi=self.dpi) as renderer:
            total = len(renderer)
            self._log(f"共 {total} 页")
            for page_num, page_image in renderer.iter_pages():
                self._log(f"  OCR 第 {page_num + 1}/{total} 页 …", end="\r")
                result = ocr.process(page_image)
                ocr_results.append(result)
                hf_filter.collect(result)

                # 将页面图像压缩为 JPEG 字节缓存（约 200–400 KB/页）
                buf = io.BytesIO()
                page_image.convert("RGB").save(buf, format="JPEG", quality=82)
                page_jpeg_bytes.append(buf.getvalue())

        self._log("")
        hf_filter.finalize()
        self._log(f"  检测到 {len(hf_filter._excluded)} 条页眉/页脚模板")

        # ── 第二遍：版面分析（使用已 finalize 的 hf_filter）────────────────
        self._log("第二遍：版面分析 …")
        raw_page_elements: List[List[DocumentElement]] = []

        for page_num, (ocr_result, jpeg_bytes) in enumerate(
            zip(ocr_results, page_jpeg_bytes)
        ):
            self._log(f"  分析第 {page_num + 1}/{total} 页 …", end="\r")
            page_image = Image.open(io.BytesIO(jpeg_bytes))
            elements = analyzer.analyze_page(ocr_result, page_image, hf_filter)
            raw_page_elements.append(elements)

        self._log("")

        # ── 后处理：跨页段落合并 ─────────────────────────────────────────
        merged = _merge_cross_page_paragraphs(raw_page_elements)

        # ── 按 H1 标题切分章节 ────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    def _log(self, msg: str, end: str = "\n") -> None:
        if self.verbose or end == "\n":
            print(msg, end=end, flush=True)
