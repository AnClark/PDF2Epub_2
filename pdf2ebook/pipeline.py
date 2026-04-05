"""
转换流水线：PDF → Document → EPUB / Typst

流程：
  1. 用 PyMuPDF 逐页渲染 PDF
  2. 用 Tesseract OCR 识别每页文字
  3. 用 LayoutAnalyzer 还原标题/段落/图片结构
  4. 在 H1 标题处切分章节
  5. 调用对应导出器输出文件
"""

import os
import sys
from pathlib import Path
from typing import Optional

from .document_model import Chapter, Document, HeadingLevel, ImageBlock, TextBlock
from .exporters.epub_exporter import EPUBExporter
from .exporters.typst_exporter import TypstExporter
from .layout_analyzer import LayoutAnalyzer
from .ocr_processor import OCRProcessor
from .pdf_renderer import PDFRenderer


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

        document = Document(title=self.title, author=self.author)
        # 第一个章节作为"前言/引言"承载首个 H1 前的内容
        current_chapter = Chapter(title="")
        document.chapters.append(current_chapter)

        with PDFRenderer(self.pdf_path, dpi=self.dpi) as renderer:
            total = len(renderer)
            self._log(f"共 {total} 页")

            for page_num, page_image in renderer.iter_pages():
                self._log(f"  OCR 第 {page_num + 1}/{total} 页 …", end="\r")

                ocr_result = ocr.process(page_image)
                elements = analyzer.analyze_page(ocr_result, page_image)

                for elem in elements:
                    # 遇到一级标题 → 开启新章节
                    if (
                        isinstance(elem, TextBlock)
                        and elem.is_heading
                        and elem.heading_level == HeadingLevel.H1
                    ):
                        current_chapter = Chapter(title=elem.text)
                        document.chapters.append(current_chapter)
                        current_chapter.elements.append(elem)
                        continue

                    current_chapter.elements.append(elem)

            self._log("")  # 换行

        # 移除空章节（可能是开头无内容的占位章节）
        document.chapters = [c for c in document.chapters if c.elements]
        if not document.chapters:
            document.chapters = [Chapter(title=self.title)]

        return document

    # ──────────────────────────────────────────────────────────────────────
    def _log(self, msg: str, end: str = "\n") -> None:
        if self.verbose or end == "\n":
            print(msg, end=end, flush=True)
