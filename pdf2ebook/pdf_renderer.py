"""PDF页面渲染器：使用 PyMuPDF 将 PDF 页面渲染为 PIL 图像。"""

import io
from typing import Iterator, Tuple

import fitz  # pymupdf
from PIL import Image


class PDFRenderer:
    """将 PDF 各页渲染为高分辨率 PIL Image。"""

    def __init__(self, pdf_path: str, dpi: int = 300) -> None:
        self.pdf_path = pdf_path
        self.dpi = dpi
        self._doc = fitz.open(pdf_path)

    def __len__(self) -> int:
        return len(self._doc)

    def __enter__(self) -> "PDFRenderer":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._doc.close()

    def render_page(self, page_num: int) -> Image.Image:
        """将指定页码渲染为 PIL Image。"""
        page = self._doc[page_num]
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    def iter_pages(self) -> Iterator[Tuple[int, Image.Image]]:
        """逐页渲染，返回 (页码, PIL Image) 迭代器。"""
        for i in range(len(self)):
            yield i, self.render_page(i)
