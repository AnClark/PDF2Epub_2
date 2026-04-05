"""
布局分析器：根据 OCR 结果重建文档结构。

核心任务：
  1. 检测标题（根据字高比例）
  2. 将多行合并为自然段（根据缩进规则）
  3. 检测整页插图
  4. 过滤页码等干扰内容
"""

import io
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .document_model import DocumentElement, HeadingLevel, ImageBlock, TextBlock
from .ocr_processor import OCRLine, OCRResult

# 中文字符间的空白（Tesseract 有时在汉字间插入空格）
_CJK_SPACE_RE = re.compile(
    r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])"
    r"\s+"
    r"(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])"
)


def _remove_cjk_spaces(text: str) -> str:
    return _CJK_SPACE_RE.sub("", text)


def _is_page_number(text: str) -> bool:
    """判断是否为页码行：仅含数字、连字符、间隔号等。"""
    return bool(re.fullmatch(r"[\d\s\-—–·•··○◎]+", text.strip()))


def _classify_heading_level(height_ratio: float) -> Optional[HeadingLevel]:
    """根据行高与正文行高之比判断标题级别。"""
    if height_ratio >= 2.5:
        return HeadingLevel.H1
    if height_ratio >= 2.0:
        return HeadingLevel.H2
    if height_ratio >= 1.5:
        return HeadingLevel.H3
    return None


def _page_image_to_block(image: Image.Image) -> ImageBlock:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return ImageBlock(
        image_data=buf.getvalue(),
        mime_type="image/jpeg",
        width=image.width,
        height=image.height,
    )


class LayoutAnalyzer:
    """
    分析单页 OCR 结果，提取文档元素列表（标题、段落、图片）。

    段落重建算法（适用于简体中文横排书籍）：
      - 中文段落首行缩进约 2 个字符宽（即首行 left > 左边距 + 阈值）
      - 非首行紧贴左边距
      - 前一行行尾与右边距距离 > 2.5 字符宽 → 该行是段落末行
      - 大垂直间距 → 强制分段
    """

    def analyze_page(
        self,
        ocr_result: OCRResult,
        page_image: Optional[Image.Image] = None,
    ) -> List[DocumentElement]:
        lines = ocr_result.get_lines()

        # 无文字 → 整页插图
        if not lines:
            if page_image is not None:
                return [_page_image_to_block(page_image)]
            return []

        # 文字极少 → 也当图片处理（图片页带少量说明文字的情况先跳过）
        if ocr_result.total_char_count() < 30 and page_image is not None:
            return [_page_image_to_block(page_image)]

        median_h = ocr_result.get_median_line_height()
        page_h = ocr_result.page_height
        page_w = ocr_result.page_width

        # ── 1. 过滤页眉页脚/页码 ──────────────────────────────────────────
        margin_zone = median_h * 2.0
        body_lines: List[OCRLine] = []
        for line in lines:
            txt = line.text.strip()
            if not txt:
                continue
            at_edge = line.top < margin_zone or line.bottom > page_h - margin_zone
            if at_edge and _is_page_number(txt):
                continue
            body_lines.append(line)

        if not body_lines:
            return []

        # ── 2. 估算版心左右边距 ────────────────────────────────────────────
        lefts = [l.left for l in body_lines]
        rights = [l.right for l in body_lines]
        text_left = float(np.percentile(lefts, 10))   # 左边距
        text_right = float(np.percentile(rights, 90))  # 右边距

        # 中文正方形字符：字宽 ≈ 行高
        char_w = median_h
        indent_thresh = char_w * 1.2    # 首行缩进判断阈值
        short_line_gap = char_w * 2.5   # 行尾距右边距 > 此值 → 段末短行

        # ── 3. 逐行分析 ────────────────────────────────────────────────────
        elements: List[DocumentElement] = []
        para_lines: List[OCRLine] = []
        prev: Optional[OCRLine] = None

        def flush() -> None:
            if not para_lines:
                return
            text = _remove_cjk_spaces("".join(l.text.strip() for l in para_lines))
            text = text.strip()
            if text:
                elements.append(TextBlock(text=text))
            para_lines.clear()

        for line in body_lines:
            txt = line.text.strip()
            if not txt:
                continue

            # 大间距 → 强制分段
            if prev is not None and (line.top - prev.bottom) > median_h * 2.5:
                flush()

            # 标题检测（行高比）
            h_ratio = line.height / median_h if median_h > 0 else 1.0
            h_level = _classify_heading_level(h_ratio)

            # 补充判断：短居中行作为三级标题
            if h_level is None and len(txt) <= 20:
                line_cx = line.left + line.width / 2
                if abs(line_cx - page_w / 2) < page_w * 0.12:
                    h_level = HeadingLevel.H3

            if h_level is not None:
                flush()
                elements.append(TextBlock(text=txt, is_heading=True, heading_level=h_level))
                prev = line
                continue

            # ── 段落边界判断 ──────────────────────────────────────────────
            new_para = False
            if not para_lines:
                new_para = True
            elif (line.left - text_left) > indent_thresh:
                # 首行缩进 → 新段落
                new_para = True
            elif prev is not None:
                # 前一行是短行（段末） → 当前行开始新段落
                gap_to_right = text_right - prev.right
                if gap_to_right > short_line_gap:
                    new_para = True

            if new_para and para_lines:
                flush()

            para_lines.append(line)
            prev = line

        flush()
        return elements
