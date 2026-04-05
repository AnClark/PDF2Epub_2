"""
布局分析器：根据 OCR 结果重建文档结构。（v2）

核心改进：
  1. 启发式多因子标题识别（关键词模式 + 行高比 + 居中 + 留白评分）
  2. HeaderFooterFilter：跨页统计过滤重复页眉/页脚
  3. 更准确的自然段边界检测（首行缩进 + 行尾短行双规则）
  4. 整页插图检测
"""

import io
import re
from collections import Counter
from typing import List, Optional, Set

import numpy as np
from PIL import Image

from .document_model import DocumentElement, HeadingLevel, ImageBlock, TextBlock
from .ocr_processor import OCRLine, OCRResult

# ─── 文本规范化 ────────────────────────────────────────────────────────────
_CJK_SPACE_RE = re.compile(
    r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])"
    r"\s+"
    r"(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])"
)
_DIGIT_NORM_RE = re.compile(r"\d+")
_PAGE_NUM_RE = re.compile(r"^[\d\s\-—–·•·○◎]+$")

# 句末标点：用于跨页段落合并判断（供 pipeline.py 使用）
SENTENCE_END_RE = re.compile(r"[。！？…」』）】!?\u201d\u2019]+\s*$")

# ─── 标题关键词（re.match → 从行首匹配，text 已 strip）─────────────────────
# H1：章/篇/部/卷/编 及前言、后记等特殊节点
_H1_KW = re.compile(
    r"^(第\s*[零一二三四五六七八九十百千万\d]+\s*[章篇部卷编]"
    r"|[上中下]\s*[册篇卷]"
    r"|序\s*言?|前\s*言|引\s*言|绪\s*[论言]"
    r"|附\s*录\s*[A-Za-z一二三四五六七八九十]?"
    r"|结\s*[论语]|后\s*记|跋"
    r"|参\s*考\s*文\s*献|索\s*引|目\s*录"
    r"|致\s*谢|鸣\s*谢|内\s*容\s*简\s*介"
    r"|作\s*者\s*简\s*介)$",
    re.UNICODE,
)
# H2：节/条/讲，以及"一、…"和"1. …"格式
_H2_KW = re.compile(
    r"^(第\s*[零一二三四五六七八九十百千\d]+\s*[节条讲]"
    r"|[一二三四五六七八九十]+[、．.]\s*\S"
    r"|\d{1,2}[．.]\s*[^\d\s]\S*$)",
    re.UNICODE,
)
# H3：括号序号、小数点序号、带圈数字
_H3_KW = re.compile(
    r"^([（(][一二三四五六七八九十\d]+[）)]"
    r"|\d{1,2}[．.]\d{1,2}"
    r"|[①②③④⑤⑥⑦⑧⑨⑩])",
    re.UNICODE,
)


def _norm_text(text: str) -> str:
    """去除 CJK 字符间多余空格并 strip。"""
    return _CJK_SPACE_RE.sub("", text.strip())


def _is_page_number(text: str) -> bool:
    return bool(_PAGE_NUM_RE.match(text.strip()))


def _page_image_to_block(image: Image.Image) -> ImageBlock:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return ImageBlock(
        image_data=buf.getvalue(),
        mime_type="image/jpeg",
        width=image.width,
        height=image.height,
    )


# ─── HeaderFooterFilter ───────────────────────────────────────────────────────
class HeaderFooterFilter:
    """
    跨页统计过滤器：收集多页顶部/底部文本的出现频率，将高频文本判定为
    页眉/页脚并在 analyze_page 时过滤掉。

    使用方式（两阶段）：
      1. 对每页 OCR 结果调用 collect()
      2. 调用 finalize() 计算阈值
      3. 在 analyze_page 时传入 hf_filter，用 is_header_footer() 过滤
    """

    #: 页高的多少比例算作"页眉/页脚区域"（用于统计收集）
    MARGIN_RATIO: float = 0.10
    #: 出现在多少比例的页面上才算页眉/页脚
    FREQ_THRESHOLD: float = 0.25

    def __init__(self) -> None:
        self._top_counter: Counter = Counter()
        self._bottom_counter: Counter = Counter()
        self._page_count: int = 0
        self._excluded: Set[str] = set()

    @staticmethod
    def _norm(text: str) -> str:
        """规范化：转小写、去首尾空格、将连续数字串替换为 '#'（消除页码差异）。"""
        return _DIGIT_NORM_RE.sub("#", text.strip().lower())

    def collect(self, ocr_result: OCRResult) -> None:
        """收集单页顶部/底部区域的文本，更新统计计数。"""
        self._page_count += 1
        margin = ocr_result.page_height * self.MARGIN_RATIO
        for line in ocr_result.get_lines():
            txt = line.text.strip()
            if not txt:
                continue
            nd = self._norm(txt)
            if line.top < margin:
                self._top_counter[nd] += 1
            elif line.bottom > ocr_result.page_height - margin:
                self._bottom_counter[nd] += 1

    def finalize(self) -> None:
        """根据频率阈值确定要过滤的规范化文本集合。"""
        if self._page_count == 0:
            return
        min_count = max(2, int(self._page_count * self.FREQ_THRESHOLD))
        for nd, cnt in self._top_counter.items():
            if cnt >= min_count:
                self._excluded.add(nd)
        for nd, cnt in self._bottom_counter.items():
            if cnt >= min_count:
                self._excluded.add(nd)

    def is_header_footer(self, line: OCRLine, page_height: int) -> bool:
        """判断该行是否为页眉/页脚（应被过滤）。"""
        # 检测区域：上/下各 20% 页高（稍宽于收集区域以增强覆盖）
        margin = page_height * self.MARGIN_RATIO * 2.0
        at_edge = (line.top < margin) or (line.bottom > page_height - margin)
        if not at_edge:
            return False
        txt = line.text.strip()
        if _is_page_number(txt):
            return True
        return self._norm(txt) in self._excluded


# ─── 启发式标题检测 ───────────────────────────────────────────────────────────
def _detect_heading(
    line: OCRLine,
    median_h: float,
    page_w: int,
    prev_gap: float = 0.0,
    next_gap: float = 0.0,
) -> Optional[HeadingLevel]:
    """
    多因子标题检测，返回标题级别，或 None（非标题）。

    优先级：
      1. 关键词模式匹配（H1/H2/H3 关键词，最可信）
      2. 行高比阈值（字体明显大于正文）
      3. 短文本 + 居中对齐 + 上下留白综合评分（≥ 55 分时判定为 H3）
    """
    txt = line.text.strip()
    if not txt:
        return None

    # 1. 关键词模式（快速路径）
    if _H1_KW.match(txt):
        return HeadingLevel.H1
    if _H2_KW.match(txt):
        return HeadingLevel.H2
    if _H3_KW.match(txt):
        return HeadingLevel.H3

    # 2. 行高比
    h_ratio = (line.height / median_h) if median_h > 0 else 1.0
    if h_ratio >= 2.5:
        return HeadingLevel.H1
    if h_ratio >= 1.8:
        return HeadingLevel.H2
    if h_ratio >= 1.35:
        return HeadingLevel.H3

    # 3. 短文本 + 居中 + 留白评分
    char_count = len(txt)
    if char_count > 30:
        return None

    line_cx = line.left + line.width / 2.0
    center_offset = abs(line_cx - page_w / 2.0) / max(page_w, 1)
    is_centered = center_offset < 0.15

    score = 0
    if is_centered:
        score += 25
    if char_count <= 20:
        score += 15
    if char_count <= 10:
        score += 15
    if prev_gap > median_h * 1.5:
        score += 15
    if next_gap > median_h * 1.5:
        score += 15

    if score >= 55:
        return HeadingLevel.H3
    return None


# ─── LayoutAnalyzer ───────────────────────────────────────────────────────────
class LayoutAnalyzer:
    """
    分析单页 OCR 结果，提取文档元素列表（标题、段落、图片）。

    段落重建规则（适用于简体中文横排书籍）：
      - 首行相对版心左边距缩进 > 1.2 字符宽 → 新段落开始
      - 前一行尾部距版心右边距 > 2.5 字符宽 → 前一行为段末短行
      - 行间垂直间距 > 2.5 倍行高 → 强制分段
    """

    def analyze_page(
        self,
        ocr_result: OCRResult,
        page_image: Optional[Image.Image] = None,
        hf_filter: Optional[HeaderFooterFilter] = None,
    ) -> List[DocumentElement]:
        lines = ocr_result.get_lines()

        # 无文字 → 整页插图
        if not lines:
            if page_image is not None:
                return [_page_image_to_block(page_image)]
            return []

        # 文字极少 → 也当图片处理
        if ocr_result.total_char_count() < 30 and page_image is not None:
            return [_page_image_to_block(page_image)]

        median_h = ocr_result.get_median_line_height()
        page_h = ocr_result.page_height
        page_w = ocr_result.page_width

        # ── 1. 过滤页眉/页脚/页码 ──────────────────────────────────────────
        body_lines: List[OCRLine] = []
        for line in lines:
            txt = line.text.strip()
            if not txt:
                continue
            if hf_filter is not None:
                if hf_filter.is_header_footer(line, page_h):
                    continue
            else:
                # 退化模式：仅过滤边缘纯页码
                margin_zone = median_h * 2.0
                at_edge = line.top < margin_zone or line.bottom > page_h - margin_zone
                if at_edge and _is_page_number(txt):
                    continue
            body_lines.append(line)

        if not body_lines:
            return []

        # ── 2. 估算版心左右边距 ────────────────────────────────────────────
        lefts = [l.left for l in body_lines]
        rights = [l.right for l in body_lines]
        text_left = float(np.percentile(lefts, 10))
        text_right = float(np.percentile(rights, 90))
        char_w = median_h
        indent_thresh = char_w * 1.2
        short_line_gap = char_w * 2.5

        # ── 3. 预计算相邻行间距（供标题检测使用）──────────────────────────
        n = len(body_lines)
        gaps_above = [0.0] * n
        gaps_below = [0.0] * n
        for i in range(n):
            if i > 0:
                gaps_above[i] = max(0.0, body_lines[i].top - body_lines[i - 1].bottom)
            if i < n - 1:
                gaps_below[i] = max(0.0, body_lines[i + 1].top - body_lines[i].bottom)

        # ── 4. 逐行分析 ────────────────────────────────────────────────────
        elements: List[DocumentElement] = []
        para_lines: List[OCRLine] = []
        prev: Optional[OCRLine] = None

        def flush() -> None:
            if not para_lines:
                return
            text = _norm_text("".join(l.text.strip() for l in para_lines))
            if text:
                elements.append(TextBlock(text=text))
            para_lines.clear()

        for i, line in enumerate(body_lines):
            txt = line.text.strip()
            if not txt:
                continue

            # 大间距 → 强制分段
            if prev is not None and gaps_above[i] > median_h * 2.5:
                flush()

            # 标题检测（多因子）
            h_level = _detect_heading(
                line, median_h, page_w,
                prev_gap=gaps_above[i],
                next_gap=gaps_below[i],
            )
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
                new_para = True
            elif prev is not None:
                gap_to_right = text_right - prev.right
                if gap_to_right > short_line_gap:
                    new_para = True

            if new_para and para_lines:
                flush()

            para_lines.append(line)
            prev = line

        flush()
        return elements
