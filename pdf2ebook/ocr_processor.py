"""OCR 处理器：用 Tesseract 对页面图像进行识别，返回结构化数据。"""

from typing import Dict, List, Optional

import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output


class OCRLine:
    """代表一行 OCR 识别结果，含文本和位置信息。"""

    __slots__ = ("block_num", "par_num", "line_num", "left", "top", "width", "height", "text", "conf")

    def __init__(
        self,
        block_num: int,
        par_num: int,
        line_num: int,
        left: int,
        top: int,
        width: int,
        height: int,
        text: str,
        conf: float,
    ) -> None:
        self.block_num = block_num
        self.par_num = par_num
        self.line_num = line_num
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.text = text
        self.conf = conf

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width


class OCRResult:
    """一页 OCR 识别结果的封装。"""

    def __init__(self, raw: Dict[str, List], page_width: int, page_height: int) -> None:
        self.raw = raw
        self.page_width = page_width
        self.page_height = page_height
        self._lines: Optional[List[OCRLine]] = None

    def get_lines(self) -> List[OCRLine]:
        """提取行级别的识别结果（level=4），过滤掉无文本行。"""
        if self._lines is not None:
            return self._lines

        raw = self.raw
        n = len(raw["level"])

        # 收集行边界框（level=4 代表行）
        line_boxes: Dict[tuple, dict] = {}
        for i in range(n):
            if raw["level"][i] != 4:
                continue
            key = (raw["block_num"][i], raw["par_num"][i], raw["line_num"][i])
            line_boxes[key] = {
                "block_num": raw["block_num"][i],
                "par_num": raw["par_num"][i],
                "line_num": raw["line_num"][i],
                "left": raw["left"][i],
                "top": raw["top"][i],
                "width": raw["width"][i],
                "height": raw["height"][i],
                "words": [],
            }

        # 将单词（level=5）聚合到对应行
        for i in range(n):
            if raw["level"][i] != 5:
                continue
            conf = float(raw["conf"][i])
            if conf < 0:
                continue
            word = raw["text"][i].strip()
            if not word:
                continue
            key = (raw["block_num"][i], raw["par_num"][i], raw["line_num"][i])
            if key in line_boxes:
                line_boxes[key]["words"].append((word, conf))

        lines: List[OCRLine] = []
        for info in line_boxes.values():
            if not info["words"]:
                continue
            text = "".join(w for w, _ in info["words"])
            avg_conf = float(np.mean([c for _, c in info["words"]]))
            lines.append(
                OCRLine(
                    block_num=info["block_num"],
                    par_num=info["par_num"],
                    line_num=info["line_num"],
                    left=info["left"],
                    top=info["top"],
                    width=info["width"],
                    height=info["height"],
                    text=text,
                    conf=avg_conf,
                )
            )

        # 按从上到下、从左到右排序
        lines.sort(key=lambda l: (l.top, l.left))
        self._lines = lines
        return lines

    def get_median_line_height(self) -> float:
        """返回正文行高度的中位数，用作基准字号。"""
        heights = [l.height for l in self.get_lines() if l.text.strip()]
        if not heights:
            return 30.0
        return float(np.median(heights))

    def total_char_count(self) -> int:
        """返回本页识别到的总字符数。"""
        return sum(len(l.text.strip()) for l in self.get_lines())


class OCRProcessor:
    """对页面图像运行 Tesseract OCR，返回结构化 OCRResult。"""

    def __init__(self, lang: str = "chi_sim", config: str = "") -> None:
        self.lang = lang
        # --psm 3: 全自动分页，适合书籍正文；--oem 1: LSTM 引擎
        self.config = config or "--psm 3 --oem 1"

    def process(self, image: Image.Image) -> OCRResult:
        """对单页图像进行 OCR，返回 OCRResult。"""
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self.config,
            output_type=Output.DICT,
        )
        return OCRResult(data, image.width, image.height)
