"""文档数据模型：定义转换过程中使用的主要数据结构。"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Union


class HeadingLevel(IntEnum):
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6


@dataclass
class TextBlock:
    """段落或标题文本块。"""
    text: str
    is_heading: bool = False
    heading_level: Optional[HeadingLevel] = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()


@dataclass
class ImageBlock:
    """插图/图片块。"""
    image_data: bytes
    mime_type: str = "image/jpeg"
    caption: str = ""
    width: int = 0
    height: int = 0


DocumentElement = Union[TextBlock, ImageBlock]


@dataclass
class Chapter:
    """书籍章节，包含若干元素。"""
    title: str = ""
    elements: List[DocumentElement] = field(default_factory=list)


@dataclass
class Document:
    """整本书的文档模型。"""
    title: str
    author: str = ""
    language: str = "zh-CN"
    chapters: List[Chapter] = field(default_factory=list)
    cover_image: Optional[bytes] = None
