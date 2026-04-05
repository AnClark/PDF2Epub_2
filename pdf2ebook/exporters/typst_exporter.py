"""Typst 导出器：将 Document 模型输出为 .typ 文件。"""

import os
from typing import List

from ..document_model import Chapter, Document, HeadingLevel, ImageBlock, TextBlock

# Typst 标题前缀（= / == / === …）
_HEADING_PREFIX = {
    HeadingLevel.H1: "= ",
    HeadingLevel.H2: "== ",
    HeadingLevel.H3: "=== ",
    HeadingLevel.H4: "==== ",
    HeadingLevel.H5: "===== ",
    HeadingLevel.H6: "====== ",
}

_TYPST_PREAMBLE = """\
// 由 pdf2ebook 自动生成
#set document(title: "{title}", author: "{author}")
#set text(
  lang: "zh",
  font: ("Source Han Serif SC", "Noto Serif CJK SC", "FZShuSong-Z01", "SimSun"),
  size: 11pt,
)
#set page(
  paper: "a5",
  margin: (top: 2cm, bottom: 2.2cm, left: 2cm, right: 2cm),
)
#set par(
  leading: 0.9em,
  spacing: 1.3em,
  first-line-indent: 2em,
  justify: true,
)
#set heading(numbering: none)

#show heading.where(level: 1): it => [
  #pagebreak(weak: true)
  #v(2em)
  #align(center)[#text(size: 18pt, weight: "bold")[#it.body]]
  #v(1em)
]
#show heading.where(level: 2): it => [
  #v(1.5em)
  #align(center)[#text(size: 15pt, weight: "bold")[#it.body]]
  #v(0.8em)
]
#show heading.where(level: 3): it => [
  #v(1em)
  #text(size: 13pt, weight: "bold")[#it.body]
  #v(0.4em)
]

"""


def _escape(text: str) -> str:
    """转义 Typst 的特殊字符。"""
    # 顺序很重要：先转义反斜杠
    for ch in ("\\", "#", "@", "$", "<", ">", "`"):
        text = text.replace(ch, f"\\{ch}")
    return text


class TypstExporter:
    """将 Document 导出为 Typst (.typ) 源文件。"""

    def export(self, document: Document, output_path: str) -> None:
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        images_dir = os.path.join(out_dir, "images")

        lines: List[str] = []
        img_counter = 0

        # 文件头
        lines.append(
            _TYPST_PREAMBLE.format(
                title=_escape(document.title),
                author=_escape(document.author),
            )
        )

        # 扉页
        lines.append("#align(center)[")
        lines.append("  #v(25%)")
        lines.append(f'  #text(size: 22pt, weight: "bold")[{_escape(document.title)}]')
        if document.author:
            lines.append("  #v(1em)")
            lines.append(f'  #text(size: 14pt)[{_escape(document.author)}]')
        lines.append("]")
        lines.append("#pagebreak()")
        lines.append("")

        # 正文
        for chapter in document.chapters:
            for elem in chapter.elements:
                if isinstance(elem, TextBlock):
                    lines.append(self._render_text(elem))
                elif isinstance(elem, ImageBlock):
                    img_counter += 1
                    img_path = self._save_image(elem, images_dir, img_counter)
                    rel = os.path.relpath(img_path, out_dir)
                    lines.append(self._render_image(elem, rel))

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        print(f"  ✓ Typst: {output_path}")
        if img_counter:
            print(f"    └─ 图片: {images_dir}/ ({img_counter} 张)")

    # ──────────────────────────────────────────────────────────────────────
    def _render_text(self, block: TextBlock) -> str:
        text = _escape(block.text)
        if block.is_heading and block.heading_level:
            prefix = _HEADING_PREFIX.get(block.heading_level, "= ")
            return f"\n{prefix}{text}\n"
        return f"\n{text}\n"

    def _render_image(self, block: ImageBlock, rel_path: str) -> str:
        parts = ["\n#figure(", f'  image("{rel_path}", width: 88%),']
        if block.caption:
            parts.append(f"  caption: [{_escape(block.caption)}],")
        parts.append(")\n")
        return "\n".join(parts)

    def _save_image(self, block: ImageBlock, images_dir: str, counter: int) -> str:
        os.makedirs(images_dir, exist_ok=True)
        ext = block.mime_type.split("/")[-1].replace("jpeg", "jpg")
        path = os.path.join(images_dir, f"img_{counter:03d}.{ext}")
        with open(path, "wb") as fh:
            fh.write(block.image_data)
        return path
