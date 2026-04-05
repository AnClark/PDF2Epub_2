"""EPUB 导出器：将 Document 模型输出为适合电子阅读器的 EPUB 3 文件。"""

import uuid
from typing import List, Tuple

from ebooklib import epub

from ..document_model import Chapter, Document, HeadingLevel, ImageBlock, TextBlock

_CSS = """\
@charset "UTF-8";

body {
  font-family: "Source Han Serif SC", "Noto Serif CJK SC", serif;
  font-size: 1em;
  line-height: 1.85;
  margin: 0;
  padding: 0;
}

h1 {
  font-size: 1.55em;
  font-weight: bold;
  text-align: center;
  margin: 0 0 1em 0;
  padding-top: 2em;
  page-break-before: always;
}

h2 {
  font-size: 1.3em;
  font-weight: bold;
  text-align: center;
  margin: 1.4em 0 0.7em 0;
}

h3 {
  font-size: 1.15em;
  font-weight: bold;
  margin: 1.1em 0 0.5em 0;
}

h4, h5, h6 {
  font-size: 1.05em;
  font-weight: bold;
  margin: 0.9em 0 0.4em 0;
}

p {
  margin: 0;
  padding: 0;
  text-indent: 2em;
  text-align: justify;
}

.figure {
  text-align: center;
  margin: 1.2em 0;
}

.figure img {
  max-width: 100%;
  height: auto;
}

.caption {
  font-size: 0.88em;
  color: #555;
  text-align: center;
  margin-top: 0.4em;
  text-indent: 0;
}
"""


def _esc(text: str) -> str:
    """HTML 实体转义。"""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _heading_tag(level: HeadingLevel) -> str:
    return f"h{level.value}"


class EPUBExporter:
    """将 Document 导出为 EPUB 3 文件。"""

    def export(self, document: Document, output_path: str) -> None:
        self._img_counter = 0

        book = epub.EpubBook()
        book.set_identifier(str(uuid.uuid4()))
        book.set_title(document.title)
        book.set_language(document.language)
        if document.author:
            book.add_author(document.author)

        # 封面
        if document.cover_image:
            book.set_cover("images/cover.jpg", document.cover_image)

        # 样式表
        css_item = epub.EpubItem(
            uid="main_css",
            file_name="style/main.css",
            media_type="text/css",
            content=_CSS.encode("utf-8"),
        )
        book.add_item(css_item)

        # 生成章节 + 收集 H2 锚点
        chapter_data: List[Tuple[epub.EpubHtml, List[Tuple[str, str]]]] = []
        for idx, chapter in enumerate(document.chapters):
            ep_chap, h2_anchors = self._build_chapter(chapter, idx, book, css_item)
            book.add_item(ep_chap)
            chapter_data.append((ep_chap, h2_anchors))

        # ── 构建分层目录（NCX + Nav）──────────────────────────────────────
        # Kindle 及 EPUB 3 阅读器均支持两级目录（章 → 节）
        toc = []
        for idx, (ep_chap, h2_anchors) in enumerate(chapter_data):
            chap_title = ep_chap.title or f"第{idx + 1}章"
            chap_link = epub.Link(ep_chap.file_name, chap_title, f"nav_chap_{idx}")
            if h2_anchors:
                section_links = [
                    epub.Link(
                        f"{ep_chap.file_name}#{aid}",
                        h2_text,
                        f"nav_{idx}_{i}",
                    )
                    for i, (aid, h2_text) in enumerate(h2_anchors)
                ]
                toc.append((epub.Section(chap_title, ep_chap.file_name), section_links))
            else:
                toc.append(chap_link)

        book.toc = toc
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav"] + [ep_chap for ep_chap, _ in chapter_data]

        epub.write_epub(output_path, book)
        print(f"  ✓ EPUB:  {output_path}")

    # ──────────────────────────────────────────────────────────────────────
    def _build_chapter(
        self,
        chapter: Chapter,
        idx: int,
        book: epub.EpubBook,
        css_item: epub.EpubItem,
    ) -> Tuple[epub.EpubHtml, List[Tuple[str, str]]]:
        """
        构建单个章节的 XHTML 内容。

        返回值：
          (EpubHtml, h2_anchors)
          h2_anchors 是 (anchor_id, heading_text) 的列表，用于生成分层目录。
        """
        html_parts: List[str] = []
        h2_anchors: List[Tuple[str, str]] = []
        h2_counter = 0

        for elem in chapter.elements:
            if isinstance(elem, TextBlock):
                if elem.is_heading and elem.heading_level in (HeadingLevel.H2, HeadingLevel.H3):
                    h2_counter += 1
                    anchor_id = f"s{idx}_{h2_counter}"
                    tag = f"h{elem.heading_level.value}"
                    html_parts.append(f'  <{tag} id="{anchor_id}">{_esc(elem.text)}</{tag}>')
                    if elem.heading_level == HeadingLevel.H2:
                        h2_anchors.append((anchor_id, elem.text))
                else:
                    html_parts.append(self._render_text(elem))
            elif isinstance(elem, ImageBlock):
                self._img_counter += 1
                img_item = self._add_image(elem, book, self._img_counter)
                html_parts.append(self._render_image(elem, img_item))

        title = chapter.title or f"第{idx + 1}章"
        body = "\n".join(html_parts)

        xhtml = (
            "<?xml version='1.0' encoding='utf-8'?>\n"
            "<!DOCTYPE html>\n"
            '<html xmlns="http://www.w3.org/1999/xhtml"'
            ' xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="zh-CN">\n'
            "<head>\n"
            '  <meta charset="utf-8"/>\n'
            f"  <title>{_esc(title)}</title>\n"
            '  <link rel="stylesheet" type="text/css" href="../style/main.css"/>\n'
            "</head>\n"
            "<body>\n"
            f"{body}\n"
            "</body>\n"
            "</html>"
        )

        chap = epub.EpubHtml(
            title=title,
            file_name=f"chapter_{idx + 1:03d}.xhtml",
            lang="zh-CN",
        )
        chap.content = xhtml.encode("utf-8")
        chap.add_item(css_item)
        return chap, h2_anchors

    def _render_text(self, block: TextBlock) -> str:
        text = _esc(block.text)
        if block.is_heading and block.heading_level:
            tag = _heading_tag(block.heading_level)
            return f"<{tag}>{text}</{tag}>"
        return f"<p>{text}</p>"

    def _render_image(self, block: ImageBlock, img_item: epub.EpubItem) -> str:
        src = f"../{img_item.file_name}"
        alt = _esc(block.caption)
        html = f'<div class="figure"><img src="{src}" alt="{alt}"/>'
        if block.caption:
            html += f'<p class="caption">{_esc(block.caption)}</p>'
        html += "</div>"
        return html

    def _add_image(
        self, block: ImageBlock, book: epub.EpubBook, counter: int
    ) -> epub.EpubItem:
        ext = block.mime_type.split("/")[-1].replace("jpeg", "jpg")
        file_name = f"images/img_{counter:03d}.{ext}"
        item = epub.EpubItem(
            uid=f"img_{counter}",
            file_name=file_name,
            media_type=block.mime_type,
            content=block.image_data,
        )
        book.add_item(item)
        return item
