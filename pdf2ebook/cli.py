"""CLI 入口：pdf2ebook 命令行工具。"""

import sys

import click
from rich.console import Console

from .pipeline import Pipeline

_err_console = Console(stderr=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("pdf_file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "--output", "-o",
    default=".",
    show_default=True,
    help="输出目录",
)
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["typst", "epub", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="输出格式",
)
@click.option(
    "--lang", "-l",
    default="chi_sim",
    show_default=True,
    help="Tesseract 语言代码（简体中文：chi_sim；繁体：chi_tra）",
)
@click.option(
    "--dpi", "-d",
    default=300,
    show_default=True,
    help="PDF 页面渲染分辨率（DPI）",
)
@click.option("--title", "-t", default=None, help="书名（默认取文件名）")
@click.option("--author", "-a", default=None, help="作者名")
@click.option("--verbose", "-v", is_flag=True, default=False, help="显示详细进度")
@click.option(
    "--workers", "-w",
    default=1,
    show_default=True,
    metavar="N",
    help="并行 OCR 线程数（建议 ≤ CPU 核心数；1 = 串行）",
)
def main(
    pdf_file: str,
    output: str,
    output_format: str,
    lang: str,
    dpi: int,
    title: str,
    author: str,
    verbose: bool,
    workers: int,
) -> None:
    """将扫描版 PDF 书籍转换为 EPUB 和/或 Typst 格式。

    \b
    示例：
      pdf2ebook book.pdf
      pdf2ebook book.pdf -o ./output -f epub -t "书名" -a "作者"
      pdf2ebook book.pdf --dpi 400 --lang chi_tra
      pdf2ebook book.pdf --workers 4
    """
    try:
        pipeline = Pipeline(
            pdf_path=pdf_file,
            output_dir=output,
            output_format=output_format,
            lang=lang,
            dpi=dpi,
            title=title,
            author=author,
            verbose=verbose,
            workers=workers,
        )
        pipeline.run()
    except Exception as exc:  # noqa: BLE001
        _err_console.print(f"[bold red]错误：[/]{exc}")
        if verbose:
            import traceback
            _err_console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
