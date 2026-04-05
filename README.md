# pdf2ebook

将简体中文扫描版 PDF 书籍转换为 **EPUB**（电子阅读器优化）和 **Typst** 格式的 CLI 工具。

## 功能

- 使用 **Tesseract OCR** 识别扫描页面文字
- 自动还原**标题层级**（一/二/三级）
- 正确重建**自然段**（不将每行当作独立段落）
- 检测并嵌入**插图**（整页图片自动提取）
- 过滤页码、页眉/页脚等干扰内容
- 输出适合 Kindle Paperwhite 等设备的 EPUB 3 文件
- 同时输出可二次编排的 Typst 源文件

## 安装依赖

### 系统依赖

```bash
# Debian/Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# macOS
brew install tesseract tesseract-lang
```

### Python 依赖

```bash
pip install -e .
# 或
pip install click pytesseract Pillow ebooklib lxml pymupdf numpy
```

## 使用

```bash
# 转换为 EPUB + Typst（默认输出到当前目录）
pdf2ebook 书名.pdf

# 指定输出目录和书名作者
pdf2ebook 书名.pdf -o ./output -t "书名" -a "作者"

# 只输出 EPUB
pdf2ebook 书名.pdf -f epub

# 繁体中文
pdf2ebook 書名.pdf --lang chi_tra

# 提高分辨率（扫描质量较低时）
pdf2ebook 书名.pdf --dpi 400

# 显示进度
pdf2ebook 书名.pdf -v
```

## 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output` | 输出目录 | `.` |
| `-f, --format` | 输出格式：`epub` / `typst` / `both` | `both` |
| `-l, --lang` | Tesseract 语言代码 | `chi_sim` |
| `-d, --dpi` | PDF 渲染分辨率 | `300` |
| `-t, --title` | 书名 | （取自文件名） |
| `-a, --author` | 作者名 | — |
| `-v, --verbose` | 详细输出 | 关 |

## 项目结构

```
pdf2ebook/
├── cli.py              # 命令行入口
├── pipeline.py         # 主处理流水线
├── pdf_renderer.py     # PDF 页面渲染（PyMuPDF）
├── ocr_processor.py    # Tesseract OCR 封装
├── layout_analyzer.py  # 布局分析：标题/段落/图片检测
├── document_model.py   # 文档数据模型
└── exporters/
    ├── typst_exporter.py
    └── epub_exporter.py
```

## 段落重建原理

中文书籍每段首行缩进 2 个字符。工具通过以下规则重建段落：

1. **首行缩进**：行左边距比版心左边界多 ≥ 1.2 字符宽 → 新段落开始
2. **段末短行**：前一行行尾距右边距 > 2.5 字符宽 → 该行为段末，下一行另起段落
3. **大间距**：行间垂直间距 > 2.5 倍行高 → 强制分段

## 注意事项

- OCR 质量取决于扫描分辨率，建议原始 PDF ≥ 300 DPI
- 首次运行时请确认 `tesseract-ocr-chi-sim` 语言包已安装
- 输出的 Typst 文件需要安装思源宋体（`Source Han Serif SC`）方可编译
