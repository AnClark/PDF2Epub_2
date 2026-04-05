#!/usr/bin/env bash
# install.sh — 安装 pdf2ebook 及其依赖

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. 检查 Python ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.9 或更高版本。" >&2
    exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(sys.version_info[:2])')"
echo "使用 Python: $(python3 --version)"

# ── 2. 检查 Tesseract OCR ────────────────────────────────────────────────────
if ! command -v tesseract &>/dev/null; then
    echo "警告: 未找到 tesseract，请手动安装并确保中文语言包可用。"
    echo "  Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-chi-sim"
    echo "  Fedora:        sudo dnf install tesseract tesseract-langpack-chi_sim"
    echo "  macOS:         brew install tesseract tesseract-lang"
else
    echo "Tesseract 已安装: $(tesseract --version 2>&1 | head -1)"
    if ! tesseract --list-langs 2>/dev/null | grep -q "chi_sim"; then
        echo "警告: 未检测到中文简体语言包 (chi_sim)，OCR 效果可能不佳。"
    fi
fi

# ── 3. 创建虚拟环境（若不存在）──────────────────────────────────────────────
VENV_DIR="${SCRIPT_DIR}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "创建虚拟环境 ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
fi

# ── 4. 激活虚拟环境 ──────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── 5. 安装 Python 依赖 ──────────────────────────────────────────────────────
echo "安装 Python 依赖 ..."
pip install --upgrade pip -q
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ── 6. 以可编辑模式安装本项目 ────────────────────────────────────────────────
echo "安装 pdf2ebook ..."
pip install -e "${SCRIPT_DIR}"

echo ""
echo "✓ 安装完成。"
echo "  激活环境: source ${VENV_DIR}/bin/activate"
echo "  运行工具: pdf2ebook --help"
