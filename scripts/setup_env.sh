#!/usr/bin/env bash
set -euo pipefail

# 快速配置本项目运行环境：venv + pip 依赖 + (可选) NFS 组件

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$BASE_DIR/.venv}"

INSTALL_NFS_SERVER=0
INSTALL_NFS_CLIENT=0
SKIP_APT=0
NO_VENV=0

usage() {
  cat << 'EOF'
用法:
  bash scripts/setup_env.sh [--nfs-server] [--nfs-client] [--skip-apt] [--no-venv]

选项:
  --nfs-server   安装 NFS 服务端组件（nfs-kernel-server）
  --nfs-client   安装 NFS 客户端组件（nfs-common）
  --skip-apt     跳过 apt 安装步骤（仅做 venv + pip）
  --no-venv      不创建 venv，直接对当前 python 环境 pip install

环境变量:
  PYTHON_BIN     Python 可执行文件（默认 python3）
  VENV_DIR       venv 目录（默认 <repo>/.venv）
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nfs-server)
      INSTALL_NFS_SERVER=1; shift ;;
    --nfs-client)
      INSTALL_NFS_CLIENT=1; shift ;;
    --skip-apt)
      SKIP_APT=1; shift ;;
    --no-venv)
      NO_VENV=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$SKIP_APT" -eq 0 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    # venv 常见依赖（不额外装 docker 引擎，默认你已按课程环境装好）
    sudo apt-get install -y python3-venv python3-pip
    if [[ "$INSTALL_NFS_SERVER" -eq 1 ]]; then
      sudo apt-get install -y nfs-kernel-server
    fi
    if [[ "$INSTALL_NFS_CLIENT" -eq 1 ]]; then
      sudo apt-get install -y nfs-common
    fi
  else
    echo "未检测到 apt-get：已跳过系统依赖安装。" >&2
  fi
fi

cd "$BASE_DIR"

if [[ ! -f "$BASE_DIR/requirements.txt" ]]; then
  cat > "$BASE_DIR/requirements.txt" << 'EOF'
openai>=1.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.0.0
numpy>=1.21.0
EOF
  echo "已生成 requirements.txt（openai/sentence-transformers/sklearn/numpy）"
fi

if [[ "$NO_VENV" -eq 0 ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

python -m pip install -U pip setuptools wheel

# Master/Worker 核心依赖（合并你列出的两条 pip install，修正 sqlmode -> sqlmodel）
python -m pip install -U \
  fastapi \
  uvicorn \
  sqlmodel \
  aiohttp \
  python-multipart \
  pydantic \
  docker \
  pyyaml \
  psutil

# 你列出的额外能力依赖
python -m pip install -U -r requirements.txt
python -m pip install -U PyPDF2 PyMuPDF opencc-python-reimplemented

echo "环境安装完成。"
if [[ "$NO_VENV" -eq 0 ]]; then
  echo "激活虚拟环境: source $VENV_DIR/bin/activate"
fi
