#!/usr/bin/env bash
# 用法: ./run.sh 输入视频.mov [输出.mp4]
# 其余参数原样传给 python -m hough_tangent，例如:
#   ./run.sh in.mov out.mp4 --corners round --margin-frac 0.1
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  echo "需要 Python 3.9 或更高版本。当前: $("$PY" --version 2>&1)" >&2
  echo "装好新版后可以用: PYTHON=/path/to/python3 ./run.sh ..." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "首次运行，正在创建虚拟环境..."
  "$PY" -m venv .venv
  ./.venv/bin/python -m pip install --quiet --upgrade pip
  ./.venv/bin/python -m pip install --quiet -r requirements.txt
fi

if [ $# -lt 1 ]; then
  echo "用法: ./run.sh 输入视频.mov [输出.mp4] [其他参数...]" >&2
  exit 1
fi

SRC="$1"; shift
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  DST="$1"; shift
else
  DST="out/$(basename "${SRC%.*}")_tangent.mp4"
fi

exec ./.venv/bin/python -m hough_tangent "$SRC" -o "$DST" "$@"
