#!/usr/bin/env bash
# 批量处理: ./batch.sh 输出文件夹 视频1 视频2 ...
# 额外参数放在 -- 后面: ./batch.sh out/ a.mov b.mov -- --stride 30
set -uo pipefail
cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
  echo "用法: ./batch.sh 输出文件夹 视频1 [视频2 ...] [-- 其他参数]" >&2
  exit 1
fi

DEST="$1"; shift
VIDEOS=(); EXTRA=()
seen_sep=0
for a in "$@"; do
  if [ "$a" = "--" ]; then seen_sep=1; continue; fi
  if [ $seen_sep -eq 1 ]; then EXTRA+=("$a"); else VIDEOS+=("$a"); fi
done

mkdir -p "$DEST"
missing=0
for v in "${VIDEOS[@]}"; do
  [ -f "$v" ] || { echo "找不到: $v" >&2; missing=1; }
done
[ $missing -eq 0 ] || exit 1

ok=(); failed=()
for v in "${VIDEOS[@]}"; do
  base="$(basename "${v%.*}")"
  echo ""
  echo "=== $base ==="
  if ./run.sh "$v" "$DEST/${base}_tangent.mp4" "${EXTRA[@]+"${EXTRA[@]}"}"; then
    ok+=("$base")
  else
    failed+=("$base")
  fi
done

echo ""
echo "完成 ${#ok[@]}/${#VIDEOS[@]} -> $DEST"
[ ${#failed[@]} -eq 0 ] || { echo "失败: ${failed[*]}" >&2; exit 1; }
