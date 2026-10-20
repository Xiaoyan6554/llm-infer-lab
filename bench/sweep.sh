#!/usr/bin/env bash
# 并发扫描：对同一个服务，依次用 1,2,4,...,128 并发各跑一轮。
#
#   bash bench/sweep.sh <tag> [额外传给 bench.py 的参数...]
# 例：
#   bash bench/sweep.sh fp16
#   bash bench/sweep.sh prefix-on --shared-prefix-len 1024
#
# 可用环境变量覆盖：BASE_URL MODEL CONCURRENCIES INPUT_LEN OUTPUT_LEN OUT
set -euo pipefail

TAG=${1:?用法: bash bench/sweep.sh <tag> [bench.py 参数...]}
shift || true
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
MODEL=${MODEL:-qwen}
CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16 32 64 128"}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
OUT=${OUT:-results/results.jsonl}

cd "$(dirname "$0")/.."

echo "等待服务就绪: $BASE_URL ..."
for _ in $(seq 1 120); do
  curl -sf "$BASE_URL/v1/models" >/dev/null && break
  sleep 5
done
curl -sf "$BASE_URL/v1/models" >/dev/null || { echo "服务没起来"; exit 1; }

for C in $CONCURRENCIES; do
  # 每轮请求数 = 并发数的 4 倍，至少 32 个，保证统计分位数时样本够
  N=$(( C * 4 > 32 ? C * 4 : 32 ))
  python3 bench/bench.py --base-url "$BASE_URL" --model "$MODEL" --tag "$TAG" \
    --concurrency "$C" --num-prompts "$N" --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
    --out "$OUT" --quiet "$@"
done
echo "完成，画图：python3 bench/plot.py --input $OUT"
