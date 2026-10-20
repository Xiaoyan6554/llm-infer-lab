#!/usr/bin/env bash
# 没有 GPU 时，用 mock 服务把整条压测链路（bench.py -> sweep.sh -> plot.py -> report.py）跑一遍。
# 结果写到 results/mock/，tag 都带 mock- 前缀，和真实 GPU 数据完全分开，不能写进简历。
#
#   bash scripts/mock_bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-8765}
OUT=results/mock/results.jsonl
mkdir -p results/mock
rm -f "$OUT"

start_mock() {
  python3 bench/mock_server.py --port "$PORT" "$@" > /dev/null 2>&1 &
  MOCK_PID=$!
  trap 'kill $MOCK_PID 2>/dev/null || true' EXIT
}
stop_mock() { kill "$MOCK_PID" 2>/dev/null || true; wait "$MOCK_PID" 2>/dev/null || true; }

export BASE_URL=http://127.0.0.1:$PORT MODEL=mock OUT

# 1) 并发扫描：吞吐-延迟曲线
start_mock
CONCURRENCIES="1 2 4 8 16 32 64 128" INPUT_LEN=512 OUTPUT_LEN=128 bash bench/sweep.sh mock
stop_mock

# 2) 前缀缓存：共享 1024 词前缀 + 各自 256 词，对比 mock 服务开/关前缀缓存
start_mock
CONCURRENCIES="1 4 16 64" INPUT_LEN=256 OUTPUT_LEN=64 bash bench/sweep.sh mock-prefix-off --shared-prefix-len 1024
stop_mock
start_mock --prefix-cache
CONCURRENCIES="1 4 16 64" INPUT_LEN=256 OUTPUT_LEN=64 bash bench/sweep.sh mock-prefix-on --shared-prefix-len 1024
stop_mock

echo "mock 压测完成：$OUT"
