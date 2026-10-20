#!/usr/bin/env bash
# 一键在本地跑完所有不需要 GPU 的部分，最后生成可视化面板 dashboard/index.html：
#   编译 C++ -> 单元测试（C++ + Python）-> 调度模拟器 -> 显存动画数据 -> mock 压测 -> 生成面板
#
#   bash scripts/run_local.sh              # 全部，约 6 分钟（mock 压测占大头）
#   SKIP_MOCK=1 bash scripts/run_local.sh  # 跳过 mock 压测，十几秒
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results kvcache/build

echo "==> 编译 C++"
if command -v cmake >/dev/null 2>&1; then
  cmake -S kvcache -B kvcache/build -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build kvcache/build -j >/dev/null
else
  # 没装 cmake 时直接调用编译器，结果一样
  CXX=${CXX:-c++}
  FLAGS="-std=c++17 -O2 -Wall -Wextra -Ikvcache/include"
  $CXX $FLAGS kvcache/src/kv_cache_manager.cpp kvcache/src/simulate.cpp -o kvcache/build/kv_sim
  $CXX $FLAGS kvcache/src/kv_cache_manager.cpp kvcache/tests/test_kv_cache.cpp -o kvcache/build/kv_tests
fi

echo "==> 单元测试"
{ ./kvcache/build/kv_tests; python3 -m unittest discover -s bench 2>&1; } | tee results/tests.txt

echo "==> 调度模拟器"
./kvcache/build/kv_sim --json results/sim.json | tee results/sim.md
./kvcache/build/kv_sim --trace results/trace.json --capacity-tokens 16384 --requests 120 --trace-every 20

if [ "${SKIP_MOCK:-0}" != 1 ]; then
  echo "==> mock 压测（约 5 分钟）"
  bash scripts/mock_bench.sh
  python3 bench/report.py --input results/mock/results.jsonl --tag-prefix mock- --out-dir results/mock
fi
if [ -f results/results.jsonl ]; then
  echo "==> GPU 结果汇总"
  python3 bench/report.py
fi

echo "==> 生成面板"
python3 dashboard/build.py
echo "完成。打开 dashboard/index.html 查看（macOS: open dashboard/index.html）"
