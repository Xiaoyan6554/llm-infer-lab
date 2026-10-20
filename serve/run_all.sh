#!/usr/bin/env bash
# 一键跑完所有对比实验（在 GPU 机器上用 tmux 跑，RTX 4090 单卡大约 1~1.5 小时）。
# 每个配置：启动 vLLM -> 等就绪 -> 并发扫描 -> 关掉服务。
#
#   tmux new -s bench
#   bash serve/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs

# 记录实验环境，写 REPORT 时要用（GPU 型号、驱动、vLLM / torch / CUDA 版本）
{
  date
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
} > results/env.txt 2>&1 || true
# 最后出图要用 matplotlib，GPU 镜像里不一定有
python3 -c "import matplotlib" 2>/dev/null || pip install -q matplotlib

run_cfg() {
  local cfg=$1; shift
  bash serve/launch_vllm.sh "$cfg" > /dev/null 2>&1 &
  local pid=$!
  bash bench/sweep.sh "$cfg" "$@"
  kill "$pid" 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null
  sleep 15  # 等显存释放
}

# 实验一：权重量化 / KV cache 量化对比（输入 512，输出 256）
for cfg in fp16 awq gptq-int8 kv-fp8; do
  run_cfg "$cfg"
done

# 实验二：前缀缓存（所有请求共享 1024 token 的前缀，每个请求自己的部分 256 token）
INPUT_LEN=256 run_cfg prefix-off --shared-prefix-len 1024
INPUT_LEN=256 run_cfg prefix-on --shared-prefix-len 1024

python3 bench/plot.py --input results/results.jsonl --outdir results --tags fp16 awq gptq-int8 kv-fp8
mv results/benchmark.png results/quantization.png
mv results/summary.md results/quantization.md
python3 bench/plot.py --input results/results.jsonl --outdir results --tags prefix-off prefix-on
mv results/benchmark.png results/prefix_cache.png
mv results/summary.md results/prefix_cache.md

# 提取关键数字（含 vLLM 日志里的 KV cache 容量）+ 生成简历句子，再刷新可视化面板
python3 bench/report.py
python3 dashboard/build.py
echo "全部完成，结果在 results/，面板在 dashboard/index.html"
