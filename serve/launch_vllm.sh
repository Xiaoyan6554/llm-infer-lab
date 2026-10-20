#!/usr/bin/env bash
# 按实验配置启动 vLLM 的 OpenAI 兼容服务（端口 8000，模型名统一叫 qwen）。
#
#   bash serve/launch_vllm.sh <配置名> [额外的 vllm serve 参数...]
#
# 配置名：
#   fp16         Qwen2.5-7B-Instruct，BF16 权重（基线）
#   awq          AWQ 4bit 权重量化
#   gptq-int8    GPTQ 8bit 权重量化
#   kv-fp8       BF16 权重 + FP8 KV cache（KV cache 容量翻倍）
#   prefix-off   BF16，关闭前缀缓存
#   prefix-on    BF16，开启前缀缓存
#
# 启动后注意看日志里这几行，记到 docs/REPORT.md：
#   - GPU KV cache size: xxx tokens       （KV cache 一共能放多少 token）
#   - Maximum concurrency for xxx tokens per request: xx.xx x
#   - 权重占用多少显存（Model loading took xx GiB）
#
# 不同 vLLM 版本参数名可能有变化，报错时先跑：vllm serve --help | grep -i <关键词>
set -euo pipefail

CFG=${1:?用法: bash serve/launch_vllm.sh <fp16|awq|gptq-int8|kv-fp8|prefix-off|prefix-on> [vllm 参数...]}
shift || true
MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models}
MAX_LEN=${MAX_LEN:-4096}
GPU_UTIL=${GPU_UTIL:-0.90}

BASE="$MODEL_DIR/Qwen2.5-7B-Instruct"
EXTRA=()
case "$CFG" in
  fp16)       MODEL=$BASE ;;
  awq)        MODEL="$MODEL_DIR/Qwen2.5-7B-Instruct-AWQ" ;;
  gptq-int8)  MODEL="$MODEL_DIR/Qwen2.5-7B-Instruct-GPTQ-Int8" ;;
  kv-fp8)     MODEL=$BASE; EXTRA+=(--kv-cache-dtype fp8) ;;
  prefix-off) MODEL=$BASE; EXTRA+=(--no-enable-prefix-caching) ;;
  prefix-on)  MODEL=$BASE; EXTRA+=(--enable-prefix-caching) ;;
  *) echo "未知配置 $CFG"; exit 1 ;;
esac

# 除了 prefix-on 以外都关掉前缀缓存：新版 vLLM 默认开启，
# 不关的话压测里随机 prompt 的公共部分也可能命中，干扰其他实验的对比。
if [[ "$CFG" != prefix-* ]]; then
  EXTRA+=(--no-enable-prefix-caching)
fi

mkdir -p logs
LOG="logs/vllm-$CFG-$(date +%m%d-%H%M).log"
echo "启动 $CFG：$MODEL"
echo "日志：$LOG"
set -x
vllm serve "$MODEL" \
  --served-model-name qwen \
  --host 0.0.0.0 --port 8000 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs 256 \
  "${EXTRA[@]}" "$@" 2>&1 | tee "$LOG"
