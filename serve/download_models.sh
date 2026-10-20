#!/usr/bin/env bash
# 从 ModelScope（魔搭）下载实验用的模型，国内网络不用翻墙。
# AutoDL 上建议放在数据盘 /root/autodl-tmp，系统盘只有 30G 左右，放不下。
#
#   bash serve/download_models.sh            # 下载全部三个
#   bash serve/download_models.sh fp16       # 只下载 FP16 版
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models}
pip install -q modelscope

download() {
  local id=$1
  local name=${id#*/}
  echo ">>> 下载 $id -> $MODEL_DIR/$name"
  modelscope download --model "$id" --local_dir "$MODEL_DIR/$name"
}

case "${1:-all}" in
  fp16) download Qwen/Qwen2.5-7B-Instruct ;;              # 约 15GB
  awq)  download Qwen/Qwen2.5-7B-Instruct-AWQ ;;          # 4bit，约 5.5GB
  gptq-int8) download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 ;;  # 8bit，约 9GB
  all)
    download Qwen/Qwen2.5-7B-Instruct
    download Qwen/Qwen2.5-7B-Instruct-AWQ
    download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8
    ;;
  *) echo "未知参数 $1"; exit 1 ;;
esac
df -h "$MODEL_DIR"
