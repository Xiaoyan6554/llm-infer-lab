# 实验报告

> 第一部分（模拟实验）已经完成，数字来自 `kv_sim` 的原始输出；第二部分（GPU 实验）跑完 `serve/run_all.sh` 后填写。
> 每个结论都要有图或表格支撑。`bench/report.py` 会生成 `results/report_tables.md`，里面的表可以直接贴进下面对应的位置。

## 第一部分：KV cache 调度模拟（已完成）

完整表格见 [README](../README.md#实验结果) 和 `results/sim.md`，可视化见 `dashboard/index.html`。

| 结论 | 支撑数据 |
|---|---|
| 分页分配把显存浪费从 58.1% 降到 0.9%，吞吐上限 2.23 倍 | 实验一：平均 batch 30.7 → 68.7 |
| 前缀缓存（共享 512 token system prompt）再提升 1.65 倍 | 实验二：平均 batch 50.3 → 83.1，命中 46976 块 |
| 1% 准入水位线让抢占减少 90%，吞吐只降 0.6% | 实验三：抢占 469 → 48，重算 token 252229 → 23783 |
| 块大小 16 是浪费和块表长度的折中 | 实验四：浪费 0.9%，块表 54 项；256 时浪费 12.9%、吞吐 2.23× → 1.99× |
| LRU 延迟驱逐让前缀命中率 48.4% → 60.5% | 实验五：需要 prefill 的 token 666161 → 509857（−23%） |

---

## 第二部分：GPU 实验（待完成）

## 环境

（`serve/run_all.sh` 会把 GPU 型号、驱动、vLLM / torch / CUDA 版本写到 `results/env.txt`）

| 项目 | 值 |
|---|---|
| GPU | RTX 4090 24GB |
| CUDA / 驱动 | |
| vLLM 版本 | |
| 模型 | Qwen2.5-7B-Instruct（BF16 / AWQ / GPTQ-Int8） |
| max_model_len / gpu_memory_utilization | 4096 / 0.90 |
| 压测参数 | 输入约 512 token，输出 256 token，并发 1~128，每轮请求数 = 并发 × 4 |

## 显存：手算 vs vLLM 日志

| 配置 | 权重显存 | 日志 KV cache 容量 (token) | 手算：KV 显存 / 56KB | 最大并发（4096 长度） |
|---|---|---|---|---|
| fp16 | | | | |
| awq | | | | |
| gptq-int8 | | | | |
| kv-fp8 | | | | |

`bench/report.py` 会从 `logs/vllm-*.log` 里自动提取权重显存、KV cache 容量和最大并发（见 `results/report_tables.md`）。

差异分析：

## 实验一：量化对比

![](../results/quantization.png)

（贴 results/quantization.md 的表）

| 配置 | 并发 1 的 P50 TPOT | 最高输出吞吐 | 达到最高吞吐的并发数 | 满足 P95 TPOT < 50ms 的最大吞吐 |
|---|---|---|---|---|
| fp16 | | | | |
| awq | | | | |
| gptq-int8 | | | | |
| kv-fp8 | | | | |

预期：

结果：

原因分析：

## 实验二：前缀缓存

共享前缀 1024 token + 每个请求各自 256 token，输出 256 token。

![](../results/prefix_cache.png)

| 并发 | prefix-off P50 TTFT | prefix-on P50 TTFT | 下降比例 | prefix-off 吞吐 | prefix-on 吞吐 |
|---|---|---|---|---|---|
| 1 | | | | | |
| 16 | | | | | |
| 64 | | | | | |

结论：

## 总结

一句话结论（会写进简历）：
