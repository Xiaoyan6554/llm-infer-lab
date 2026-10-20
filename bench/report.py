#!/usr/bin/env python3
"""从压测结果和 vLLM 日志里提取关键数字，生成 REPORT 用的表格和可以直接写进简历的句子。

只用标准库。GPU 实验跑完（serve/run_all.sh）之后：

  python3 bench/report.py                       # 读 results/results.jsonl + logs/vllm-*.log
  python3 bench/report.py --prefix-concurrency 64

输出：
  results/headline.json   关键数字（dashboard 会读它）
  results/report_tables.md  可以直接贴进 docs/REPORT.md 的表格
  终端打印两句简历用的话（数字全部来自上面两份数据）

没有 GPU 时可以用 mock 数据检查流程：
  python3 bench/report.py --input results/mock/results.jsonl --tag-prefix mock- --out-dir results/mock
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

QUANT_TAGS = ["fp16", "awq", "gptq-int8", "kv-fp8"]
PREFIX_TAGS = ["prefix-off", "prefix-on"]
LABELS = {"fp16": "BF16", "awq": "AWQ-INT4", "gptq-int8": "GPTQ-INT8", "kv-fp8": "BF16 + FP8 KV cache",
          "prefix-off": "前缀缓存关", "prefix-on": "前缀缓存开"}


def load_results(path: Path) -> dict[str, dict[int, dict]]:
    by_tag: dict[str, dict[int, dict]] = {}
    if not path.exists():
        return by_tag
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_tag.setdefault(r["tag"], {})[r["concurrency"]] = r  # 重复跑时保留最后一次
    return by_tag


def num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_vllm_log(text: str) -> dict:
    """兼容 vLLM V0/V1 几个版本的日志格式，找不到的字段就不填。"""
    out: dict = {}
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    if m:
        out["kv_cache_tokens"] = int(num(m.group(1)))
    else:
        m = re.search(r"# GPU blocks:\s*([\d,]+)", text)  # V0：块数 × 块大小 16
        if m:
            out["kv_cache_tokens"] = int(num(m.group(1))) * 16
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", text)
    if m:
        out["max_concurrency"] = float(m.group(2))
        out["max_concurrency_len"] = int(num(m.group(1)))
    m = re.search(r"(?:Model loading took|Loading model weights took)\s*([\d.]+)\s*(GiB|GB)", text)
    if m:
        out["weights_gib"] = float(m.group(1))
    m = re.search(r"vLLM API server version\s*([0-9][\w.+-]*)", text) or re.search(r"\bvLLM\b.*?\bversion\s+([0-9][\w.+-]*)", text)
    if m:
        out["vllm_version"] = m.group(1)
    return out


def load_logs(log_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(log_dir.glob("vllm-*.log")):  # 文件名带时间戳，排序后同一配置保留最新一份
        m = re.match(r"vllm-(.+)-\d{4}-\d{4}\.log$", p.name)
        if m:
            info = parse_vllm_log(p.read_text(encoding="utf-8", errors="replace"))
            if info:
                out[m.group(1)] = {**info, "log": p.name}
    return out


def fmt(x: float | None, nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}"


def quant_summary(rows: dict[int, dict], slo_tpot_ms: float) -> dict:
    cs = sorted(rows)
    best_c = max(cs, key=lambda c: rows[c]["output_throughput"])
    ok = [c for c in cs if rows[c]["p95_tpot_ms"] < slo_tpot_ms]
    return {
        "tpot_c1_p50_ms": rows[1]["p50_tpot_ms"] if 1 in rows else None,
        "max_output_throughput": rows[best_c]["output_throughput"],
        "best_concurrency": best_c,
        "max_throughput_under_slo": max((rows[c]["output_throughput"] for c in ok), default=None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="results/results.jsonl")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag-prefix", default="", help="mock 数据用 mock-")
    ap.add_argument("--prefix-concurrency", type=int, default=16, help="前缀缓存那句话取哪个并发下的数字")
    ap.add_argument("--slo-tpot-ms", type=float, default=50)
    args = ap.parse_args()

    data = load_results(Path(args.input))
    logs = load_logs(Path(args.logs))
    tp = args.tag_prefix
    headline: dict = {"source": args.input, "tag_prefix": tp, "quant": {}, "kv": logs, "prefix": None, "resume": []}
    md: list[str] = []

    # ---- 实验一：量化 ----
    quant = {t: quant_summary(data[tp + t], args.slo_tpot_ms) for t in QUANT_TAGS if tp + t in data}
    headline["quant"] = quant
    if quant:
        md += ["## 实验一：量化对比", "",
               f"| 配置 | 并发 1 的 P50 TPOT (ms) | 最高输出吞吐 (tok/s) | 达到最高吞吐的并发数 | 满足 P95 TPOT < {args.slo_tpot_ms:g}ms 的最大吞吐 |",
               "|---|---|---|---|---|"]
        for t, q in quant.items():
            md.append(f"| {LABELS[t]} | {fmt(q['tpot_c1_p50_ms'])} | {fmt(q['max_output_throughput'], 0)} | "
                      f"{q['best_concurrency']} | {fmt(q['max_throughput_under_slo'], 0)} |")
        md.append("")
    if logs:
        md += ["## 显存：vLLM 日志", "", "| 配置 | 权重显存 (GiB) | KV cache 容量 (token) | 最大并发（日志） | 日志文件 |",
               "|---|---|---|---|---|"]
        for t, info in logs.items():
            md.append(f"| {LABELS.get(t, t)} | {fmt(info.get('weights_gib'), 2)} | {fmt(info.get('kv_cache_tokens'), 0)} | "
                      f"{fmt(info.get('max_concurrency'), 2)} | {info['log']} |")
        md.append("")

    if "fp16" in quant and "awq" in quant and quant["fp16"]["tpot_c1_p50_ms"] and quant["awq"]["tpot_c1_p50_ms"]:
        s = (f"AWQ-INT4 单并发下 TPOT 由 {quant['fp16']['tpot_c1_p50_ms']:.1f} ms 降至 {quant['awq']['tpot_c1_p50_ms']:.1f} ms，"
             f"最高输出吞吐 {quant['awq']['max_output_throughput']:,.0f} tok/s")
        kv16, kv8 = logs.get("fp16", {}).get("kv_cache_tokens"), logs.get("kv-fp8", {}).get("kv_cache_tokens")
        if kv16 and kv8:
            s += f"；FP8 KV cache 使 KV 容量由 {kv16:,} 提升至 {kv8:,} token"
        headline["resume"].append("量化对比：在 RTX 4090 上对比 BF16、AWQ-INT4、GPTQ-INT8 与 FP8 KV cache：" + s + "。")

    # ---- 实验二：前缀缓存 ----
    off, on = data.get(tp + "prefix-off"), data.get(tp + "prefix-on")
    if off and on:
        common = sorted(set(off) & set(on))
        md += ["## 实验二：前缀缓存", "",
               "| 并发 | 关 P50 TTFT (ms) | 开 P50 TTFT (ms) | TTFT 下降 | 关 输出吞吐 | 开 输出吞吐 | 吞吐提升 |",
               "|---|---|---|---|---|---|---|"]
        rows = []
        for c in common:
            a, b = off[c], on[c]
            row = {"concurrency": c, "ttft_off": a["p50_ttft_ms"], "ttft_on": b["p50_ttft_ms"],
                   "ttft_drop": 1 - b["p50_ttft_ms"] / a["p50_ttft_ms"],
                   "tput_off": a["output_throughput"], "tput_on": b["output_throughput"],
                   "tput_gain": b["output_throughput"] / a["output_throughput"] - 1}
            rows.append(row)
            md.append(f"| {c} | {row['ttft_off']:.0f} | {row['ttft_on']:.0f} | {row['ttft_drop']:.1%} | "
                      f"{row['tput_off']:.0f} | {row['tput_on']:.0f} | {row['tput_gain']:+.1%} |")
        md.append("")
        pick = next((r for r in rows if r["concurrency"] == args.prefix_concurrency), rows[-1] if rows else None)
        headline["prefix"] = {"rows": rows, "picked": pick}
        if pick:
            headline["resume"].append(
                f"前缀缓存：1024 token 共享前缀场景下（并发 {pick['concurrency']}），开启前缀缓存后 P50 TTFT 降低 "
                f"{pick['ttft_drop'] * 100:.0f}%，输出吞吐提升 {pick['tput_gain'] * 100:.0f}%。")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "headline.json").write_text(json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"已写入 {out / 'headline.json'} 和 {out / 'report_tables.md'}")
    if not quant and not (off and on):
        print(f"在 {args.input} 里没找到 {tp}fp16 / {tp}prefix-on 等 tag 的数据，先跑 serve/run_all.sh。")
    if headline["resume"]:
        if tp:
            print("\n流程检查（mock 数据，数字没有意义，不能写进简历）：")
        else:
            print("\n简历可用句子（数字全部来自上面的数据文件）：")
        for s in headline["resume"]:
            print("  • " + s)


if __name__ == "__main__":
    main()
