#!/usr/bin/env python3
"""把 results.jsonl 画成 4 张对比图，并输出一张 Markdown 汇总表。

  pip install matplotlib
  python bench/plot.py --input results/results.jsonl --outdir results
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path: str) -> dict[str, list[dict]]:
    by_tag: dict[str, dict[int, dict]] = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_tag[r["tag"]][r["concurrency"]] = r  # 同一 tag 同一并发重复跑时，保留最后一次
    return {t: [d[c] for c in sorted(d)] for t, d in by_tag.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/results.jsonl")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tags", nargs="*", help="只画这些 tag（默认全部）")
    args = ap.parse_args()

    data = load(args.input)
    if args.tags:
        data = {t: v for t, v in data.items() if t in args.tags}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 图里用英文标题，避免服务器上没有中文字体时显示成方框
    panels = [
        ("output_throughput", "Output throughput (tok/s)"),
        ("p95_ttft_ms", "P95 TTFT (ms)"),
        ("p50_tpot_ms", "P50 TPOT (ms)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, en) in zip(axes.flat, panels):
        for tag, rows in data.items():
            ax.plot([r["concurrency"] for r in rows], [r[key] for r in rows], marker="o", label=tag)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Concurrency")
        ax.set_title(en)
        ax.grid(alpha=0.3)
        ax.legend()

    # 第 4 张：吞吐-延迟权衡曲线。越靠右下越好。
    ax = axes.flat[3]
    for tag, rows in data.items():
        xs = [r["output_throughput"] for r in rows]
        ys = [r["p50_tpot_ms"] for r in rows]
        ax.plot(xs, ys, marker="o", label=tag)
        for r, x, y in zip(rows, xs, ys):
            ax.annotate(str(r["concurrency"]), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Output throughput (tok/s)")
    ax.set_ylabel("P50 TPOT (ms)")
    ax.set_title("Throughput vs latency (labels = concurrency)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    png = out / "benchmark.png"
    fig.savefig(png, dpi=150)
    print(f"图已保存：{png}")

    lines = ["| tag | 并发 | 输出吞吐 tok/s | 请求/s | P50 TTFT ms | P95 TTFT ms | P50 TPOT ms | P95 E2E ms |",
             "|---|---|---|---|---|---|---|---|"]
    for tag, rows in data.items():
        for r in rows:
            lines.append(f"| {tag} | {r['concurrency']} | {r['output_throughput']:.1f} | {r['request_throughput']:.2f} | "
                         f"{r['p50_ttft_ms']:.0f} | {r['p95_ttft_ms']:.0f} | {r['p50_tpot_ms']:.1f} | {r['p95_e2e_ms']:.0f} |")
    md = out / "summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总表已保存：{md}")


if __name__ == "__main__":
    main()
