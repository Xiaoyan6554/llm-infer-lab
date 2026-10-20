#!/usr/bin/env python3
"""把 results/ 里的实验数据打包进一个自包含的 HTML 面板（离线双击就能打开，不依赖任何 CDN）。

  python3 dashboard/build.py            # 生成 dashboard/index.html
  open dashboard/index.html             # macOS；Linux 用 xdg-open，Windows 直接双击

读取的文件（缺哪个就跳过哪一节）：
  results/sim.json            ./kv_sim --json 的输出（必需）
  results/trace.json          ./kv_sim --trace 的输出（必需，显存回放动画）
  results/tests.txt           单元测试输出（scripts/run_local.sh 生成）
  results/mock/results.jsonl  mock 服务压测结果
  results/results.jsonl       GPU 压测结果（serve/run_all.sh 生成）
  results/headline.json       bench/report.py 从 GPU 结果里提取的关键数字
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_sweeps(p: Path) -> dict[str, list[dict]]:
    by_tag: dict[str, dict[int, dict]] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_tag.setdefault(r["tag"], {})[r["concurrency"]] = r
    keep = ("concurrency", "output_throughput", "request_throughput", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
            "p50_tpot_ms", "p95_tpot_ms", "p50_itl_ms", "p95_e2e_ms", "mean_e2e_ms", "littles_law_concurrency")
    return {t: [{k: d[c].get(k) for k in keep} for c in sorted(d)] for t, d in by_tag.items()}


def parse_tests(p: Path) -> dict:
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8", errors="replace")
    out = {}
    m = re.search(r"(\d+) tests, (\d+) failed", text)
    if m:
        total, failed = int(m.group(1)), int(m.group(2))
        out["cpp"] = {"total": total, "passed": total - failed}
    m = re.search(r"Ran (\d+) tests?", text)
    if m:
        total = int(m.group(1))
        failed = 0
        f = re.search(r"FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?", text)
        if f:
            failed = int(f.group(1) or 0) + int(f.group(2) or 0)
        out["py"] = {"total": total, "passed": total - failed}
    return out


def main() -> None:
    sim, trace = load_json(RES / "sim.json"), load_json(RES / "trace.json")
    if not sim or not trace:
        sys.exit("缺少 results/sim.json 或 results/trace.json，先跑 bash scripts/run_local.sh（或 kv_sim --json / --trace）")
    gpu = load_sweeps(RES / "results.jsonl")
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "sim": sim,
        "trace": trace,
        "tests": parse_tests(RES / "tests.txt"),
        "mock": load_sweeps(RES / "mock" / "results.jsonl"),
        "gpu": {t: v for t, v in gpu.items() if not t.startswith("mock")},
        "headline": load_json(RES / "headline.json"),
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    marker = "/*__DATA__*/null"
    if marker not in tpl:
        sys.exit("template.html 里找不到数据占位符")
    out = ROOT / "dashboard" / "index.html"
    out.write_text(tpl.replace(marker, blob), encoding="utf-8")
    print(f"已生成 {out.relative_to(ROOT)}（{out.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
