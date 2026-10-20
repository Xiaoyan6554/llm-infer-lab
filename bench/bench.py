#!/usr/bin/env python3
"""LLM 推理服务压测工具（只用 Python 标准库，不需要 pip install）。

对任何 OpenAI 兼容的 /v1/completions 流式接口（vLLM、SGLang、本项目的 mock_server）
以固定并发数（闭环）或泊松到达（开环，--request-rate）发请求，统计：
  - TTFT  首 token 延迟：发出请求 -> 收到第一个生成的 token
  - TPOT  每个输出 token 的平均耗时：(总耗时 - TTFT) / (输出 token 数 - 1)
  - E2E   端到端延迟
  - 吞吐：每秒完成请求数、每秒输出 token 数
并给出 mean / p50 / p95 / p99。结果追加写入一个 JSONL 文件，供 plot.py 画图。

示例：
  python bench/bench.py --base-url http://127.0.0.1:8000 --model qwen \
      --concurrency 16 --num-prompts 200 --input-len 512 --output-len 256 --tag fp16
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# 常见英文单词，在 Qwen / Llama 的分词器里基本一个词就是一个 token，
# 用来粗略控制 prompt 长度。实际 token 数以服务端返回的 usage.prompt_tokens 为准。
WORDS = (
    "the of and to in is was for on that with as by at from his her an this are "
    "be have it not they which one you were all we can there when more been would "
    "time some into its two other than then only first new about also people may "
    "after over most made such these years city world day state part work life "
    "water house year number way long little good great small right old place"
).split()


@dataclass
class RequestResult:
    ok: bool
    ttft: float = 0.0  # 秒
    e2e: float = 0.0  # 秒
    prompt_tokens: int = 0
    output_tokens: int = 0
    itl: list[float] = field(default_factory=list)  # 相邻两个 token 之间的间隔
    error: str = ""

    @property
    def tpot(self) -> float | None:
        if self.output_tokens <= 1:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)


def make_prompt(rng: random.Random, n_words: int, shared_prefix: str) -> str:
    return shared_prefix + " ".join(rng.choice(WORDS) for _ in range(n_words))


def send_one(base_url: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool, timeout: float) -> RequestResult:
    """发一个流式请求，边读 SSE 边记时间。"""
    u = urlparse(base_url)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    port = u.port or (443 if u.scheme == "https" else 80)
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True  # vLLM 扩展参数：不在 EOS 处停，保证每个请求都生成 max_tokens 个 token

    res = RequestResult(ok=False)
    start = time.perf_counter()
    last = start
    chunks_with_text = 0
    try:
        conn = conn_cls(u.hostname, port, timeout=timeout)
        conn.request(
            "POST",
            (u.path.rstrip("/") or "") + "/v1/completions",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            res.error = f"HTTP {resp.status}: {resp.read(300).decode(errors='replace')}"
            return res
        for raw in resp:  # http.client 会自动处理 chunked 编码，这里按行读 SSE
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            msg = json.loads(data)
            now = time.perf_counter()
            if msg.get("usage"):
                res.prompt_tokens = msg["usage"].get("prompt_tokens", 0)
                res.output_tokens = msg["usage"].get("completion_tokens", 0)
            choices = msg.get("choices") or []
            if choices and choices[0].get("text"):
                if chunks_with_text == 0:
                    res.ttft = now - start
                else:
                    res.itl.append(now - last)
                chunks_with_text += 1
                last = now
        conn.close()
    except Exception as e:  # 超时、连接被拒等
        res.error = f"{type(e).__name__}: {e}"
        return res

    res.e2e = time.perf_counter() - start
    if res.output_tokens == 0:  # 服务端没返回 usage 时，用收到的文本块数近似
        res.output_tokens = chunks_with_text
    res.ok = chunks_with_text > 0
    if not res.ok:
        res.error = "no tokens received"
    return res


def pct(xs: list[float], p: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认方式一致）。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[RequestResult], duration: float) -> dict:
    ok = [r for r in results if r.ok]
    out = {
        "num_requests": len(results),
        "num_success": len(ok),
        "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_throughput": sum(r.output_tokens for r in ok) / duration,
        "total_token_throughput": sum(r.prompt_tokens + r.output_tokens for r in ok) / duration,
        "mean_prompt_tokens": statistics.mean(r.prompt_tokens for r in ok) if ok else 0,
        "mean_output_tokens": statistics.mean(r.output_tokens for r in ok) if ok else 0,
    }
    metrics = {
        "ttft_ms": [r.ttft * 1000 for r in ok],
        "tpot_ms": [r.tpot * 1000 for r in ok if r.tpot is not None],
        "itl_ms": [x * 1000 for r in ok for x in r.itl],
        "e2e_ms": [r.e2e * 1000 for r in ok],
    }
    for name, xs in metrics.items():
        out[f"mean_{name}"] = statistics.mean(xs) if xs else float("nan")
        for p in (50, 95, 99):
            out[f"p{p}_{name}"] = pct(xs, p)
    # Little 定律：系统里平均同时有多少请求 = 请求吞吐 × 平均端到端延迟。
    # 闭环压测时它应该接近设定的并发数，差得多说明客户端本身成了瓶颈或者有请求失败。
    out["littles_law_concurrency"] = out["request_throughput"] * out["mean_e2e_ms"] / 1000 if ok else 0.0
    return out


def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    shared = make_prompt(rng, args.shared_prefix_len, "") + " " if args.shared_prefix_len > 0 else ""
    prompts = [make_prompt(rng, args.input_len, shared) for _ in range(args.num_prompts)]

    def call(p: str) -> RequestResult:
        return send_one(args.base_url, args.model, p, args.output_len, not args.no_ignore_eos, args.timeout)

    # 预热：让服务端完成 CUDA graph 捕获、显存分配等一次性开销，不计入结果
    for _ in range(args.warmup):
        w = call(make_prompt(rng, 16, ""))
        if not w.ok:
            sys.exit(f"预热请求失败，检查服务是否启动：{w.error}")

    done = 0
    lock = threading.Lock()

    def tracked(p: str) -> RequestResult:
        nonlocal done
        r = call(p)
        with lock:
            done += 1
            if not args.quiet:
                print(f"\r  进度 {done}/{len(prompts)}", end="", file=sys.stderr, flush=True)
        return r

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.request_rate > 0:
            # 泊松到达（开环）：请求间隔服从指数分布，不管前面的请求有没有结束，到点就发，
            # 更接近线上真实流量。此时 concurrency 只是"同时在飞请求数"的上限。
            arrival = random.Random(args.seed + 1)
            futures = []
            for p in prompts:
                futures.append(pool.submit(tracked, p))
                time.sleep(arrival.expovariate(args.request_rate))
            results = [f.result() for f in futures]
        else:
            # 固定并发（闭环）：线程池里同时最多 concurrency 个请求在飞，一个结束立刻补下一个
            results = list(pool.map(tracked, prompts))
    duration = time.perf_counter() - start
    if not args.quiet:
        print(file=sys.stderr)

    summary = summarize(results, duration)
    errors = [r.error for r in results if not r.ok]
    summary.update(
        tag=args.tag,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        input_len=args.input_len,
        output_len=args.output_len,
        shared_prefix_len=args.shared_prefix_len,
        model=args.model,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        sample_errors=errors[:3],
    )
    return summary


def print_summary(s: dict) -> None:
    mode = f"泊松到达 {s['request_rate']} req/s，并发上限 {s['concurrency']}" if s.get("request_rate") else f"并发 {s['concurrency']}"
    print(f"\n==== [{s['tag']}] {mode} | 成功 {s['num_success']}/{s['num_requests']} | 用时 {s['duration_s']:.1f}s ====")
    print(f"平均输入 {s['mean_prompt_tokens']:.0f} token，平均输出 {s['mean_output_tokens']:.0f} token")
    print(f"吞吐：{s['request_throughput']:.2f} req/s，输出 {s['output_throughput']:.1f} tok/s，总计 {s['total_token_throughput']:.1f} tok/s")
    print(f"Little 定律验算：请求吞吐 × 平均 E2E = {s['littles_law_concurrency']:.2f} 个同时在飞的请求")
    print(f"{'指标':<8}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}  (ms)")
    for m in ("ttft", "tpot", "itl", "e2e"):
        k = f"{m}_ms"
        print(f"{m.upper():<8}{s['mean_'+k]:>10.1f}{s['p50_'+k]:>10.1f}{s['p95_'+k]:>10.1f}{s['p99_'+k]:>10.1f}")
    if s["sample_errors"]:
        print("错误示例：", *s["sample_errors"], sep="\n  ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="和服务端 --served-model-name 一致")
    ap.add_argument("--concurrency", type=int, default=8, help="闭环并发数；开了 --request-rate 时是在飞请求的上限")
    ap.add_argument("--request-rate", type=float, default=0, help="每秒请求数，>0 时按泊松过程到达（开环）")
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--input-len", type=int, default=512, help="prompt 大约多少个词（≈token）")
    ap.add_argument("--output-len", type=int, default=128, help="每个请求生成多少 token")
    ap.add_argument("--shared-prefix-len", type=int, default=0, help="所有请求共享的前缀长度，测前缀缓存用")
    ap.add_argument("--no-ignore-eos", action="store_true", help="允许模型提前结束（默认强制生成满 output-len）")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="default", help="这组实验的名字，比如 fp16 / awq-int4")
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    s = run(args)
    print_summary(s)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n结果已追加到 {args.out}")


if __name__ == "__main__":
    main()
