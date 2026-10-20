#!/usr/bin/env python3
"""bench/ 的单元测试（标准库 unittest）：python3 -m unittest discover -s bench -v"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import RequestResult, pct, summarize  # noqa: E402
from report import parse_vllm_log, quant_summary  # noqa: E402


class PercentileTest(unittest.TestCase):
    def test_matches_numpy_linear_interpolation(self):
        xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertAlmostEqual(pct(xs, 50), 5.5)
        self.assertAlmostEqual(pct(xs, 95), 9.55)
        self.assertAlmostEqual(pct(xs, 99), 9.91)
        self.assertEqual(pct([7], 99), 7)

    def test_empty_is_nan(self):
        self.assertNotEqual(pct([], 50), pct([], 50))


class SummarizeTest(unittest.TestCase):
    def test_tpot_and_littles_law(self):
        # 4 个请求，每个 2 秒完成、TTFT 0.2 秒、11 个 token -> TPOT = 1.8 / 10 = 180ms
        rs = [RequestResult(ok=True, ttft=0.2, e2e=2.0, prompt_tokens=100, output_tokens=11) for _ in range(4)]
        rs.append(RequestResult(ok=False, error="boom"))
        s = summarize(rs, duration=4.0)
        self.assertEqual(s["num_success"], 4)
        self.assertAlmostEqual(s["p50_tpot_ms"], 180.0)
        self.assertAlmostEqual(s["output_throughput"], 11.0)
        # 1 req/s × 2 s = 2 个同时在飞
        self.assertAlmostEqual(s["littles_law_concurrency"], 2.0)


class VllmLogTest(unittest.TestCase):
    def test_v1_format(self):
        log = (
            "INFO 10-20 [api_server.py:1] vLLM API server version 0.9.2\n"
            "INFO 10-20 [gpu_model_runner.py:1] Model loading took 14.2487 GiB and 4.7 seconds\n"
            "INFO 10-20 [kv_cache_utils.py:1] GPU KV cache size: 12,345 tokens\n"
            "INFO 10-20 [kv_cache_utils.py:1] Maximum concurrency for 4,096 tokens per request: 3.01x\n"
        )
        info = parse_vllm_log(log)
        self.assertEqual(info["kv_cache_tokens"], 12345)
        self.assertAlmostEqual(info["weights_gib"], 14.2487)
        self.assertAlmostEqual(info["max_concurrency"], 3.01)
        self.assertEqual(info["vllm_version"], "0.9.2")

    def test_v0_format_uses_block_count(self):
        log = "INFO [executor_base.py:1] # GPU blocks: 1000, # CPU blocks: 512\nLoading model weights took 5.5 GB\n"
        info = parse_vllm_log(log)
        self.assertEqual(info["kv_cache_tokens"], 16000)
        self.assertAlmostEqual(info["weights_gib"], 5.5)


class QuantSummaryTest(unittest.TestCase):
    def test_best_and_slo(self):
        rows = {
            1: {"p50_tpot_ms": 20, "p95_tpot_ms": 21, "output_throughput": 50},
            8: {"p50_tpot_ms": 30, "p95_tpot_ms": 40, "output_throughput": 300},
            32: {"p50_tpot_ms": 60, "p95_tpot_ms": 70, "output_throughput": 500},
        }
        q = quant_summary(rows, slo_tpot_ms=50)
        self.assertEqual(q["tpot_c1_p50_ms"], 20)
        self.assertEqual(q["best_concurrency"], 32)
        self.assertEqual(q["max_throughput_under_slo"], 300)


if __name__ == "__main__":
    unittest.main()
