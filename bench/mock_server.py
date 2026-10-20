#!/usr/bin/env python3
"""模拟 OpenAI 兼容推理服务（只用标准库），在没有 GPU 的电脑上调试 bench.py 用。

它不跑模型，只是按一个粗糙的"batching 代价模型"控制吐 token 的速度：
  - prefill 耗时随 prompt 长度线性增长
  - 同时在跑的请求越多，每个 token 越慢（模拟 decode 阶段 batch 变大）
  - 可选 --prefix-cache：按 16 个词一块做哈希链，命中的前缀块不再计入 prefill 耗时
所以并发升高时，你能看到和真实服务一样的趋势：吞吐上升、TPOT 变大。
这些数字只用来调试压测流程，不能当作真实 GPU 结果。

  python bench/mock_server.py --port 8000
  python bench/mock_server.py --port 8000 --prefix-cache
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACTIVE = 0
LOCK = threading.Lock()
ARGS: argparse.Namespace
PREFIX_BLOCKS: set[int] = set()  # 已经"算过"的前缀块哈希（和 kvcache/ 里的哈希链是同一个思路）
BLOCK_WORDS = 16


def cached_prefix_tokens(words: list[str]) -> int:
    """返回能命中前缀缓存的词数，并把这个 prompt 的所有满块登记进去。"""
    hit, prev, chain_alive = 0, 0, True
    with LOCK:
        for i in range(len(words) // BLOCK_WORDS):
            prev = hash((prev, tuple(words[i * BLOCK_WORDS:(i + 1) * BLOCK_WORDS])))
            if chain_alive and prev in PREFIX_BLOCKS:
                hit += BLOCK_WORDS
            else:
                chain_alive = False
                PREFIX_BLOCKS.add(prev)
    return hit


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024  # 默认只有 5，高并发时连接会被丢，客户端要等 1 秒重试 SYN

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # 才能用 chunked 流式输出

    def log_message(self, *_):  # 不刷屏
        pass

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _chunk(self, s: str) -> None:
        b = s.encode()
        self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": ARGS.model, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        global ACTIVE
        if self.path != "/v1/completions":
            self._json(404, {"error": "not found"})
            return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        words = str(req.get("prompt", "")).split()
        prompt_tokens = len(words)
        prefill_tokens = prompt_tokens - (cached_prefix_tokens(words) if ARGS.prefix_cache else 0)
        max_tokens = int(req.get("max_tokens", 16))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        with LOCK:
            ACTIVE += 1
        try:
            time.sleep(ARGS.prefill_base + ARGS.prefill_per_token * prefill_tokens)
            for i in range(max_tokens):
                if i > 0:
                    with LOCK:
                        batch = ACTIVE
                    time.sleep(ARGS.decode_base * (1 + ARGS.batch_penalty * (batch - 1)))
                chunk = {"id": "cmpl-mock", "object": "text_completion", "model": ARGS.model,
                         "choices": [{"index": 0, "text": " tok", "finish_reason": None}]}
                self._chunk(f"data: {json.dumps(chunk)}\n\n")
            if (req.get("stream_options") or {}).get("include_usage"):
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                         "total_tokens": prompt_tokens + max_tokens}
                self._chunk(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
            self._chunk("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with LOCK:
                ACTIVE -= 1


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--prefill-base", type=float, default=0.02, help="秒")
    ap.add_argument("--prefill-per-token", type=float, default=0.00005, help="秒/token")
    ap.add_argument("--decode-base", type=float, default=0.01, help="batch=1 时每个 token 的秒数")
    ap.add_argument("--batch-penalty", type=float, default=0.03, help="每多一个并发请求，单 token 变慢的比例")
    ap.add_argument("--prefix-cache", action="store_true", help="模拟前缀缓存：命中的前缀不计 prefill 耗时")
    ARGS = ap.parse_args()
    srv = Server((ARGS.host, ARGS.port), Handler)
    print(f"mock server on http://{ARGS.host}:{ARGS.port}  model={ARGS.model}  prefix_cache={ARGS.prefix_cache}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
