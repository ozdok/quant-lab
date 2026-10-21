#!/usr/bin/env python3
"""Warm up a vLLM serve before benchmark runs.

Sends one tiny generation to trigger Triton / torch.compile kernels,
then a second to confirm the server is warm. stdlib only.

Usage:
  ./warmup.py --base-url http://<serve-host>:8051/v1 --model my-model
  ./warmup.py --base-url http://<serve-host>:8051/v1  # auto-resolves model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def get(path: str, timeout: int = 10) -> dict | list:
    with urllib.request.urlopen(path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://<serve-host>:8051/v1")
    ap.add_argument("--model", default="auto",
                    help="'auto' resolves to the first id at {base}/models")
    ap.add_argument("--prompt", default="Say OK.")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="Keep room for thinking traces (reasoning models "
                    "eat short budgets in deliberation)")
    ap.add_argument("--health-retries", type=int, default=60)
    a = ap.parse_args()

    base = a.base_url.rstrip("/")
    host = base[: -len("/v1")] if base.endswith("/v1") else base

    for i in range(a.health_retries):
        try:
            with urllib.request.urlopen(host + "/health", timeout=10) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
        print(f"waiting for health... ({i + 1}/{a.health_retries})", flush=True)
    else:
        print("error: server never became healthy", file=sys.stderr)
        return 1

    model = a.model
    if model in ("", "auto", "my-model", "default"):
        ids = get(base + "/models")
        model = ids["data"][0]["id"]
    print(f"model: {model}")

    for n in (1, 2):
        t0 = time.time()
        resp = post(base + "/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": a.prompt}],
            "max_tokens": a.max_tokens,
            "temperature": 0,
        })
        dt = (time.time() - t0) * 1000
        msg = resp["choices"][0]["message"]
        text = msg.get("content") or ""
        think = msg.get("reasoning") or ""
        shown = text.strip()[:80] or f"[thinking {len(think)} chars]"
        print(f"ping {n}: {dt:.0f}ms -> {shown!r}")

    print("warm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
