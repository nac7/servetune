# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: load one vLLM engine, run one config, write results to a file.

Invoked as ``python -m servetune.backends._vllm_worker`` with a JSON payload on stdin.
Runs exactly one configuration so the OS reclaims all GPU memory on exit. Results are
written to ``payload['result_path']`` (not stdout) so vLLM's own logging cannot corrupt
the parsed output.
"""

from __future__ import annotations

import json
import sys
import time
from typing import List


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def _build_llm(payload: dict):
    from vllm import LLM

    kwargs = dict(payload["engine_kwargs"])
    if payload.get("gpu_memory_utilization") is not None:
        kwargs["gpu_memory_utilization"] = payload["gpu_memory_utilization"]

    if payload.get("spec_decode") and payload.get("draft_model"):
        spec = {
            "model": payload["draft_model"],
            "num_speculative_tokens": payload.get("num_spec_tokens", 5),
        }
        try:
            return LLM(model=payload["model"], speculative_config=spec, **kwargs)
        except TypeError:
            # Older vLLM API took flat speculative_* kwargs.
            return LLM(
                model=payload["model"],
                speculative_model=payload["draft_model"],
                num_speculative_tokens=payload.get("num_spec_tokens", 5),
                **kwargs,
            )
    return LLM(model=payload["model"], **kwargs)


def main() -> int:
    payload = json.load(sys.stdin)
    prompts: List[str] = payload["prompts"]
    max_new_tokens: int = payload["max_new_tokens"]

    from vllm import SamplingParams

    llm = _build_llm(payload)
    # Greedy decoding is required so token agreement vs. the reference is meaningful.
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0

    token_ids = [list(o.outputs[0].token_ids) for o in outputs]
    total_tokens = sum(len(t) for t in token_ids) or 1

    ttfts: List[float] = []
    latencies: List[float] = []
    for o in outputs:
        m = getattr(o, "metrics", None)
        if m is None:
            continue
        arrival = getattr(m, "arrival_time", None)
        first = getattr(m, "first_token_time", None)
        finished = getattr(m, "finished_time", None)
        if arrival is not None and first is not None:
            ttfts.append((first - arrival) * 1000.0)
        if arrival is not None and finished is not None:
            latencies.append((finished - arrival) * 1000.0)

    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
    ttft_ms = (sum(ttfts) / len(ttfts)) if ttfts else (elapsed * 1000.0 / max(1, len(prompts)))
    p95_ms = _p95(latencies) if latencies else (elapsed * 1000.0)

    result = {
        "tokens_per_sec": round(tokens_per_sec, 1),
        "ttft_ms": round(ttft_ms, 1),
        "p95_latency_ms": round(p95_ms, 1),
        "outputs": token_ids,
    }
    with open(payload["result_path"], "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
