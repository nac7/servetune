# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point: `servetune optimize <model> ...`."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .backends import get_backend
from .config import Quant
from .core import optimize
from .profiler import detect_gpu
from .report import render_console, render_html


def _parse_quants(value: Optional[str]) -> Optional[List[Quant]]:
    if not value:
        return None
    out: List[Quant] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(Quant(item))
        except ValueError:
            valid = ", ".join(q.value for q in Quant)
            raise SystemExit(f"error: unknown quant {item!r} (valid: {valid})")
    return out or None


def _load_prompts(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="servetune",
        description="Auto-tuning + fidelity-gated local LLM serving.",
    )
    p.add_argument("--version", action="version", version=f"servetune {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize", help="Find the fastest fidelity-passing serve config.")
    opt.add_argument("model", help="Model id or path (e.g. meta-llama/Llama-3.1-8B).")
    opt.add_argument("--params", type=float, default=8.0, help="Model size in billions of params.")
    opt.add_argument("--backend", default="mock", choices=["mock", "vllm"], help="Serving backend.")
    opt.add_argument("--gpu", default="auto", help="GPU selection (auto-detected by default).")
    opt.add_argument("--max-degradation", type=float, default=0.03, help="Max tolerated agreement drop.")
    opt.add_argument("--max-new-tokens", type=int, default=64, help="Tokens generated per probe prompt.")
    opt.add_argument("--no-spec", action="store_true", help="Exclude speculative decoding from the search.")
    opt.add_argument("--quants", help="Comma-separated quant schemes to test (e.g. fp16,fp8). Default: all.")
    opt.add_argument("--max-configs", type=int, help="Cap the number of configs (reference kept). For cheap smoke runs.")
    opt.add_argument("--draft-model", help="Draft model id for speculative-decoding configs (vllm backend).")
    opt.add_argument("--gpu-memory-utilization", type=float, help="vLLM gpu_memory_utilization (0-1).")
    opt.add_argument("--prompts", help="Path to a newline-delimited probe prompt file.")
    opt.add_argument("--report", help="Write an HTML report to this path.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "optimize":
        gpu = detect_gpu()
        print(
            f"GPU: {gpu.name}"
            + (f" | {gpu.total_vram_gb} GB | CC {gpu.compute_capability}" if gpu.available else "")
        )
        if not gpu.available and args.backend == "vllm":
            print("error: --backend vllm needs a CUDA GPU. Use --backend mock for a dry run.", file=sys.stderr)
            return 2

        if args.backend == "vllm":
            backend_kwargs = {
                "model": args.model,
                "draft_model": args.draft_model,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        else:
            backend_kwargs = {}
        backend = get_backend(args.backend, **backend_kwargs)
        result = optimize(
            model=args.model,
            num_params_b=args.params,
            backend=backend,
            prompts=_load_prompts(args.prompts),
            max_new_tokens=args.max_new_tokens,
            max_degradation=args.max_degradation,
            include_spec=not args.no_spec,
            vram_gb=gpu.total_vram_gb,
            quants=_parse_quants(args.quants),
            max_configs=args.max_configs,
        )
        print(render_console(result))
        if args.report:
            render_html(result, args.report)
            print(f"HTML report written to {args.report}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
