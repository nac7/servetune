# SPDX-License-Identifier: Apache-2.0
"""Real vLLM backend.

Each config is run in a **fresh subprocess** (``_vllm_worker``). vLLM does not reliably
release GPU memory in-process, and a per-config subprocess also isolates OOM/crashes — so
one bad config can't take down the whole sweep. ServeTune *orchestrates* vLLM here; it does
not reimplement any inference logic.

The parent process does not import vllm (only the worker does), so importing this module is
cheap and does not require a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

from ..config import KVDtype, Quant, ServeConfig
from .base import Backend, RunResult


def _vllm_kwargs(config: ServeConfig) -> dict:
    """Translate a ServeConfig into vLLM engine kwargs (pure; unit-tested in CI)."""
    kwargs: dict = {}
    if config.quant == Quant.FP8:
        kwargs["quantization"] = "fp8"
    elif config.quant == Quant.INT4_AWQ:
        # Requires a pre-quantized AWQ checkpoint as the model id.
        kwargs["quantization"] = "awq"
    kwargs["kv_cache_dtype"] = "fp8" if config.kv_dtype == KVDtype.FP8 else "auto"
    kwargs["max_num_seqs"] = config.max_batch
    kwargs["max_model_len"] = config.max_len
    return kwargs


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(
        self,
        model: str,
        draft_model: Optional[str] = None,
        gpu_memory_utilization: Optional[float] = None,
        num_spec_tokens: int = 5,
    ):
        if importlib.util.find_spec("vllm") is None:
            raise ImportError(
                "vLLM is not installed. Install with `pip install servetune[vllm]` on a "
                "CUDA machine, or use --backend mock for a GPU-free dry run."
            )
        self.model = model
        self.draft_model = draft_model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_spec_tokens = num_spec_tokens

    def run(
        self,
        config: ServeConfig,
        prompts: List[str],
        max_new_tokens: int = 64,
    ) -> RunResult:
        with tempfile.TemporaryDirectory(prefix="servetune-") as tmp:
            result_path = os.path.join(tmp, "result.json")
            payload = {
                "model": self.model,
                "draft_model": self.draft_model,
                "spec_decode": config.spec_decode,
                "num_spec_tokens": self.num_spec_tokens,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "engine_kwargs": _vllm_kwargs(config),
                "prompts": prompts,
                "max_new_tokens": max_new_tokens,
                "result_path": result_path,
            }
            proc = subprocess.run(
                [sys.executable, "-m", "servetune.backends._vllm_worker"],
                input=json.dumps(payload),
                text=True,
            )
            if proc.returncode != 0 or not os.path.exists(result_path):
                raise RuntimeError(
                    f"vLLM worker failed for config '{config.label()}' "
                    f"(exit {proc.returncode}). See the vLLM output above for the cause "
                    f"(common causes: OOM, missing AWQ checkpoint, unsupported flag)."
                )
            with open(result_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

        return RunResult(
            config=config,
            tokens_per_sec=data["tokens_per_sec"],
            ttft_ms=data["ttft_ms"],
            p95_latency_ms=data["p95_latency_ms"],
            outputs=data["outputs"],
        )
