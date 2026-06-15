# SPDX-License-Identifier: Apache-2.0
"""Real vLLM backend (scaffold).

This intentionally lives behind a lazy import so the core package installs without
a GPU. The implementation below sketches the intended v0.1 wiring; filling in the
TODOs is the first GPU-side milestone. ServeTune *orchestrates* vLLM — it does not
reimplement any inference logic.
"""

from __future__ import annotations

from typing import List

from ..config import KVDtype, Quant, ServeConfig
from .base import Backend, RunResult


def _vllm_kwargs(config: ServeConfig) -> dict:
    """Translate a ServeConfig into vLLM LLM() engine kwargs."""
    kwargs: dict = {}
    if config.quant == Quant.FP8:
        kwargs["quantization"] = "fp8"
    elif config.quant == Quant.INT4_AWQ:
        kwargs["quantization"] = "awq"
    kwargs["kv_cache_dtype"] = "fp8" if config.kv_dtype == KVDtype.FP8 else "auto"
    kwargs["max_num_seqs"] = config.max_batch
    kwargs["max_model_len"] = config.max_len
    # Speculative decoding is configured via speculative_config in recent vLLM;
    # the draft model is chosen upstream and passed in by the optimizer.
    return kwargs


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model: str, draft_model: str | None = None):
        try:
            import vllm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "vLLM is not installed. Install with `pip install servetune[vllm]` "
                "on a CUDA machine, or use --backend mock for a GPU-free dry run."
            ) from exc
        self.model = model
        self.draft_model = draft_model

    def run(
        self,
        config: ServeConfig,
        prompts: List[str],
        max_new_tokens: int = 64,
    ) -> RunResult:  # pragma: no cover - requires GPU
        raise NotImplementedError(
            "vLLM backend execution is the first GPU-side milestone. "
            "Wiring: build LLM(model, **_vllm_kwargs(config)), time generate() with "
            "greedy SamplingParams(max_tokens=max_new_tokens), collect token ids into "
            "RunResult.outputs, and record tokens_per_sec / ttft / p95."
        )
