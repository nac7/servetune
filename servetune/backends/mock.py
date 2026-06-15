# SPDX-License-Identifier: Apache-2.0
"""Deterministic mock backend.

This lets the *entire* ServeTune pipeline run end-to-end with no GPU, no model
download, and no vLLM install — useful for CI, tests, demos, and developing the
optimizer/fidelity logic. It models the two things that matter for the thesis:

  * speed scales up with more aggressive quantization and with speculative decoding
  * output quality degrades with lossy choices (quant, fp8 KV cache), while
    speculative decoding is *lossless* and therefore introduces no token flips

The numbers are illustrative, not calibrated to any real GPU.
"""

from __future__ import annotations

import hashlib
import random
from typing import List

from ..config import KVDtype, Quant, ServeConfig
from .base import Backend, RunResult

# Per-token probability that a config flips the greedy argmax vs. the reference.
_QUANT_FLIP = {Quant.FP16: 0.0, Quant.FP8: 0.01, Quant.INT4_AWQ: 0.06}
_KV_FLIP = {KVDtype.FP16: 0.0, KVDtype.FP8: 0.005}

# Speed multipliers relative to the fp16 baseline.
_QUANT_SPEED = {Quant.FP16: 1.0, Quant.FP8: 1.4, Quant.INT4_AWQ: 2.1}
_SPEC_SPEED = 1.6


def _stable_hash(s: str) -> int:
    """Hash that is stable across processes (unlike the builtin hash())."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class MockBackend(Backend):
    name = "mock"

    def __init__(self, base_tps: float = 40.0, vocab: int = 32000, seed: int = 0):
        self.base_tps = base_tps
        self.vocab = vocab
        self.seed = seed

    def _reference_tokens(self, prompt: str, n: int) -> List[int]:
        """Config-independent 'true' output for a prompt — the reference."""
        rng = random.Random(_stable_hash(f"{prompt}|{self.seed}"))
        return [rng.randrange(self.vocab) for _ in range(n)]

    def run(
        self,
        config: ServeConfig,
        prompts: List[str],
        max_new_tokens: int = 64,
    ) -> RunResult:
        flip_p = _QUANT_FLIP[config.quant] + _KV_FLIP[config.kv_dtype]
        # Speculative decoding is provably lossless: it does not change which tokens
        # are produced, only how fast. So it adds speed but never flips.
        outputs: List[List[int]] = []
        for prompt in prompts:
            ref = self._reference_tokens(prompt, max_new_tokens)
            rng = random.Random(
                _stable_hash(f"{prompt}|{self.seed}") ^ _stable_hash(config.label())
            )
            toks: List[int] = []
            for t in ref:
                if flip_p > 0 and rng.random() < flip_p:
                    toks.append((t + 1 + rng.randrange(self.vocab - 1)) % self.vocab)
                else:
                    toks.append(t)
            outputs.append(toks)

        speed = (
            self.base_tps
            * _QUANT_SPEED[config.quant]
            * (_SPEC_SPEED if config.spec_decode else 1.0)
        )
        ttft = 120.0 / _QUANT_SPEED[config.quant]
        p95 = (1000.0 / speed) * 1.5
        return RunResult(
            config=config,
            tokens_per_sec=round(speed, 1),
            ttft_ms=round(ttft, 1),
            p95_latency_ms=round(p95, 1),
            outputs=outputs,
        )
