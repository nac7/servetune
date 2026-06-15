# SPDX-License-Identifier: Apache-2.0
"""Build the candidate search space, pruned by VRAM when the GPU is known."""

from __future__ import annotations

from itertools import product
from typing import List

from .config import KVDtype, Quant, ServeConfig

# Approximate bytes-per-parameter by quantization scheme.
_BYTES_PER_PARAM = {Quant.FP16: 2.0, Quant.FP8: 1.0, Quant.INT4_AWQ: 0.5}

# Headroom multiplier on top of raw weight size to account for activations / KV cache.
_VRAM_HEADROOM = 1.3


def estimate_weight_gb(num_params_b: float, quant: Quant) -> float:
    """Rough weight footprint in GB for a model of `num_params_b` billion params."""
    return num_params_b * _BYTES_PER_PARAM[quant]


def build_search_space(
    num_params_b: float,
    vram_gb: float = 0.0,
    include_spec: bool = True,
) -> List[ServeConfig]:
    """Enumerate candidate configs, dropping those that cannot fit in VRAM.

    The full-precision reference config is always included (it is what every
    candidate is scored against), even if it does not fit — callers may still
    want the comparison via a smaller reference run or CPU offload.
    """
    quants = [Quant.FP16, Quant.FP8, Quant.INT4_AWQ]
    kvs = [KVDtype.FP16, KVDtype.FP8]
    specs = [False, True] if include_spec else [False]

    candidates: List[ServeConfig] = []
    for quant, kv, spec in product(quants, kvs, specs):
        cfg = ServeConfig(quant=quant, spec_decode=spec, kv_dtype=kv)
        if vram_gb > 0:
            if estimate_weight_gb(num_params_b, quant) * _VRAM_HEADROOM > vram_gb:
                continue
        candidates.append(cfg)

    reference = ServeConfig()
    if reference not in candidates:
        candidates.insert(0, reference)
    return candidates
