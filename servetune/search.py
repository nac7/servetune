# SPDX-License-Identifier: Apache-2.0
"""Build the candidate search space, pruned by VRAM when the GPU is known."""

from __future__ import annotations

from itertools import product
from typing import List, Optional

from .config import KVDtype, Quant, ServeConfig

# Approximate bytes-per-parameter by quantization scheme.
_BYTES_PER_PARAM = {Quant.FP16: 2.0, Quant.FP8: 1.0, Quant.INT4_AWQ: 0.5}

# Headroom multiplier on top of raw weight size to account for activations / KV cache.
_VRAM_HEADROOM = 1.3

# Heuristic ordering: surface the configs most likely to be fast first, so that a
# --max-configs cap keeps the promising (accelerated) candidates rather than extra
# full-precision variants.
_QUANT_RANK = {Quant.FP16: 0, Quant.FP8: 1, Quant.INT4_AWQ: 2}


def _aggressiveness(cfg: ServeConfig) -> float:
    return (
        _QUANT_RANK[cfg.quant] * 2
        + (1.0 if cfg.spec_decode else 0.0)
        + (0.5 if cfg.kv_dtype == KVDtype.FP8 else 0.0)
    )


def estimate_weight_gb(num_params_b: float, quant: Quant) -> float:
    """Rough weight footprint in GB for a model of `num_params_b` billion params."""
    return num_params_b * _BYTES_PER_PARAM[quant]


def build_search_space(
    num_params_b: float,
    vram_gb: float = 0.0,
    include_spec: bool = True,
    quants: Optional[List[Quant]] = None,
    max_configs: Optional[int] = None,
) -> List[ServeConfig]:
    """Enumerate candidate configs, dropping those that cannot fit in VRAM.

    The full-precision reference config is always included and placed first (it is what
    every candidate is scored against), even if it does not fit in VRAM.

    `quants` restricts which quantization schemes to consider (default: all). `max_configs`
    caps the total number of configs returned (reference always kept) — handy for a cheap
    smoke run before a full sweep.
    """
    quant_opts = quants or [Quant.FP16, Quant.FP8, Quant.INT4_AWQ]
    kvs = [KVDtype.FP16, KVDtype.FP8]
    specs = [False, True] if include_spec else [False]

    candidates: List[ServeConfig] = []
    for quant, kv, spec in product(quant_opts, kvs, specs):
        cfg = ServeConfig(quant=quant, spec_decode=spec, kv_dtype=kv)
        if vram_gb > 0:
            if estimate_weight_gb(num_params_b, quant) * _VRAM_HEADROOM > vram_gb:
                continue
        candidates.append(cfg)

    # Always include the fp16 reference, first; order the rest most-aggressive-first so
    # a --max-configs cap keeps the promising accelerated candidates.
    reference = ServeConfig()
    candidates = [c for c in candidates if c != reference]
    candidates.sort(key=_aggressiveness, reverse=True)
    candidates.insert(0, reference)

    if max_configs is not None and max_configs > 0:
        candidates = candidates[:max_configs]
    return candidates
