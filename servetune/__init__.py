# SPDX-License-Identifier: Apache-2.0
"""ServeTune: auto-tuning + fidelity-gated local LLM serving.

ServeTune searches the serving config space (quantization, speculative decoding,
KV-cache dtype, batching) on *your* hardware and returns the Pareto-optimal config,
gated by a statistical fidelity guarantee against the full-precision reference.
"""

__version__ = "0.1.0"

from .config import ServeConfig, Quant, KVDtype
from .core import optimize, OptimizeResult

__all__ = [
    "ServeConfig",
    "Quant",
    "KVDtype",
    "optimize",
    "OptimizeResult",
    "__version__",
]
