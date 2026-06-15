# SPDX-License-Identifier: Apache-2.0
"""Statistical helpers: Wilson score interval and KL divergence.

The Wilson interval gives a statistically defensible bound on a proportion (here,
the greedy-token agreement rate between a candidate config and the reference). It is
far more reliable than the normal approximation at rates near 1.0, which is exactly
the regime fidelity checking lives in.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion k/n.

    Returns (low, high), each clamped to [0, 1]. For n == 0 returns the
    maximally uninformative (0.0, 1.0).
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def kl_divergence(p: Iterable[float], q: Iterable[float], eps: float = 1e-9) -> float:
    """KL(p || q) for two discrete distributions, in nats. Robust to zeros via eps."""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        total += pi * math.log((pi + eps) / (qi + eps))
    return total
