# SPDX-License-Identifier: Apache-2.0
"""The fidelity gate — ServeTune's core differentiator.

A faster config is only acceptable if it does not meaningfully change the model's
output. We measure greedy-token agreement against the full-precision reference and
require the *lower* bound of its Wilson confidence interval to clear the threshold.
Using the lower CI bound (not the point estimate) means a config is accepted only
when we are statistically confident it is faithful, not merely when it looks faithful.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backends.base import RunResult
from .metrics import wilson_ci


@dataclass
class FidelityReport:
    agreement_rate: float
    ci_low: float
    ci_high: float
    matches: int
    total: int
    threshold: float
    passed: bool


def compare_to_reference(
    candidate: RunResult,
    reference: RunResult,
    max_degradation: float = 0.03,
    z: float = 1.96,
) -> FidelityReport:
    """Score `candidate` against `reference` and apply the fidelity gate.

    `max_degradation` is the largest tolerated drop in greedy-token agreement
    (e.g. 0.03 ⇒ require ≥ 97% agreement, with statistical confidence).
    """
    matches = 0
    total = 0
    for cand_seq, ref_seq in zip(candidate.outputs, reference.outputs):
        for c, r in zip(cand_seq, ref_seq):
            total += 1
            if c == r:
                matches += 1

    rate = matches / total if total else 1.0
    low, high = wilson_ci(matches, total, z)
    threshold = 1.0 - max_degradation
    passed = low >= threshold
    return FidelityReport(
        agreement_rate=rate,
        ci_low=low,
        ci_high=high,
        matches=matches,
        total=total,
        threshold=threshold,
        passed=passed,
    )
