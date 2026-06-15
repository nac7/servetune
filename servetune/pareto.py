# SPDX-License-Identifier: Apache-2.0
"""Candidate bookkeeping, Pareto frontier, and final selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .backends.base import RunResult
from .fidelity import FidelityReport


@dataclass
class Candidate:
    result: RunResult
    fidelity: FidelityReport


def select_best(candidates: List[Candidate]) -> Optional[Candidate]:
    """Pick the fastest candidate that passes the fidelity gate.

    Returns None if nothing passes (caller should fall back to the reference).
    """
    passing = [c for c in candidates if c.fidelity.passed]
    if not passing:
        return None
    return max(passing, key=lambda c: c.result.tokens_per_sec)


def pareto_frontier(candidates: List[Candidate]) -> List[Candidate]:
    """Non-dominated set maximizing both throughput and agreement rate."""
    frontier: List[Candidate] = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            faster_eq = other.result.tokens_per_sec >= c.result.tokens_per_sec
            better_eq = other.fidelity.agreement_rate >= c.fidelity.agreement_rate
            strictly = (
                other.result.tokens_per_sec > c.result.tokens_per_sec
                or other.fidelity.agreement_rate > c.fidelity.agreement_rate
            )
            if faster_eq and better_eq and strictly:
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return frontier
