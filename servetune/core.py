# SPDX-License-Identifier: Apache-2.0
"""Top-level optimize() orchestration: search → run → gate → select."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .backends.base import Backend, RunResult
from .config import ServeConfig
from .fidelity import compare_to_reference
from .pareto import Candidate, pareto_frontier, select_best
from .search import build_search_space

DEFAULT_PROMPTS = [
    "Explain how a CPU pipeline works.",
    "Summarize the plot of Hamlet in three sentences.",
    "Write a Python function that reverses a linked list.",
    "What are the trade-offs between TCP and UDP?",
    "Translate 'good morning' into French, Spanish, and Japanese.",
    "Give three tips for reducing cloud infrastructure costs.",
    "Describe the water cycle for a ten year old.",
    "What is the difference between supervised and unsupervised learning?",
    "Draft a polite email declining a meeting invitation.",
    "List the first ten prime numbers and their sum.",
]


@dataclass
class OptimizeResult:
    model: str
    num_params_b: float
    reference: RunResult
    candidates: List[Candidate]
    best: Optional[Candidate]

    @property
    def frontier(self) -> List[Candidate]:
        return pareto_frontier(self.candidates)


def optimize(
    model: str,
    num_params_b: float,
    backend: Backend,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 64,
    max_degradation: float = 0.03,
    include_spec: bool = True,
    vram_gb: float = 0.0,
) -> OptimizeResult:
    """Search serving configs for `model`, gate each by fidelity, return the best.

    The reference (full-precision) run is executed once; every candidate's outputs
    are compared against it. `best` is the fastest fidelity-passing config, or None
    if every faster config degrades quality beyond `max_degradation`.
    """
    prompts = prompts or DEFAULT_PROMPTS
    space = build_search_space(num_params_b, vram_gb, include_spec)

    reference = backend.run(ServeConfig(), prompts, max_new_tokens)

    candidates: List[Candidate] = []
    for cfg in space:
        result = backend.run(cfg, prompts, max_new_tokens)
        report = compare_to_reference(result, reference, max_degradation)
        candidates.append(Candidate(result=result, fidelity=report))

    best = select_best(candidates)
    return OptimizeResult(
        model=model,
        num_params_b=num_params_b,
        reference=reference,
        candidates=candidates,
        best=best,
    )
