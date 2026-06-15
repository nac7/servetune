# SPDX-License-Identifier: Apache-2.0
"""Backend interface and the result type all backends return."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..config import ServeConfig


@dataclass
class RunResult:
    """Outcome of serving `prompts` under a given config.

    `outputs` holds the greedy-decoded token ids per prompt, which the fidelity
    gate compares against the reference run. Speed metrics are what the optimizer
    maximizes subject to the fidelity constraint.
    """

    config: ServeConfig
    tokens_per_sec: float
    ttft_ms: float
    p95_latency_ms: float
    outputs: List[List[int]]


class Backend:
    """Abstract serving backend. Implementations must be deterministic enough that
    a reference run is reproducible within a session."""

    name = "base"

    def run(
        self,
        config: ServeConfig,
        prompts: List[str],
        max_new_tokens: int = 64,
    ) -> RunResult:
        raise NotImplementedError
