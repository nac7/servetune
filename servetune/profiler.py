# SPDX-License-Identifier: Apache-2.0
"""Hardware detection. Falls back gracefully when no CUDA GPU / torch is present."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUInfo:
    name: str
    total_vram_gb: float
    compute_capability: str
    available: bool


def detect_gpu() -> GPUInfo:
    """Detect the primary CUDA GPU, or report CPU-only if unavailable."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GPUInfo(
                name=props.name,
                total_vram_gb=round(props.total_memory / 1e9, 1),
                compute_capability=f"{props.major}.{props.minor}",
                available=True,
            )
    except Exception:
        pass
    return GPUInfo(
        name="CPU (no CUDA GPU detected)",
        total_vram_gb=0.0,
        compute_capability="n/a",
        available=False,
    )
