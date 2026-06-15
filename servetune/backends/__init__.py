# SPDX-License-Identifier: Apache-2.0
"""Pluggable inference backends. ServeTune orchestrates these; it never reimplements them."""

from .base import Backend, RunResult
from .mock import MockBackend

__all__ = ["Backend", "RunResult", "MockBackend", "get_backend"]


def get_backend(name: str, **kwargs) -> Backend:
    """Resolve a backend by name. Real engines are imported lazily so the core
    package stays installable without a GPU / vLLM."""
    name = name.lower()
    if name == "mock":
        return MockBackend(**kwargs)
    if name == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r} (expected 'mock' or 'vllm')")
