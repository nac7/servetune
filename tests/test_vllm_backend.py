# SPDX-License-Identifier: Apache-2.0
"""Tests for the vLLM backend.

The config->kwargs mapping is pure and runs in CI (no GPU, no vllm import). The actual
serving run is gated behind an installed vllm AND an explicit opt-in env var, so it is
skipped everywhere except a real GPU box.
"""

import importlib.util
import os

import pytest

from servetune.backends.vllm_backend import _vllm_kwargs
from servetune.config import KVDtype, Quant, ServeConfig


def test_fp16_has_no_quantization():
    kw = _vllm_kwargs(ServeConfig(quant=Quant.FP16))
    assert "quantization" not in kw
    assert kw["kv_cache_dtype"] == "auto"


def test_fp8_mapping():
    kw = _vllm_kwargs(ServeConfig(quant=Quant.FP8, kv_dtype=KVDtype.FP8))
    assert kw["quantization"] == "fp8"
    assert kw["kv_cache_dtype"] == "fp8"


def test_int4_maps_to_awq():
    kw = _vllm_kwargs(ServeConfig(quant=Quant.INT4_AWQ))
    assert kw["quantization"] == "awq"


def test_batch_and_len_passthrough():
    kw = _vllm_kwargs(ServeConfig(max_batch=16, max_len=8192))
    assert kw["max_num_seqs"] == 16
    assert kw["max_model_len"] == 8192


_GPU_OPT_IN = importlib.util.find_spec("vllm") is not None and os.environ.get(
    "SERVETUNE_GPU_TEST"
) == "1"


@pytest.mark.skipif(not _GPU_OPT_IN, reason="needs vllm + SERVETUNE_GPU_TEST=1 (real GPU)")
def test_vllm_run_smoke():
    from servetune.backends.vllm_backend import VLLMBackend

    model = os.environ.get("SERVETUNE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    backend = VLLMBackend(model=model)
    result = backend.run(ServeConfig(), ["Hello, world."], max_new_tokens=8)
    assert result.outputs and len(result.outputs[0]) > 0
    assert result.tokens_per_sec > 0
