# SPDX-License-Identifier: Apache-2.0
from servetune.config import Quant, ServeConfig
from servetune.search import build_search_space, estimate_weight_gb


def test_reference_always_present():
    space = build_search_space(num_params_b=8.0, vram_gb=0.0)
    assert ServeConfig() in space


def test_vram_pruning_drops_fp16_on_small_gpu():
    # 8B fp16 ~ 16 GB * 1.3 headroom = ~20.8 GB; an 8 GB GPU can't hold it.
    space = build_search_space(num_params_b=8.0, vram_gb=8.0)
    fp16_candidates = [c for c in space if c.quant == Quant.FP16 and not c.is_reference]
    assert fp16_candidates == []
    # int4 (~4 GB) should survive.
    assert any(c.quant == Quant.INT4_AWQ for c in space)


def test_no_spec_flag_excludes_speculative():
    space = build_search_space(num_params_b=8.0, vram_gb=0.0, include_spec=False)
    assert all(not c.spec_decode for c in space)


def test_estimate_weight_gb():
    assert estimate_weight_gb(8.0, Quant.FP16) == 16.0
    assert estimate_weight_gb(8.0, Quant.INT4_AWQ) == 4.0


def test_quants_filter_restricts_schemes_but_keeps_reference():
    space = build_search_space(num_params_b=8.0, vram_gb=0.0, quants=[Quant.FP8])
    assert ServeConfig() in space  # fp16 reference always present
    non_ref = [c for c in space if not c.is_reference]
    assert non_ref and all(c.quant == Quant.FP8 for c in non_ref)


def test_max_configs_caps_total_and_keeps_reference_first():
    space = build_search_space(num_params_b=8.0, vram_gb=0.0, max_configs=3)
    assert len(space) == 3
    assert space[0].is_reference
