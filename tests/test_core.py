# SPDX-License-Identifier: Apache-2.0
from servetune.backends import MockBackend
from servetune.config import Quant
from servetune.core import optimize


def test_optimize_picks_fast_faithful_config_and_rejects_int4():
    result = optimize(
        model="test/model",
        num_params_b=8.0,
        backend=MockBackend(seed=0),
        max_new_tokens=64,
        max_degradation=0.03,
        vram_gb=24.0,
    )
    assert result.best is not None
    # The fastest faithful pick should be faster than the reference.
    assert result.best.result.tokens_per_sec > result.reference.tokens_per_sec
    # The selected config must pass its own gate.
    assert result.best.fidelity.passed

    # int4-awq is the most aggressive and should fail the fidelity gate in the mock model.
    int4 = [c for c in result.candidates if c.result.config.quant == Quant.INT4_AWQ]
    assert int4 and all(not c.fidelity.passed for c in int4)


def test_reference_is_self_consistent():
    result = optimize(
        model="test/model",
        num_params_b=8.0,
        backend=MockBackend(seed=1),
        vram_gb=24.0,
    )
    # The reference compared to itself is always perfect.
    ref_candidates = [c for c in result.candidates if c.result.config.is_reference]
    assert ref_candidates
    assert ref_candidates[0].fidelity.agreement_rate == 1.0


def test_determinism():
    a = optimize("m", 8.0, MockBackend(seed=7), vram_gb=24.0)
    b = optimize("m", 8.0, MockBackend(seed=7), vram_gb=24.0)
    assert a.best.result.config.label() == b.best.result.config.label()
