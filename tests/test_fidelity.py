# SPDX-License-Identifier: Apache-2.0
from servetune.backends.base import RunResult
from servetune.config import ServeConfig
from servetune.fidelity import compare_to_reference


def _result(outputs):
    return RunResult(ServeConfig(), 100.0, 50.0, 20.0, outputs)


def test_identical_outputs_pass():
    # Enough probe tokens (1000) so the Wilson lower bound clears the 97% gate.
    ref = _result([[1, 2, 3, 4, 5]] * 200)
    cand = _result([[1, 2, 3, 4, 5]] * 200)
    report = compare_to_reference(cand, ref, max_degradation=0.03)
    assert report.agreement_rate == 1.0
    assert report.passed


def test_perfect_agreement_with_too_few_tokens_is_rejected():
    # The gate demands statistical evidence: 100 perfect tokens is not enough
    # to be confident the true agreement is >= 97%, so it must fail.
    ref = _result([[1, 2, 3, 4, 5]] * 20)
    cand = _result([[1, 2, 3, 4, 5]] * 20)
    report = compare_to_reference(cand, ref, max_degradation=0.03)
    assert report.agreement_rate == 1.0
    assert not report.passed


def test_heavily_degraded_outputs_fail():
    ref = _result([[1, 2, 3, 4, 5]] * 20)
    cand = _result([[9, 9, 9, 9, 9]] * 20)
    report = compare_to_reference(cand, ref, max_degradation=0.03)
    assert report.agreement_rate == 0.0
    assert not report.passed


def test_gate_uses_lower_ci_bound_not_point_estimate():
    # 97/100 agreement: point estimate 0.97 meets a 0.97 threshold, but the
    # lower CI bound is below 0.97, so the gate must reject it.
    ref = _result([[i for i in range(100)]])
    cand_tokens = [i for i in range(97)] + [-1, -1, -1]
    cand = _result([cand_tokens])
    report = compare_to_reference(cand, ref, max_degradation=0.03)
    assert abs(report.agreement_rate - 0.97) < 1e-9
    assert report.ci_low < 0.97
    assert not report.passed
