# SPDX-License-Identifier: Apache-2.0
from servetune.metrics import kl_divergence, wilson_ci


def test_wilson_ci_perfect_agreement_has_high_lower_bound():
    low, high = wilson_ci(500, 500)
    assert high > 0.999  # mathematically 1.0; float gives 0.9999...
    assert low > 0.99  # 500/500 should be tightly bounded near 1


def test_wilson_ci_widens_with_small_n():
    low_small, high_small = wilson_ci(9, 10)
    low_big, high_big = wilson_ci(900, 1000)
    assert (high_small - low_small) > (high_big - low_big)


def test_wilson_ci_empty():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_kl_divergence_zero_for_identical():
    p = [0.25, 0.25, 0.25, 0.25]
    assert abs(kl_divergence(p, p)) < 1e-6


def test_kl_divergence_positive_for_different():
    assert kl_divergence([0.9, 0.1], [0.5, 0.5]) > 0
