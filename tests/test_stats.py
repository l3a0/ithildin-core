"""Mechanics of the autocorrelation-robust significance block.

The module exists because an ordinary t-statistic overstates the evidence in a
series whose neighbouring values move together. `TestAutocorrelationMovesTheT`
is that claim executed in both directions, and it is the reason to prefer this
block over `scipy.stats.ttest_1samp` on a return series.

The seeded summaries are pinned tightly on purpose. Consuming repositories
commit numbers that trace to this arithmetic, so a change to the degrees of
freedom, the Bartlett weights, the lag rule or the floor has to fail here
rather than surface later as a result that quietly moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from ithildincore.stats import NeweyWestSummary, newey_west_lag, newey_west_summary, newey_west_t


def _ar1(phi: float, *, seed: int, n: int = 252, shift: float = 0.0) -> np.ndarray:
    """One AR(1) path, ``z_t = phi * z_{t-1} + e_t``, shifted off zero."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.standard_normal()
    return z + shift


class TestTheLagRule:
    def test_the_rule_is_the_closed_form(self) -> None:
        """``L = int(4 * (n / 100) ** (2 / 9))``, checked at the points it steps."""
        assert [newey_west_lag(n) for n in (1, 2, 10, 50, 99, 100, 252)] == [1, 1, 2, 3, 3, 4, 4]
        assert [newey_west_lag(n) for n in (500, 1000, 5000)] == [5, 6, 9]

    def test_the_summary_reports_the_lag_it_used(self) -> None:
        """A reader re-derives the result from the summary alone, or cannot."""
        for n in (50, 252, 1000):
            assert newey_west_summary(_ar1(0.6, seed=1, n=n)).lag == newey_west_lag(n)


class TestAutocorrelationMovesTheT:
    """Why the robust statistic is worth computing, in both directions."""

    def test_positive_autocorrelation_shrinks_the_t(self) -> None:
        """The case the module is for: the ordinary t claims evidence it lacks."""
        summary = newey_west_summary(_ar1(0.6, seed=11, shift=0.35))
        assert summary.t_naive == pytest.approx(5.026374, abs=1e-6)
        assert summary.t_newey_west == pytest.approx(3.113492, abs=1e-6)
        assert summary.t_newey_west < summary.t_naive

    def test_negative_autocorrelation_raises_the_t(self) -> None:
        """The correction has no preferred direction, which is easy to forget."""
        summary = newey_west_summary(_ar1(-0.6, seed=5, shift=0.3))
        assert summary.t_naive == pytest.approx(3.751206, abs=1e-6)
        assert summary.t_newey_west == pytest.approx(5.862606, abs=1e-6)
        assert summary.t_newey_west > summary.t_naive

    def test_independent_draws_leave_the_two_close(self) -> None:
        """With nothing to correct, the correction is small rather than absent."""
        rng = np.random.default_rng(3)
        summary = newey_west_summary(rng.standard_normal(252) + 0.2)
        assert summary.t_naive == pytest.approx(3.285974, abs=1e-6)
        assert summary.t_newey_west == pytest.approx(3.520079, abs=1e-6)
        assert abs(summary.t_newey_west - summary.t_naive) < 0.3


class TestThePinnedArithmetic:
    """Every field of one summary, to the precision a refactor would disturb."""

    def test_every_field_of_one_seeded_summary(self) -> None:
        summary = newey_west_summary(_ar1(0.6, seed=11, shift=0.35))
        assert summary.n == 252
        assert summary.mean == pytest.approx(0.367626, abs=1e-6)
        assert summary.var == pytest.approx(1.348041, abs=1e-6)
        assert summary.lag == 4

    def test_var_is_the_sample_variance_not_the_population_one(self) -> None:
        """``ddof=1``. The two differ by n/(n-1), which is 0.4 percent at n=252."""
        series = _ar1(0.6, seed=11, shift=0.35)
        assert newey_west_summary(series).var == pytest.approx(float(np.var(series, ddof=1)))
        assert newey_west_summary(series).var != pytest.approx(float(np.var(series)))

    def test_the_naive_t_is_the_textbook_one(self) -> None:
        """``mean / sqrt(var / n)``, derived here rather than taken from the code."""
        series = _ar1(0.6, seed=11, shift=0.35)
        summary = newey_west_summary(series)
        expected = float(np.mean(series)) / np.sqrt(float(np.var(series, ddof=1)) / series.size)
        assert summary.t_naive == pytest.approx(expected, abs=1e-12)

    def test_the_robust_variance_is_the_bartlett_sum(self) -> None:
        """The estimator re-derived from the formula in the module docstring."""
        series = _ar1(0.6, seed=11, shift=0.35)
        n = series.size
        mean = float(np.mean(series))
        lag = newey_west_lag(n)
        gamma_sum = sum(
            (1.0 - k / (lag + 1)) * float(np.mean((series[:-k] - mean) * (series[k:] - mean)))
            for k in range(1, lag + 1)
        )
        var_mean = (float(np.var(series, ddof=1)) + 2.0 * gamma_sum) / n
        assert newey_west_summary(series).t_newey_west == pytest.approx(
            mean / np.sqrt(var_mean), abs=1e-12
        )


class TestTheGuards:
    def test_fewer_than_two_observations_returns_zeros(self) -> None:
        """A caller's own sufficiency check fires first, so this raises nothing."""
        assert newey_west_summary(np.array([])) == NeweyWestSummary(0, 0.0, 0.0, 0.0, 0.0, 0)
        assert newey_west_summary(np.array([5.0])) == NeweyWestSummary(1, 5.0, 0.0, 0.0, 0.0, 0)

    def test_a_constant_series_reports_no_evidence_rather_than_dividing_by_zero(self) -> None:
        summary = newey_west_summary(np.full(50, 2.0))
        assert (summary.mean, summary.var) == (2.0, 0.0)
        assert (summary.t_naive, summary.t_newey_west) == (0.0, 0.0)

    def test_the_variance_floor_is_not_reached_by_ordinary_data(self) -> None:
        """The docstring calls the floor defensive. This is the search behind it.

        Re-derives the robust variance without the floor and asserts it never
        goes negative. Kept to 760 samples so the suite stays fast. The full
        search the module docstring cites covered 152,000.
        """
        rng = np.random.default_rng(0)
        for n in range(2, 40):
            for _ in range(20):
                series = rng.standard_normal(n)
                mean = float(np.mean(series))
                lag = newey_west_lag(n)
                gamma_sum = sum(
                    (1.0 - k / (lag + 1))
                    * float(np.mean((series[:-k] - mean) * (series[k:] - mean)))
                    for k in range(1, lag + 1)
                )
                unfloored = (float(np.var(series, ddof=1)) + 2.0 * gamma_sum) / n
                assert unfloored >= 0.0


class TestTheConvenienceWrapper:
    def test_newey_west_t_is_the_summary_field(self) -> None:
        series = _ar1(0.6, seed=11, shift=0.35)
        assert newey_west_t(series) == newey_west_summary(series).t_newey_west
