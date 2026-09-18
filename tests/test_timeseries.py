"""Mechanics of the OLS, ADF and Ornstein-Uhlenbeck primitives.

Hand-built fixtures are pinned exactly, because their answers are known in
closed form. The synthetic ADF cases assert the threshold property, which is
the verdict the statistic implies, rather than a seeded value that would say
nothing if it moved.

One case here is not about arithmetic at all. ``TestTheFixedLagIsLoadBearing``
holds the module's central warning by executing it: on one series the verdict
flips when the lag count is chosen from the data instead of fixed. A consuming
repository that changes that default silently changes what its results say.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from quantcore.timeseries import ADF_CRIT_CONST, adf_tstat, ols, ou_half_life


class TestTimeseriesPrimitives:
    def test_ols_recovers_exact_line(self) -> None:
        """OLS of y = 2x + 1 on design [x, 1] returns [2, 1] with ~0 residual."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 1.0
        fit = ols(y, np.column_stack([x, np.ones(5)]))
        assert fit.beta[0] == pytest.approx(2.0, abs=1e-9)
        assert fit.beta[1] == pytest.approx(1.0, abs=1e-9)
        assert float(np.abs(fit.resid).max()) == pytest.approx(0.0, abs=1e-9)

    def test_ou_half_life_of_known_decay(self) -> None:
        """A z_{t+1} = 0.5 z_t decay has slope -0.5, so half-life = ln(2)/0.5."""
        z = np.zeros(200)
        z[0] = 1.0
        for t in range(1, 200):
            z[t] = 0.5 * z[t - 1]
        assert ou_half_life(z) == pytest.approx(np.log(2.0) / 0.5, abs=1e-4)

    def test_ou_half_life_infinite_when_not_mean_reverting(self) -> None:
        """A pure random walk does not mean-revert, so the half-life is huge or +inf."""
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.standard_normal(500))
        assert ou_half_life(walk) > 100.0

    def test_adf_random_walk_not_rejected(self) -> None:
        """A unit-root random walk should NOT reject the no-stationarity null."""
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.standard_normal(2000))
        tstat, nobs = adf_tstat(walk, lags=1, constant=True)
        assert tstat > ADF_CRIT_CONST["10%"]
        assert nobs == 1998

    def test_adf_stationary_ar1_rejected(self) -> None:
        """A stationary AR(1) at phi=0.2 rejects the unit-root null hard."""
        rng = np.random.default_rng(2)
        ar = np.zeros(2000)
        for t in range(1, 2000):
            ar[t] = 0.2 * ar[t - 1] + rng.standard_normal()
        tstat, _ = adf_tstat(ar, lags=1, constant=True)
        assert tstat < ADF_CRIT_CONST["1%"]


class TestTheFixedLagIsLoadBearing:
    """The module docstring says a data-chosen lag can reverse a verdict.

    This is that claim, executed. The series is a slowly mean-reverting AR(1)
    carrying a moving-average term, which is the shape that makes a
    lag-selection rule want more lags than one. At the fixed lag of one the
    statistic clears the 10 percent critical value. At every other lag tried
    it does not, and the rule that reads the lag from the data picks ten.

    The effect is not a quirk of one seed. Of the first sixty seeds of this
    construction, twenty-three reject at lag one and fail to reject under
    AIC selection.
    """

    @staticmethod
    def _series() -> np.ndarray:
        rng = np.random.default_rng(0)
        shocks = rng.standard_normal(320)
        z = np.zeros(320)
        for t in range(2, 320):
            z[t] = 0.965 * z[t - 1] + shocks[t] + 0.9 * shocks[t - 1]
        return z

    def test_the_two_lag_rules_reach_opposite_verdicts(self) -> None:
        series = self._series()
        fixed, _ = adf_tstat(series, lags=1, constant=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = adfuller(series, autolag="AIC")
        auto, chosen_lag = float(result[0]), int(result[2])
        assert fixed == pytest.approx(-3.199, abs=5e-4)
        assert auto == pytest.approx(-1.9323, abs=5e-4)
        assert fixed < ADF_CRIT_CONST["10%"] < auto
        assert chosen_lag == 10

    def test_only_the_fixed_lag_rejects(self) -> None:
        """The verdict belongs to the lag count, not to the data."""
        series = self._series()
        assert adf_tstat(series, lags=1, constant=True)[0] < ADF_CRIT_CONST["10%"]
        for k in (2, 4, 6, 8, 12):
            assert adf_tstat(series, lags=k, constant=True)[0] > ADF_CRIT_CONST["10%"]

    def test_the_observation_count_falls_as_lags_rise(self) -> None:
        """Each added lag costs one observation, which is why the count is returned."""
        counts = [adf_tstat(self._series(), lags=k, constant=True)[1] for k in (1, 4, 8)]
        assert counts == [318, 315, 311]
