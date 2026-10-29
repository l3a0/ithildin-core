"""What the reversal weights and their helpers compute, on inputs small enough to check by hand.

Each helper is held on an input where it and the numpy default disagree, with
the default's answer asserted beside it. So a helper quietly swapped for the
default fails here rather than agreeing on easy inputs. The cases come from
quantitative-trading's ``tests/test_matlab_helpers.py`` at ``0237abf2``.

The weights are held on a three-stock panel worked out by hand, from
quantitative-trading's ``tests/test_khandani_lo.py`` at the same commit. One
stock enters on its second day, which is the case that separates the rule
from the obvious reading of it: the divisor counts closes, not returns, and
the entering stock's own weight is 0. Exact equality is used wherever the
arithmetic is exact, because quantitative-trading has committed numbers that
trace to these bodies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ithildincore.reversal import (
    backshift,
    daily_returns,
    lag1,
    reversal_weights,
    smartmean,
    smartsum,
)

NAN = math.nan


class TestSmartmean:
    def test_it_skips_a_nan_where_the_plain_mean_does_not(self) -> None:
        x = [1.0, NAN, 3.0]
        assert smartmean(x) == 2.0
        assert math.isnan(np.mean(x))

    def test_it_skips_an_infinity_where_nanmean_does_not(self) -> None:
        x = [1.0, math.inf, 3.0]
        assert smartmean(x) == 2.0
        assert np.nanmean(x) == math.inf

    def test_a_column_with_nothing_finite_is_nan(self) -> None:
        assert math.isnan(smartmean([NAN, NAN]))

    def test_it_reduces_down_each_column_by_default(self) -> None:
        x = np.array([[1.0, NAN], [3.0, 4.0]])
        np.testing.assert_array_equal(smartmean(x), [2.0, 4.0])
        np.testing.assert_array_equal(smartmean(x, axis=1), [1.0, 3.5])

    def test_a_vector_gives_a_scalar(self) -> None:
        assert np.ndim(smartmean([1.0, 2.0])) == 0


class TestSmartsum:
    def test_a_row_with_nothing_finite_is_nan_where_nansum_gives_zero(self) -> None:
        assert math.isnan(smartsum([NAN, NAN]))
        assert np.nansum([NAN, NAN]) == 0.0

    def test_it_skips_a_nan(self) -> None:
        np.testing.assert_array_equal(
            smartsum(np.array([[1.0, NAN], [2.0, NAN]]), axis=1), [1.0, 2.0]
        )

    def test_it_skips_an_infinity_where_nansum_does_not(self) -> None:
        assert smartsum([1.0, math.inf, 2.0]) == 3.0
        assert np.nansum([1.0, math.inf, 2.0]) == math.inf

    def test_it_reduces_down_each_column_by_default(self) -> None:
        x = np.array([[1.0, NAN], [3.0, NAN]])
        np.testing.assert_array_equal(smartsum(x), [4.0, NAN])

    def test_a_vector_gives_a_scalar(self) -> None:
        assert np.ndim(smartsum([1.0, 2.0])) == 0


class TestShifts:
    def test_backshift_moves_rows_later_and_pads_with_nan(self) -> None:
        np.testing.assert_array_equal(backshift(2, [1.0, 2.0, 3.0, 4.0]), [NAN, NAN, 1.0, 2.0])

    def test_lag1_is_a_backshift_of_one(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(lag1(x), [[NAN, NAN], [1.0, 2.0]])

    def test_a_shift_of_zero_returns_the_rows_unmoved(self) -> None:
        np.testing.assert_array_equal(backshift(0, [1.0, 2.0]), [1.0, 2.0])

    def test_a_shift_of_every_row_leaves_only_nan(self) -> None:
        np.testing.assert_array_equal(backshift(3, [1.0, 2.0, 3.0]), [NAN, NAN, NAN])

    def test_a_shift_longer_than_the_rows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="backshift by 5 rows is longer than the 3 given"):
            backshift(5, [1.0, 2.0, 3.0])

    def test_a_negative_shift_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="periods must be at least 0, not -1"):
            backshift(-1, [1.0])

    def test_integers_come_back_as_floats(self) -> None:
        shifted = backshift(1, [1, 2])
        assert shifted.dtype == np.float64
        np.testing.assert_array_equal(shifted, [NAN, 1.0])


# --- a panel small enough to work by hand ------------------------------------------
#
# Three stocks over four days. C is first priced on day 1, so it counts in that
# day's n without having a return.
#
#   day   A      B       C
#   0     10     20      NaN
#   1     11     19      5       returns  +0.10  -0.05  NaN   market +0.025
#   2     11     19.95   5.5     returns   0     +0.05  +0.10 market +0.05
#   3     12.1   19.95   5.5     returns  +0.10   0      0    market +0.0333

HAND = np.array(
    [
        [10.0, 20.0, np.nan],
        [11.0, 19.0, 5.0],
        [11.0, 19.95, 5.5],
        [12.1, 19.95, 5.5],
    ]
)


class TestDailyReturns:
    def test_each_day_against_the_day_before(self) -> None:
        returns = daily_returns(HAND)
        assert np.isnan(returns[0]).all()
        assert returns[1, :2] == pytest.approx([0.1, -0.05], abs=1e-12)
        assert returns[2] == pytest.approx([0.0, 0.05, 0.1], abs=1e-12)

    def test_a_missing_price_on_either_day_gives_nan(self) -> None:
        returns = daily_returns(HAND)
        assert np.isnan(returns[0, 2])
        assert np.isnan(returns[1, 2])

    def test_a_zero_price_yesterday_gives_infinity_without_a_warning(self) -> None:
        with np.errstate(all="raise"):
            returns = daily_returns(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert returns[1, 0] == math.inf
        assert np.isnan(returns[1, 1])


class TestReversalWeightsByHand:
    def test_the_weights(self) -> None:
        """Day 1 divides by 3, the count of closes, not 2, the count of returns."""
        weights = reversal_weights(HAND)
        assert weights[0].tolist() == [0.0, 0.0, 0.0]
        assert weights[1] == pytest.approx([-0.025, 0.025, 0.0], abs=1e-12)
        assert weights[2] == pytest.approx([0.05 / 3, 0.0, -0.05 / 3], abs=1e-12)
        assert weights[3] == pytest.approx(
            [-(0.1 - 0.1 / 3) / 3, (0.1 / 3) / 3, (0.1 / 3) / 3], abs=1e-12
        )

    def test_the_divisor_counts_closes_rather_than_returns(self) -> None:
        """A count of returns on day 1 would give -0.0375, not -0.025."""
        weights = reversal_weights(HAND)
        assert weights[1, 0] == pytest.approx(-0.075 / 3, abs=1e-15)
        assert weights[1, 0] != pytest.approx(-0.075 / 2, abs=1e-6)

    def test_a_stock_priced_today_and_not_yesterday_has_weight_zero(self) -> None:
        assert reversal_weights(HAND)[1, 2] == 0.0

    def test_a_stock_unpriced_today_has_weight_zero(self) -> None:
        prices = HAND.copy()
        prices[2, 1] = np.nan
        weights = reversal_weights(prices)
        assert weights[2, 1] == 0.0
        assert weights[3, 1] == 0.0

    def test_the_weights_sum_to_zero_every_day(self) -> None:
        assert reversal_weights(HAND).sum(axis=1) == pytest.approx(0.0, abs=1e-15)

    def test_buy_the_loser_and_short_the_winner(self) -> None:
        """The sign. A stock that beat the market is shorted."""
        weights = reversal_weights(HAND)
        assert weights[1, 0] < 0 < weights[1, 1]

    def test_two_stocks_over_two_days(self) -> None:
        """The shape ithildin-runtime passes: yesterday's closes over today's prices.

        The returns are +1% and -1%, so the market is 0 and each weight is its
        return, negated, over the two priced stocks.
        """
        weights = reversal_weights(np.array([[100.0, 200.0], [101.0, 198.0]]))
        assert weights[0].tolist() == [0.0, 0.0]
        assert weights[1] == pytest.approx([-0.005, 0.005], abs=1e-15)


class TestReversalWeightsAtTheEdges:
    def test_one_day_has_every_weight_zero(self) -> None:
        assert reversal_weights(np.array([[10.0, 20.0]])).tolist() == [[0.0, 0.0]]

    def test_a_day_with_no_return_at_all_has_every_weight_zero(self) -> None:
        prices = np.array([[10.0, np.nan], [np.nan, 20.0], [11.0, 21.0]])
        assert reversal_weights(prices)[1].tolist() == [0.0, 0.0]

    def test_an_empty_panel_is_refused_by_the_shift(self) -> None:
        with pytest.raises(ValueError, match="backshift by 1 rows is longer than the 0 given"):
            reversal_weights(np.empty((0, 3)))

    def test_it_does_not_change_the_prices_it_is_given(self) -> None:
        prices = HAND.copy()
        reversal_weights(prices)
        np.testing.assert_array_equal(prices, HAND)
