"""Khandani and Lo's linear reversal weights, and the helpers they are computed with.

The rule buys the stocks that fell most against the market yesterday and
shorts the ones that rose most, in proportion to how far each moved.
quantitative-trading's Khandani and Lo backtests call :func:`reversal_weights`
on a panel of closes. ithildin-runtime's first shadow run will call it as a
placeholder strategy that claims no edge. That second repository is why the
function lives here rather than in the first.

On a panel of prices with one row per day and one column per stock:

1. ``ret = (p − lag1(p)) / lag1(p)``, which is :func:`daily_returns`.
2. The market's return each day is the mean of the finite returns that day.
3. ``w = −(ret − market) / n``, where ``n`` counts the stocks with a finite
   price that day, not the ones with a finite return.
4. ``w`` is 0 wherever today's price or yesterday's is not finite.

A price of 0 is finite, so step 4 does not catch it. A stock priced 0
yesterday and above 0 today has an infinite return. The market's mean skips
it and its own weight is minus infinity. A stock priced 0 on both days has a
NaN return and a NaN weight. The code is kept as it was copied, so a caller
that can see a zero price refuses it before calling.

Step 3's count is the detail that matters. A stock priced today and not
yesterday has no return, so it takes no part in the market's mean, but it
still counts in ``n``, and step 4 then sets its own weight to 0. So the
weights on a day a stock enters are a little smaller than a count of returns
would give. That is what Chan's ``example3_7.m`` does, read in the mirror
[egorpe/EPChan-QuantitativeTrading](https://github.com/egorpe/EPChan-QuantitativeTrading)
at ``1a7195003cf3e85a806e18867e0af547d17ad5c4``.

The four public helpers keep the names of Chan's MATLAB helpers. That script
calls ``smartsum``, ``smartmean`` and ``lag1``, and ``lag1.m`` calls
``backshift``. They compute what those functions compute on the shapes the
script passes, with two differences.

1. The mirror's ``smartsum`` and ``smartmean`` need a ``dim`` argument the
   script leaves out. Here it is ``axis``, defaulting to 0.
2. :func:`backshift` refuses a shift longer than the rows it is given, where
   MATLAB would hand back more rows than it was given.

- :func:`smartsum` sums the finite entries, and is NaN where none is finite.
  numpy's ``nansum`` gives 0 there instead.
- :func:`smartmean` takes the mean of the finite entries. It skips an
  infinity as well as a NaN.
- :func:`backshift` moves rows later and pads the rows it vacates with NaN.
- :func:`lag1` is ``backshift(1, x)``.

:func:`smartsum` and :func:`smartmean` reduce down each column by default,
the reduction MATLAB's ``dim`` of 1 gives. :func:`reversal_weights` passes
``axis=1`` to reduce across the stocks of one day instead.

Everything here takes arrays and returns arrays, and imports only ``numpy``.
No other module in this package is imported, so a consumer of this one pulls
in nothing else.

**Where it came from.** Every function was copied from quantitative-trading at
``0237abf224515128bfa6473e5cc5c2b367420c82``.
:func:`reversal_weights` and :func:`daily_returns` were in
``src/chan/khandani_lo.py``, and the six helpers, with ``_finite`` and
``_within`` among them, were in ``src/chan/matlab_helpers.py``. Only the
docstrings changed on the way over. With docstrings stripped, each function
here parses to the same tree as its source. quantitative-trading has
committed numbers computed from these bodies, so any change to one is a
re-pin there, and that repository's other helpers still call ``_finite`` and
``_within``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite(x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = np.asarray(x, dtype=float)
    return values, np.isfinite(values)


def smartsum(x: ArrayLike, axis: int = 0) -> NDArray[np.float64] | np.float64:
    """The sum of the finite entries along ``axis``, NaN where none is finite."""
    values, has = _finite(x)
    total = np.where(has, values, 0.0).sum(axis=axis)
    return np.where(has.any(axis=axis), total, np.nan)[()]


def smartmean(x: ArrayLike, axis: int = 0) -> NDArray[np.float64] | np.float64:
    """The mean of the finite entries along ``axis``, NaN where none is finite."""
    values, has = _finite(x)
    count = has.sum(axis=axis)
    total = np.where(has, values, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)[()]


def _within(periods: int, x: ArrayLike, name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=float)
    if periods > len(values):
        # MATLAB would hand back more rows than it was given, and nothing here
        # shifts that far, so this refuses rather than copying that.
        raise ValueError(f"{name} by {periods} rows is longer than the {len(values)} given")
    return values


def backshift(periods: int, x: ArrayLike) -> NDArray[np.float64]:
    """Each row moved ``periods`` rows later, with NaN in the rows it vacates."""
    if periods < 0:
        raise ValueError(
            f"backshift moves rows later, so periods must be at least 0, not {periods}"
        )
    values = _within(periods, x, "backshift")
    shifted = np.full_like(values, np.nan)
    shifted[periods:] = values[: len(values) - periods]
    return shifted


def lag1(x: ArrayLike) -> NDArray[np.float64]:
    """Each row moved one row later, which is ``backshift(1, x)``."""
    return backshift(1, x)


def daily_returns(prices: np.ndarray) -> np.ndarray:
    """Each stock's simple return on each day, NaN where either price is missing."""
    previous = lag1(prices)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (prices - previous) / previous


def reversal_weights(prices: np.ndarray) -> np.ndarray:
    """Minus each stock's return against the market, over the count of priced stocks.

    ``prices`` has one row per day and one column per stock. The first row's
    weights are all 0, because no stock has a price the day before. A stock
    priced today and not yesterday counts in the divisor, takes no part in the
    market's mean, and has its own weight set to 0, as the module docstring
    explains. On prices that are positive wherever they are finite, each row's
    weights sum to zero.
    """
    returns = daily_returns(prices)
    market = smartmean(returns, axis=1)
    priced = smartsum(np.isfinite(prices).astype(float), axis=1)
    weights = -(returns - market[:, None]) / priced[:, None]
    weights[~np.isfinite(prices) | ~np.isfinite(lag1(prices))] = 0.0
    return weights
