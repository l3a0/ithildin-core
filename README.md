# quantcore

Estimators shared by more than one of this account's quantitative
repositories, so that one calculation has one implementation.

## Why a separate package

Two repositories were computing the same least squares. `timeseries.py` lived
in [trading-strategies](https://github.com/l3a0/trading-strategies) as
`common/timeseries.py` and in
[quantitative-trading](https://github.com/l3a0/quantitative-trading) as
`src/chan/timeseries.py`, and the two parsed to the same tree once docstrings
were set aside. Two copies of one calculation drift without either one looking
wrong, and the drift surfaces as a result that moved with nothing in the diff
to explain it.

The fix is not a rule about keeping them in step. It is having one of them.

## What is here

Two modules, each with consumers in both repositories.

1. **`quantcore.timeseries`** is the regression and unit-root work:
   ordinary least squares, the Augmented Dickey-Fuller t-statistic at a fixed
   lag, the Ornstein-Uhlenbeck half-life, and the MacKinnon (2010) critical
   values for the plain and Engle-Granger tests.
2. **`quantcore.stats`** is the significance block for a series whose
   observations are not independent. It reports the ordinary t-statistic and
   the Newey-West one side by side, so the gap between them is visible rather
   than a matter of which estimator someone picked.

A module earns a place here by having consumers in two repositories. Anything
with one consumer belongs in that repository, where it can carry that
project's vocabulary and change without a release. That bar is what keeps this
package from becoming a second home for code nobody shares.

## The rule that matters: a body change is a re-pin

Consuming repositories commit numbers computed from this code. Those numbers
sit in test assertions, in committed ledgers, and in published documents.
Reordering a sum or swapping one `numpy` call for another moves them.

So a change to a function body is a deliberate re-pin of every consumer, not a
tidy-up. That is what the version is for. A consumer upgrades when it is ready
to re-pin, and never by floating.

Adding a function is not a re-pin and does not carry that cost.

## Installing it

A consumer pins an exact tag, and commits whatever lockfile it uses so the
resolved commit is recorded.

```bash
uv add "quantcore @ git+https://github.com/l3a0/quant-core@v0.1.0"
```

For a repository that installs from `requirements.txt` rather than a
lockfile:

```text
quantcore @ git+https://github.com/l3a0/quant-core@v0.1.0
```

## Using it

```python
import numpy as np

from quantcore.stats import newey_west_summary
from quantcore.timeseries import ADF_CRIT_CONST, adf_tstat, ols

fit = ols(y, np.column_stack([x, np.ones(len(x))]))  # explicit intercept
tstat, nobs = adf_tstat(fit.resid, lags=1, constant=False)
reverts = tstat < ADF_CRIT_CONST["5%"]

summary = newey_west_summary(daily_returns)
print(summary.t_naive, summary.t_newey_west, summary.lag)
```

Two defaults are load-bearing rather than conveniences, and both are there to
stop a result changing for a reason nobody wrote down.

1. `ols` fits no implicit intercept. Add a column of ones when one is wanted.
   A with-intercept fit and a through-origin fit then look different where the
   caller writes them, which matters because published figures are often
   printed from one specification and read as though they came from the other.
2. `adf_tstat` uses a fixed lag. Letting the lag be chosen from the data can
   reverse a verdict on the same series, and
   [tests/test_timeseries.py](tests/test_timeseries.py) holds a case where it
   does.

## Running the checks

```bash
uv sync --dev
uv run ruff check
uv run ruff format --check
uv run pytest
```

markdownlint has no Python package, so it runs in CI rather than locally.

## Where this came from

Seeded from [l3a0/repo-template](https://github.com/l3a0/repo-template), which
carries the agent instructions, CI, and GitHub-side policy shared with
[marketlake](https://github.com/l3a0/marketlake),
[trading-strategies](https://github.com/l3a0/trading-strategies) and
[quantitative-trading](https://github.com/l3a0/quantitative-trading).

Both modules came from `trading-strategies`, where they were written and where
their arithmetic was pinned by consumers that are still there. The code was
copied unchanged and verified so: stripping docstrings from both sides leaves
files that parse to the same tree. Only the prose was rewritten, to drop
vocabulary that meant something in that repository and nothing here.
