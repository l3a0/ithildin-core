# ithildincore

Estimators shared by more than one of this account's quantitative
repositories, so that one calculation has one implementation, and the
whiteboard sketches their posts and design docs draw, so that one look has one
implementation too.

## Why the name

`quantcore` was the first name and it lasted one release. That name is taken on
PyPI by an unrelated backtesting package whose 0.1.0 release satisfies a
`quantcore>=0.1` floor, so a consumer that declared the dependency by name
rather than by URL installed a stranger's code. A direct URL closes that, and
both consumers use one, but a name nobody else has is the thing that stops the
question coming back.

`ithildincore` was free on PyPI when this was renamed, and the compound matters:
bare `ithildin` is taken.

Consumers pinned to `v0.1.0` import `quantcore`. From `v0.2.0` the package is
`ithildincore`, which is why a rename that changes no arithmetic still gets a
minor bump rather than a patch.

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

Five modules. The first two each have consumers in two repositories, and the
third gains its second at ithildin-runtime's first shadow run. The last two
draw the whiteboard sketches that quantitative-trading's posts and
ithildin-runtime's design docs carry.

1. **`ithildincore.timeseries`** is the regression and unit-root work:
   ordinary least squares, the Augmented Dickey-Fuller t-statistic at a fixed
   lag, the Ornstein-Uhlenbeck half-life, and the MacKinnon (2010) critical
   values for the plain and Engle-Granger tests.
2. **`ithildincore.stats`** is the significance block for a series whose
   observations are not independent. It reports the ordinary t-statistic and
   the Newey-West one side by side, so the gap between them is visible rather
   than a matter of which estimator someone picked.
3. **`ithildincore.reversal`** is Khandani and Lo's linear reversal. Its
   weights buy the stocks that fell most against the day's mean return and
   short the ones that rose most. It carries the four helpers they are
   computed with, under the names of Chan's MATLAB helpers.
   quantitative-trading's backtests compute from it, and ithildin-runtime's
   first shadow run will run it as a placeholder strategy.
4. **`ithildincore.whiteboard`** draws a sketch stroke by stroke, as if on a
   whiteboard, and writes it as a looping GIF. A scene module describes the
   sketch as data. This module times it, draws it, writes it, and checks that
   no label touches a line, leaves the board or sits on another label.
   quantitative-trading's posts draw their sketches with it, and
   ithildin-runtime's design docs draw theirs.
5. **`ithildincore.palette`** names the colours those sketches and
   quantitative-trading's charts draw in. It holds a colour-blind safe set for
   new work, and the brass set that quantitative-trading's older posts still
   draw in.

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
uv add "ithildincore @ git+https://github.com/l3a0/ithildin-core@v0.5.0"
```

For a repository that installs from `requirements.txt` rather than a
lockfile:

```text
ithildincore @ git+https://github.com/l3a0/ithildin-core@v0.5.0
```

`ithildincore.whiteboard` needs matplotlib and Pillow, and
`ithildincore.palette` needs matplotlib's colour conversions. The estimators
need neither, so the two come with the `whiteboard` extra rather than with
every install. A repository that draws sketches names the extra:

```bash
uv add "ithildincore[whiteboard] @ git+https://github.com/l3a0/ithildin-core@v0.5.0"
```

## Using it

```python
import numpy as np

from ithildincore.reversal import reversal_weights
from ithildincore.stats import newey_west_summary
from ithildincore.timeseries import ADF_CRIT_CONST, adf_tstat, ols

fit = ols(y, np.column_stack([x, np.ones(len(x))]))  # explicit intercept
tstat, nobs = adf_tstat(fit.resid, lags=1, constant=False)
reverts = tstat < ADF_CRIT_CONST["5%"]

summary = newey_west_summary(daily_returns)
print(summary.t_naive, summary.t_newey_west, summary.lag)

weights = reversal_weights(closes)  # one row per day, one column per stock
```

A sketch is a scene of strokes and labels, timed, drawn and checked:

```python
from ithildincore.palette import ACCENT, INKS, MUTED
from ithildincore.whiteboard import Label, Scene, arrow, draft, line, stroke

scene = Scene(
    "A pull back",
    "pull_back",
    (
        stroke("mean", line([(0.4, 2.0), (9.6, 2.0)]), color=MUTED, dashed=True),
        stroke("pull", arrow(6.9, 3.35, 6.9, 2.15), color=ACCENT, frames=6),
        Label("Pulled back?", 7.15, 3.55, names="pull", color=ACCENT),
    ),
)
problems = draft(scene, scratch, moved=True)  # writes pull_back.gif, lists rules broken
```

`draft` writes into a scratch folder. A scene module that commits its GIFs
calls `write_gif(scene, timeline(scene), path, INKS)` instead.

Four defaults are load-bearing rather than conveniences. The first two are
there to stop a result changing for a reason nobody wrote down, the third is
the reduction Chan's scripts use, and the fourth keeps committed GIFs as they
were drawn.

1. `ols` fits no implicit intercept. Add a column of ones when one is wanted.
   A with-intercept fit and a through-origin fit then look different where the
   caller writes them, which matters because published figures are often
   printed from one specification and read as though they came from the other.
2. `adf_tstat` uses a fixed lag. Letting the lag be chosen from the data can
   reverse a verdict on the same series, and
   [tests/test_timeseries.py](tests/test_timeseries.py) holds a case where it
   does.
3. `smartsum` and `smartmean` reduce down each column unless given `axis`.
   `reversal_weights` passes `axis=1`. A caller that leaves `axis` out gets
   one answer per column, and changing the default would move it.
4. `whiteboard.render`, `whiteboard.write_gif` and `whiteboard.palette` draw
   on the brass set unless given other inks, and `whiteboard.dot` is brass
   unless given a colour. quantitative-trading's older posts rely on both, so
   new work passes `palette.INKS` and names its colours, and `draft` flags a
   scene that mixes the two sets.

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

`timeseries` and `stats` came from `trading-strategies`, where they were
written and where their arithmetic was pinned by consumers that are still
there. `reversal` came from `quantitative-trading` at `0237abf2`, for the same
reason. `whiteboard` and `palette` came from `quantitative-trading` at
`715d1262`, where the drawing sat beside one post's scenes and the inks in two
of its modules. The code was copied unchanged and verified so: stripping
docstrings from both sides leaves functions that parse to the same tree. For
`whiteboard` that holds once its renamed helpers are mapped back to their old
names, and the module's docstring lists every change. Only the prose was
rewritten, to drop vocabulary that meant something in the source repository
and nothing here.
