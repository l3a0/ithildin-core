"""Estimators shared by more than one repository.

A module earns a place here by having consumers in two repositories. Anything
with one consumer belongs in that repository, where it can carry that
project's vocabulary and change without a release.

Two modules meet the bar today.

- :mod:`quantcore.timeseries`, the regression and unit-root primitives.
- :mod:`quantcore.stats`, the autocorrelation-robust significance block.

The arithmetic in both is pinned by regression tests in the consuming
repositories. Reordering a sum or swapping one ``numpy`` call for another
moves numbers those repositories have committed to disk, so a change to a
function body is a deliberate re-pin of every consumer rather than a tidy-up.
Version this package and let a consumer upgrade when it is ready to re-pin.
"""

__all__ = ["stats", "timeseries"]
__version__ = "0.1.0"
