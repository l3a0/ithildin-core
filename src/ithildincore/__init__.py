"""Estimators shared by more than one repository.

A module earns a place here by having consumers in two repositories. Anything
with one consumer belongs in that repository, where it can carry that
project's vocabulary and change without a release.

Five modules meet the bar today.

- :mod:`ithildincore.timeseries`, the regression and unit-root primitives.
- :mod:`ithildincore.stats`, the autocorrelation-robust significance block.
- :mod:`ithildincore.reversal`, Khandani and Lo's linear reversal weights.
- :mod:`ithildincore.whiteboard`, whiteboard sketches drawn as looping GIFs,
  which needs the ``whiteboard`` extra.
- :mod:`ithildincore.palette`, the colours the sketches and charts draw in.

The arithmetic in each, and every pixel of a sketch, is pinned by regression
tests in the consuming repositories. Reordering a sum or swapping one ``numpy`` call for another
moves numbers those repositories have committed to disk, so a change to a
function body is a deliberate re-pin of every consumer rather than a tidy-up.
Version this package and let a consumer upgrade when it is ready to re-pin.
"""

__all__ = ["palette", "reversal", "stats", "timeseries", "whiteboard"]
__version__ = "0.5.0"
