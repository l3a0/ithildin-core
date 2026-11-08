"""Estimators and whiteboard sketches shared by more than one repository.

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

# whiteboard and palette are left out, because both import matplotlib, which
# only the whiteboard extra installs. A star import would load them and fail on
# an install of the estimators alone.
__all__ = ["reversal", "stats", "timeseries"]
__version__ = "0.5.0"
