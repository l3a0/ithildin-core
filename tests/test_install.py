"""The estimators import on an install without the ``whiteboard`` extra.

CI installs the dev group, which pulls the extra in, so nothing there would
notice an estimator that started importing matplotlib. This test runs the
imports in a fresh interpreter where matplotlib and Pillow cannot load.
"""

from __future__ import annotations

import subprocess
import sys

BLOCKED = """
import importlib.abc
import sys


class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"matplotlib", "PIL"}:
            raise ModuleNotFoundError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, Blocked())
import ithildincore
import ithildincore.reversal
import ithildincore.stats
import ithildincore.timeseries
from ithildincore import *  # noqa: F403

loaded = sorted(n for n in sys.modules if n.split(".")[0] in {"matplotlib", "PIL"})
assert not loaded, loaded
try:
    import ithildincore.whiteboard
except ModuleNotFoundError as missing:
    assert "blocked" in str(missing), missing
else:
    raise AssertionError("whiteboard imported without matplotlib")
print("ok")
"""


def test_the_estimators_import_without_matplotlib_or_pillow() -> None:
    run = subprocess.run([sys.executable, "-c", BLOCKED], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "ok"
