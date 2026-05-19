"""Tiny stand-ins for the pytest fixtures used by the `__main__` blocks
in the test files. Lets `python tests/test_X.py` work the same as
`pytest tests/test_X.py` for tests that depend on `tmp_path` /
`monkeypatch`.

Underscore-prefixed so pytest doesn't try to collect it as a test file.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


class _Monkeypatch:
    """Minimal stand-in for pytest's `monkeypatch` fixture. Supports
    `.chdir(path)` and `.setattr(target, name, value)` and undoes all
    of them on `.undo()`."""

    def __init__(self):
        self._undo_stack: list = []

    def chdir(self, path):
        prev = Path.cwd()
        os.chdir(path)
        self._undo_stack.append(lambda: os.chdir(prev))

    def setattr(self, target, name, value):
        prev = getattr(target, name)
        setattr(target, name, value)
        self._undo_stack.append(lambda: setattr(target, name, prev))

    def undo(self):
        while self._undo_stack:
            self._undo_stack.pop()()


@contextlib.contextmanager
def fixtures(*, monkeypatch: bool = False):
    """Context manager that yields a (tmp_path, monkeypatch?) tuple
    matching the pytest fixture API. Cleans both up on exit.

    Usage in a __main__ block:

        with fixtures() as tp:
            test_uses_tmp_path(tp)

        with fixtures(monkeypatch=True) as (tp, mp):
            test_uses_both(tp, mp)
    """
    with tempfile.TemporaryDirectory() as d:
        mp = _Monkeypatch() if monkeypatch else None
        try:
            yield (Path(d), mp) if monkeypatch else Path(d)
        finally:
            if mp is not None:
                mp.undo()
