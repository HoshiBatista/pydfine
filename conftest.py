"""Root conftest — makes the repo root importable during test collection.

A few test modules share fixtures via cross-imports (e.g. ``from tests.test_dataset
import _write_split``). Under an editable install (PEP 660) only the ``dfine`` package
is on ``sys.path``, not the repo root, so ``import tests`` fails and collection aborts.
pytest prepends this file's directory (the repo root) to ``sys.path``, which restores
``tests`` as an importable top-level package. Intentionally empty otherwise.
"""
