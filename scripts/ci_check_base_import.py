"""CI check: a core-only install stays torch-free and usably degrades.

Run in an environment with **only** the base dependencies (``pip install .``, no extras).
Asserts that importing the package and touching the config/registry surface never pulls
in torch, and that building a model without the ``[torch]`` extra fails with a clear,
actionable error. Safe to run anywhere: the no-torch error check is skipped when torch
happens to be installed (e.g. a dev machine), so local runs still pass.
"""

from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    import dfine

    # The config/CLI surface must not require torch.
    dfine.list_presets()
    dfine.list_checkpoints()
    assert "torch" not in sys.modules, "importing dfine pulled in torch"
    print("ok: `import dfine` + presets/checkpoints are torch-free")

    # Building a model without the extra must raise a helpful, actionable error.
    if importlib.util.find_spec("torch") is None:
        try:
            getattr(dfine, "DFINE")  # triggers the lazy __getattr__ import  # noqa: B009
        except AttributeError as exc:
            assert "pydfine[torch]" in str(exc), exc
            print("ok: building a model without torch raises a helpful error")
        else:
            raise SystemExit("expected a helpful error when building a model without torch")
    else:
        print("skip: torch is installed — no-torch model-build error check not applicable")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
