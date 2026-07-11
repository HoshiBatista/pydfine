# Contributing to pydfine

Thanks for helping build a config-first, ultralytics-style wrapper around D-FINE.

## Start here

- **[`AGENTS.md`](AGENTS.md)** — the canonical build guide (design principles, layout,
  definition of done). Read it first.
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — pick the lowest unchecked task in the
  active phase; one task per pull request.
- **[`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md)** — the source of truth for
  public parameter names and defaults.

## Dev setup

```bash
python -m pip install -e ".[dev]"   # editable install + ruff/pytest/pre-commit
pre-commit install                  # run hooks automatically on commit
```

## Before you push

```bash
ruff format . && ruff check . && pytest -q
dfine models                        # presets resolve?
```

All three must be green — CI runs the same checks on Python 3.9–3.13.

## Ground rules

- **No YAML or registry on the user path.** All options are typed fields on
  `DFINEConfig`. See `AGENTS.md` §2 and §10.
- **Don't fabricate defaults.** Verify against upstream `D-FINE/src/` or the configs;
  mark `# TODO(verify)` and open a roadmap note if you truly can't confirm.
- **Never edit the `D-FINE/` clone** — it is a read-only reference checkout.
- **Every new module ships with a test.** Keep diffs small and reviewable.
- Public parameter names are a stable contract; don't rename them to suit a backend.

## Commit / PR

- Keep each PR focused on one roadmap task; tick its checkbox when green.
- Fill out the PR template checklist (the "Definition of Done" from `AGENTS.md` §9).
