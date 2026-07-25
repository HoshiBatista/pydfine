# Contributing to pydfine

Thanks for helping build a config-first, ultralytics-style wrapper around D-FINE.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). To report
a security vulnerability, see [`SECURITY.md`](SECURITY.md) — do **not** open a public issue.

## Start here

- **[`AGENTS.md`](AGENTS.md)** — the canonical build guide (design principles, layout,
  definition of done). Read it first.
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — pick the lowest unchecked task in the
  active phase; one task per pull request.
- **[`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md)** — the source of truth for
  public parameter names and defaults.

## Dev setup

```bash
python -m pip install -e ".[dev]"          # ruff/pytest/pre-commit (config tests only)
python -m pip install -e ".[dev,torch]"    # + torch: runs the native model tests too
pre-commit install                         # run hooks automatically on commit
```

The config/CLI tests need no torch; the backbone/encoder/decoder tests
(`test_backbone.py`, `test_encoder.py`, `test_decoder.py`) `importorskip("torch")`,
so install the `torch` extra (or `requirements.txt`) to exercise them.

## Before you push

```bash
ruff format . && ruff check . && pytest -q
dfine models                        # presets resolve?
```

All three must be green — CI runs the same checks on Python 3.9–3.13.

### Coverage badge

The README coverage badge is **static** — the number is hard-coded, not auto-generated.
If a change moves total coverage materially, measure it and update the badge by hand:

```bash
pytest --cov=dfine -q      # read the TOTAL % from the report
```

Then edit the `coverage-<N>%25` value in the `[![Coverage]...]` badge near the top of
[`README.md`](README.md).

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
- Add an entry under `## [Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) for any
  user-visible change.
- Commit messages use a conventional prefix (`feat`, `fix`, `docs`, `test`, `refactor`,
  `chore`), e.g. `fix(val): align ConfusionMatrix IoU threshold to 0.5`.
