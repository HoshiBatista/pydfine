# CLAUDE.md

**Read [`AGENTS.md`](AGENTS.md) first — it is the canonical build guide.** This file
only adds Claude Code–specific workflow notes. If anything here conflicts with
`AGENTS.md`, `AGENTS.md` wins.

## The one-line mission

Turn D-FINE into a `pip install`-able, ultralytics-style library where the whole
model is configured by **typed Python params on one class (`DFINE`)** — no YAML.

## Where to look before coding

- `docs/ARCHITECTURE.md` — how the model works + module→param map.
- `docs/CONFIG_REFERENCE.md` — the full parameter list + per-size presets (source of truth for names/defaults).
- `docs/ROADMAP.md` — pick the lowest unchecked task in the active phase.

## Working style for this repo

- **Plan before editing.** For any non-trivial task, outline the files you'll touch
  and the test you'll add, then implement.
- **One task per change set.** Keep diffs small and reviewable; don't refactor
  unrelated code in the same pass.
- **Verify, don't guess.** When a default value or layer shape is uncertain, check
  upstream `D-FINE/src/` (or its `configs/*.yml`) rather than inventing it. Mark
  `# TODO(verify)` if you truly can't confirm, and add a roadmap note.
- **Path A is the active path.** We port upstream modules into
  `dfine/backends/native/` (registry/YAML stripped, `from_config(cfg)` added),
  preserving layer/param names for `.pth` parity. `transformers` is not a dependency.
- **Parity is the bar.** A preset model must load the matching upstream `.pth` and
  reproduce its output. Prefer adding a parity test over asserting it works.
- **Keep the public API backend-agnostic** (see AGENTS.md §3). Backends live behind
  `dfine/backends/`; `DFINE(...)` kwargs never leak backend details.

## Fast checks

```bash
ruff format . && ruff check . && pytest -q
dfine models          # presets resolve?
```

## Definition of done

Use the checklist in `AGENTS.md` §9. Tick the roadmap box when green.

## Guardrails (recap — full list in AGENTS.md §10)

No YAML on the user path · no registry/`create()` · don't rename public params to
suit a backend · don't fabricate defaults · keep upstream `LICENSE`/attribution.
