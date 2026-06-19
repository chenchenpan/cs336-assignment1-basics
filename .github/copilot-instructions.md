# Copilot Instructions

These are personal working preferences for AI assistants on this repo. GitHub
Copilot CLI reads this file **in addition to** the root `AGENTS.md`, so the CS336
teaching-assistant rules in `AGENTS.md` still apply in full — this file only adds
my personal conventions on top.

> Course academic-integrity and "teaching-assistant, not solution-generator" rules
> live in `AGENTS.md` (shared, upstream-tracked). Do not duplicate or override them
> here.

## Git & commits
- **Never** add `Co-Authored-By:` trailers (or any AI co-author line) to commit
  messages or PR bodies.
- Keep commit subjects concise and imperative ("Add X", "Fix Y").
- Only commit or push when I explicitly ask. If on `main`, branch first for
  non-trivial work.

## Line endings
- This repo normalizes to **LF** via `.gitattributes` (`* text=auto eol=lf`).
  Do not reintroduce CRLF; it creates noisy diffs when syncing the Stanford
  upstream.

## Repo conventions
- Python environment is managed with **`uv`** — run code/tests via `uv run ...`
  (e.g. `uv run pytest`). The bare `python` shim is not configured.
- My implementation lives in `cs336_basics/`; graded adapters are wired in
  `tests/adapters.py`.
- Personal learning/bug/experiment notes go in `LEARNING.md` (newest on top).
