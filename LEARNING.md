# LEARNING.md

A personal record of what I learned, bugs I fixed, and experiments I ran while
working through CS336 Assignment 1. Newest entries on top.

> Format for each entry:
> - **Date** — short title
> - **Context:** what I was doing
> - **Learning / Bug / Experiment:** the takeaway
> - **Resolution / Result:** what I did, what worked

---

## Learnings & Bug-fixes

### 2026-06-19 — FFN (SwiGLU) weight-shape convention: `(in, out)` vs `(out, in)`
- **Context:** Merged upstream `26.0.1`, which *corrected* the docstrings for the
  SwiGLU FFN weights in `tests/adapters.py`.
- **Learning:** The reference/grader state dict stores Linear weights in PyTorch's
  standard `(out_features, in_features)` convention:
  - `ffn.w1.weight`, `ffn.w3.weight` → `(d_ff, d_model)` (project up)
  - `ffn.w2.weight` → `(d_model, d_ff)` (project down)

  My own `Linear` class deliberately stores `self.W` as `(in_features, out_features)`
  — the *transpose* — "for memory ordering reasons." So when loading reference
  weights into my model I must `.T` them (see `tests/adapters.py`:
  `"ffn.W1.W": weights["ffn.w1.weight"].T`).
- **Resolution:** No code change needed — my `run_swiglu` annotations already used the
  correct shapes, and `load_state_dict()` strictly checks shapes, so a wrong transpose
  would hard-fail (since `d_ff != d_model`) rather than pass silently. All FFN/transformer
  tests pass. The upstream change was docstring-only.
- **Takeaway:** Be explicit about weight storage convention. `load_state_dict()` is a
  free shape-checker — lean on it.

### 2026-06-19 — Merging upstream into my fork (line-ending noise)
- **Context:** Added `upstream` remote (`stanford-cs336/assignment1-basics`) and merged.
- **Bug/Gotcha:** The raw diff looked huge (~135k lines) but most was **CRLF vs LF**
  line-ending noise in test fixtures — no real content change. There's no
  `.gitattributes` and `core.autocrlf` is unset.
- **Resolution:** Resolved conflicts by file role: took upstream for files I only had
  EOL diffs on (tests/infra), kept my own `tests/adapters.py` (my implementation),
  hand-merged `pyproject.toml` (upstream version bumps + my `submitit/jupyter/notebook`
  extras), and regenerated `uv.lock` with `uv lock`.
- **TODO (optional):** Add a `.gitattributes` with `* text=auto eol=lf` to kill the
  EOL noise for future merges.

---

## Experiments

### Template
- **Date:**
- **Goal / Hypothesis:**
- **Setup:** (dataset, vocab size, model dims, lr, batch, steps, hardware)
- **Result:** (loss, throughput, wandb link)
- **Conclusion / Next step:**

---

## Milestones (from git history)
- BPE training (`d34e13f`)
- BPE tokenizer class (`bfb9a81`)
- Basic blocks — Linear, Embedding, RMSNorm, etc. (`a41778e`)
- Multi-head self-attention (`27a39c6`/`bb278c7`)
- Full transformer block (`27a39c6`)
- Full Transformer LM (`0a9bb07`)
- Handout part 3 complete (`4f8dd0b`)
- Cross-entropy loss (`81e3f01`)
- AdamW optimizer (`6288543`)

## Still TODO (currently `NotImplementedError` in `tests/adapters.py`)
- `run_get_batch` — data loading / batching (`test_data.py`)
- `run_silu` — standalone SiLU adapter (`test_model.py::test_silu_matches_pytorch`)
- `run_gradient_clipping` (`test_nn_utils.py`)
- `run_get_lr_cosine_schedule` — LR schedule (`test_optimizer.py`)
- `run_save_checkpoint` / `run_load_checkpoint` (`test_serialization.py`)
- BPE training speed — `test_train_bpe_speed` currently exceeds the 1.5s budget;
  may need optimization (or it's just slow on this machine).
