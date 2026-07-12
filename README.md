# Fine-Tuning 1-1 — Noosa Council

A ~60-minute presentation and live [marimo](https://marimo.io) notebook introducing
LLM fine-tuning for a local government audience: when to fine-tune (vs RAG), how LoRA
works, generating synthetic training data when real data can't be used, and a real
before/after comparison on a locally fine-tuned model — plus a short look inside the
model with TransformerLens/circuitsvis.

By [Leo Alves](https://azl.au) — Azul Labs Pty Ltd.

## The demo

A **"Noosa Council service assistant"**: synthetic resident enquiries (rates, bins,
permits, noise, beaches and trails) are generated with a locally hosted LLM, filtered,
and used to LoRA-fine-tune `Qwen/Qwen2.5-3B-Instruct` into a consistent, plain-language
council voice. The whole pipeline runs on our own hardware — no resident data ever
leaves the building.

The talk itself does **no live training**: all heavy steps run during the week before
and write their outputs to `cache/`, so the notebook works offline and GPU-free.

## Layout

| Path | Purpose |
|---|---|
| `PLAN.md` | Full plan: slide arc, scripts spec, runbook, risks |
| `STYLE.md` | Design plan: brand tokens, slide templates, validated chart palette |
| `notebook.py` | The presentation (marimo slides layout) + demo cells |
| `scripts/generate_data.py` | Synthetic enquiries/answers via an OpenAI-compatible endpoint |
| `scripts/filter_data.py` | Dedup + quality filtering → `data/train.jsonl` |
| `scripts/train.py` | LoRA fine-tune (peft + trl) → `adapters/` |
| `scripts/evaluate.py` | Base vs tuned on held-out prompts → `cache/comparisons.json` |
| `scripts/merge_and_interp.py` | Merge adapter + TransformerLens artifacts → `cache/interp/` |
| `assets/` | Azul Labs logo and brand theme CSS |

## Quick start

Everything runs through one typer CLI, installed as `tokenizer`
(`alias tokenizer="uv run tokenizer"` if you want it bare):

```bash
uv sync

# 1. Generate synthetic data
uv run tokenizer generate --base-url http://localhost:8000/v1 --model <model> --n 1500

# 2. Filter and split
uv run tokenizer filter

# 3. Fine-tune (RTX 5090: ~20-40 min)
uv run tokenizer train

# 4. Compare base vs fine-tuned
uv run tokenizer evaluate

# 5. Rebuild the deck theme after editing assets/theme-src.css
uv run tokenizer build-theme

# Run the presentation
uv run marimo run notebook.py
# ...or edit it
uv run marimo edit notebook.py
```

See `PLAN.md` for the full runbook (smoke-test on ~200 examples before the full
generation run) and the risk checklist.
