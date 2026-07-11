# Fine-Tuning 1-1 — Noosa Council

Presentation + live marimo notebook: why fine-tune, LoRA, synthetic data, and a real
pre/post comparison on a locally fine-tuned model. Bonus: a peek inside the model with
TransformerLens/circuitsvis. Delivered by Leo Alves (Azul Labs).

**Talk length:** ~60 min. No live training — the fine-tune runs during the week before;
the notebook loads cached artifacts so the talk works with no GPU and no internet.

---

## Story

**"Noosa Council service assistant."** Synthetic resident enquiries (rates, bins,
permits, noise complaints, beach/trail questions) fine-tuned into a consistent,
plain-language council voice. The meta-pitch for a council audience: the entire
pipeline — data generation, training, evaluation — runs on our own hardware. No
resident data ever leaves the building.

**Framing that anchors the whole talk:** knowledge belongs in RAG; *behaviour* (tone,
format, policy interpretation, consistency) is what fine-tuning buys you.

## Key technical choices

| Choice | Decision | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` | Ungated, TransformerLens-supported, ~20–40 min LoRA on the 5090, small enough that fine-tuning visibly changes behaviour |
| Method | LoRA (bf16, no quantization) | 32GB VRAM makes QLoRA unnecessary; simpler story |
| Stack | `peft` + `trl` SFTTrainer, ChatML/JSONL data | The default ecosystem path; what attendees will find in every tutorial |
| Synthetic data | OpenAI-compatible endpoint, `--base-url` flag | Works identically against the DGX cluster (DeepSeek) or local vLLM/Ollama (Qwen3.6 27B / Gemma 4 26B) |
| Instruct base, not raw base | Fine-tune the instruct variant | Pre/post shows tone/policy change, not "learned to talk at all" |

## Repo layout

```
tokenizer-finetuning/
├── PLAN.md                  # this file
├── README.md
├── notebook.py              # marimo — slides layout; every demo cell reads from cache/
├── assets/
│   ├── azul-logo.svg        # Azul Labs logo (footer branding)
│   └── theme.css            # brand CSS applied via marimo.App(css_file=...)
├── scripts/
│   ├── generate_data.py     # synthetic enquiries + council-voice answers → data/raw.jsonl
│   ├── filter_data.py       # dedup, length/format checks, topic filter → data/train.jsonl + data/eval_prompts.json
│   ├── train.py             # LoRA fine-tune → adapters/  (+ loss curve → cache/loss.json)
│   ├── evaluate.py          # eval prompts through base + tuned → cache/comparisons.json
│   └── merge_and_interp.py  # merge_and_unload → HookedTransformer → cache/interp/*.json
├── data/                    # raw.jsonl, train.jsonl, eval_prompts.json
├── adapters/                # LoRA adapter checkpoint (~50MB)
└── cache/                   # everything the notebook needs to run offline
```

## Branding & styling (matches azl.au)

Applied via `assets/theme.css`, loaded with `marimo.App(css_file="assets/theme.css")`.

- **Palette:** cream `#FAF8F5` background · midnight `#1A202C` text · slate `#4A5568`
  secondary · gold `#D4A853` accents/highlights · Azul blues `#1E5AA8` / `#3B82C4` /
  `#5BA3D9` for charts, links, and diagram fills
- **Type:** Fraunces (display serif) for slide titles, DM Sans for body/code labels —
  self-host or system-fallback (`Georgia`, `system-ui`) so the deck works offline
- **Footer on every slide:** fixed-position bar — Azul Labs logo (`assets/azul-logo.svg`,
  inlined as data URI in the CSS) + "azl.au" in slate, bottom-right
- **Charts (loss curve, params comparison):** Azul blues as the categorical palette,
  gold for the highlight series, cream background

## Slide/notebook arc (~60 min)

1. **About me** (2 min) — Leo Alves · Azul Labs Pty Ltd ([azl.au](https://azl.au)) ·
   AI engineer & software consultant on the Sunshine Coast · indie researcher.
   15+ years, 50+ projects. "Time to ride the AI wave."
2. **Why fine-tune — and why not** (8 min) — RAG vs fine-tuning decision framing;
   council use cases (service responses, report summaries, plain-language rewriting);
   what fine-tuning is bad at (facts that change).
3. **LoRA in one picture** (8 min) — low-rank adapter diagram; live cell:
   `model.print_trainable_parameters()` → ~0.5% of weights, ~50MB adapter vs 6GB model.
4. **Synthetic data** (12 min) — why synthetic (privacy: we generate data *because* we
   can't ship residents' emails to a training run); live-generate 3–5 examples against
   the local model (cached fallback); the filtering code — dedup, length/format checks,
   topic rejection.
5. **Training** (10 min) — walk `train.py`; show the real loss curve from the week's
   run; what the hyperparameters mean in plain language (rank, epochs, learning rate).
6. **Pre vs post** (12 min) — side-by-side base vs fine-tuned responses on 10 held-out
   prompts, from cache. The money section. Include one "how do we know it worked"
   slide (fixed eval set + judge pass).
7. **What would production cost?** (3 min) — hosting options + monthly numbers (table
   below). Punchline: *"a fine-tuned 3B assistant is a few hundred dollars a month
   always-on in the cloud, a few dollars a month serverless, or roughly a power bill
   on your own hardware — the model is not the expensive part."* Honest caveat: the
   real cost of self-hosting is ops (updates, monitoring, uptime), not electricity.
8. **Bonus: what changed inside the model** (5 min) — circuitsvis attention patterns on
   a council prompt; base-vs-tuned logit-diff on one token position. Wow-moment, not a
   section.
9. **Q&A** (2+ min buffer).

## Hosting & cost slide (numbers as of July 2026 — re-check before the talk)

Serving stack in all cases: vLLM (OpenAI-compatible API; serves merged model, or base
+ LoRA adapter natively). A 3B model is ~6–7GB bf16 — any 16–24GB GPU is ample.

| Option | Example | Always-on monthly | Notes for a council |
|---|---|---|---|
| Rented GPU (marketplace) | Vast.ai / RunPod Community, RTX 4090 @ ~$0.29–0.34/hr | ~US$210–250 | Cheapest 24/7; peer-to-peer hardware — fine for demos, not resident data |
| Rented GPU (secure) | RunPod Secure Cloud 4090 @ $0.59/hr | ~US$430 | Dedicated, single-tenant |
| Hyperscaler | AWS/GCP L4-class @ ~$0.70–0.85/hr | ~US$500–620 | **Sydney region + enterprise compliance** — usually what a council actually needs |
| Serverless per-token | Fireworks (fine-tuned LoRA served at base-model price, ~$0.20/1M tok) | ~US$1–20 at council-scale traffic | No idle cost; cold starts; data leaves your infra; confirm Qwen2.5-3B adapter support |
| Self-host | vLLM on own box (e.g. DGX Spark class, 100–250W avg) | ~A$35–65 electricity @ QLD ~30–35c/kWh | Matches the talk's data-sovereignty story; real cost is ops, not power |

Sources: runpod.io/pricing · vast.ai/pricing · fireworks.ai/pricing ·
docs.fireworks.ai/serverless/pricing (Together dedicated endpoints ~$6.49/hr H100 ≈
US$4,700/mo — cited as the "overkill" contrast point).

## Scripts spec

### `generate_data.py`
- OpenAI-compatible client; `--base-url`, `--model`, `--n`, `--out` flags
- Two-stage prompt: (1) generate diverse resident enquiries across ~10 topic seeds
  (rates, waste, permits, noise, parking, pets, trails/beaches, events, planning,
  general), varied register (angry, confused, brief, rambling); (2) answer each in the
  target council voice (plain language, empathetic, accurate structure, signposts to
  next steps). Persona/style card embedded in the system prompt.
- Output: `data/raw.jsonl`, ChatML messages format (`system`/`user`/`assistant`).

### `filter_data.py`
- Exact + near-dup removal (normalised text hashing), length bounds, format validation
  (parses as ChatML, single-turn), keyword topic filter, refusal/meta-text rejection.
- Splits off 10 held-out prompts → `data/eval_prompts.json`; rest →
  `data/train.jsonl`. Prints a kept/dropped report (this report is itself a slide).

### `train.py`
- `peft` LoRA (r=16, alpha=32, target all attention + MLP projections), `trl`
  `SFTTrainer`, bf16, ~3 epochs, cosine schedule; assistant-only loss masking.
- Saves adapter to `adapters/`, training-loss history to `cache/loss.json`.

### `evaluate.py`
- Loads base and base+adapter; generates for the 10 eval prompts with identical
  sampling settings; writes `cache/comparisons.json`.
- Optional `--judge` pass: score each pair against the style card via the local
  endpoint (consistency/tone/structure) → adds scores to the JSON.

### `merge_and_interp.py`
- `merge_and_unload()` the adapter → save merged model → load both base and merged
  into `HookedTransformer` (TransformerLens **cannot** load a PEFT-wrapped model —
  merging first is mandatory).
- Exports for one fixed prompt: attention patterns (circuitsvis JSON), per-layer
  logit-lens on the answer's first token, base-vs-tuned logit diff.
- All artifacts → `cache/interp/` so the notebook renders without a GPU.

## Week runbook (order matters)

1. **Setup:** `uv add torch transformers peft trl datasets accelerate marimo altair
   openai transformer-lens circuitsvis` (drop Python pin to 3.12 if any dep fights 3.14).
2. **Smoke test end-to-end on ~200 examples first** — generate → filter → train (1
   epoch) → evaluate. Surfaces format problems in minutes, not after hours of
   generation.
3. **Full generation:** 1–2k examples against DeepSeek on the DGX (or local 27B).
   Eyeball ~30 random samples before training.
4. **Full train** on the 5090 → `evaluate.py` → **check the pre/post is actually
   dramatic.** If too subtle: sharpen the voice in the style card (make it distinctive
   — greeting format, sign-off, structure), bump epochs or rank, regenerate.
5. **Interp:** `merge_and_interp.py` → verify circuitsvis renders in the notebook.
6. **Dress rehearsal:** run the notebook with WiFi off — every cell must have a cache
   path. Export/print fallback of the slides as PDF in case marimo misbehaves on the
   venue machine.

## Risks & mitigations

- **Live demo fails at venue** → only optional-live cell is the generation demo; all
  else cached. PDF export as last resort.
- **Pre/post difference underwhelming** → mitigation is a *distinctive* council voice
  in the data (structure + sign-off make change visible even to non-technical eyes);
  checked in step 4 with time to iterate.
- **TransformerLens + PEFT incompatibility** → merge first (handled in script).
- **Python 3.14 dependency friction** → drop to 3.12; no content impact.
- **Qwen2.5-3B behaves oddly with assistant-masking or chat template** → smoke test
  (step 2) catches this early; fallback base model: `meta-llama/Llama-3.2-3B-Instruct`.
