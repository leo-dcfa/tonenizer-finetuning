"""Peek inside the fine-tuned model: merge the adapter, export interp artifacts.

TransformerLens cannot load a PEFT-wrapped model, so the LoRA adapter is first
merged into the base weights (merge_and_unload). Both the base and the merged
model are then run through TransformerLens on one fixed council enquiry, and
three JSON artifacts land in cache/interp/ for the notebook to render offline:

  attention.json   — attention patterns for a few layers (circuitsvis format)
  logit_lens.json  — p(first answer token) by layer, base vs tuned
  logit_diff.json  — top tokens by logit difference (tuned - base) at the
                     first answer position

Example usage:
    uv run tokenizer interp
    uv run tokenizer interp --adapter adapters/smoke --layers 0,17,35
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import typer
from peft import PeftModel
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.generate_data import SYSTEM_PROMPT

INTERP_DIR = Path("cache/interp")

# One fixed enquiry so every artifact (and every slide) shows the same example.
INTERP_ENQUIRY = (
    "My yellow recycling bin hasn't been collected for two weeks now and it's overflowing. "
    "What do I need to do to get someone to come and empty it?"
)


def build_prompt_tokens(tokenizer: AutoTokenizer, device: str) -> torch.Tensor:
    """Chat-templated prompt up to (and including) the assistant-start marker."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INTERP_ENQUIRY},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return enc["input_ids"].to(device)


def run_model(
    name: str,
    hf_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_name: str,
    tokens: torch.Tensor,
    layers: list[int],
) -> dict:
    """Load into TransformerLens, cache one forward pass, extract the artifacts."""
    print(f"  loading {name} into HookedTransformer (bf16)")
    model = HookedTransformer.from_pretrained(
        base_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device="cuda",
    )
    model.eval()

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    final_logits = logits[0, -1].float().cpu()

    # Logit lens: project every layer's residual stream (at the last position,
    # where the first answer token is being decided) through the unembed.
    per_layer_logits = []
    with torch.no_grad():
        for layer in range(model.cfg.n_layers):
            resid = cache["resid_post", layer][:, -1:, :]  # keep [batch, pos, d_model]
            per_layer_logits.append(model.unembed(model.ln_final(resid))[0, -1].float().cpu())

    # Attention patterns for the requested layers: [heads, dst, src].
    patterns = {
        str(layer): cache["pattern", layer][0].float().cpu().numpy().round(4).tolist()
        for layer in layers
    }

    del cache, model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "final_logits": final_logits,
        "per_layer_logits": per_layer_logits,
        "patterns": patterns,
    }


def main(
    model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct"),
    adapter: Path = typer.Option(Path("adapters/council-voice")),
    layers: str = typer.Option("2,17,34", help="comma-separated layer indices for attention maps"),
    top_k: int = typer.Option(20, help="tokens to keep in the logit-diff chart"),
) -> None:
    """Merge the LoRA adapter and export TransformerLens artifacts for the deck."""
    layer_list = [int(x) for x in layers.split(",")]
    INTERP_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = build_prompt_tokens(tokenizer, "cuda")
    token_strs = [tokenizer.decode(t) for t in tokens[0]]

    # -- tuned pass: load base, attach adapter, merge into plain weights --------
    print("[1/3] Merging adapter into base weights")
    hf_base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(hf_base, str(adapter)).merge_and_unload()
    tuned = run_model("tuned (merged)", merged, tokenizer, model, tokens, layer_list)
    del merged, hf_base
    gc.collect()
    torch.cuda.empty_cache()

    # -- base pass ---------------------------------------------------------------
    print("[2/3] Running base model")
    hf_base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    base = run_model("base", hf_base, tokenizer, model, tokens, layer_list)
    del hf_base
    gc.collect()
    torch.cuda.empty_cache()

    print("[3/3] Writing artifacts")
    # The token the tuned model wants to emit first (should open the greeting).
    target_id = int(tuned["final_logits"].argmax())
    target_str = tokenizer.decode(target_id)

    # p(target) by layer for both models — "when does the model decide?"
    lens = {
        name: [
            float(torch.softmax(layer_logits, dim=-1)[target_id])
            for layer_logits in result["per_layer_logits"]
        ]
        for name, result in (("base", base), ("tuned", tuned))
    }
    (INTERP_DIR / "logit_lens.json").write_text(
        json.dumps(
            {"target_token": target_str, "target_id": target_id, "p_by_layer": lens}, indent=2
        )
    )

    # Top tokens by final-logit difference — what the fine-tune promoted/demoted.
    diff = tuned["final_logits"] - base["final_logits"]
    promoted = torch.topk(diff, top_k)
    demoted = torch.topk(-diff, top_k)
    (INTERP_DIR / "logit_diff.json").write_text(
        json.dumps(
            {
                "promoted": [
                    {"token": tokenizer.decode(i), "delta": round(float(d), 3)}
                    for d, i in zip(promoted.values, promoted.indices, strict=True)
                ],
                "demoted": [
                    {"token": tokenizer.decode(i), "delta": round(-float(d), 3)}
                    for d, i in zip(demoted.values, demoted.indices, strict=True)
                ],
            },
            indent=2,
        )
    )

    # Attention patterns (tuned model) in circuitsvis layout, plus the tokens.
    (INTERP_DIR / "attention.json").write_text(
        json.dumps(
            {
                "tokens": token_strs,
                "layers": {k: v for k, v in tuned["patterns"].items()},
                "base_layers": {k: v for k, v in base["patterns"].items()},
            }
        )
    )

    (INTERP_DIR / "meta.json").write_text(
        json.dumps(
            {
                "model": model,
                "adapter": str(adapter),
                "enquiry": INTERP_ENQUIRY,
                "n_prompt_tokens": len(token_strs),
                "attention_layers": layer_list,
                "first_answer_token": target_str,
            },
            indent=2,
        )
    )
    for f in sorted(INTERP_DIR.glob("*.json")):
        print(f"  {f} ({f.stat().st_size // 1024} KB)")
    print(f'Tuned model\'s first answer token: "{target_str}"')


if __name__ == "__main__":
    typer.run(main)
