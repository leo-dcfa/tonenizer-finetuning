"""Map the loss valley the fine-tune walked into.

Two slices through the loss landscape, using the LoRA adapter as the axis that
training actually moved along:

  1D — scale the trained adapter from 0 (base model) past 1 (trained) to 1.5
       (overshoot) and record council-data loss at each point.
  2D — same axis, crossed with a random low-rank direction of matching norm
       (Li et al.-style slice), giving a loss surface with the base model at
       (0, 0) and the fine-tuned model at (1, 0).

Writes cache/landscape.json for the deck. Needs the GPU (vLLM down).

Example usage:
    uv run tokenizer landscape
    uv run tokenizer landscape --n-examples 16 --grid 9
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import typer
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_JSON = Path("cache/landscape.json")


def load_eval_batch(
    tokenizer: AutoTokenizer, data_path: Path, n_examples: int, max_len: int
) -> torch.Tensor:
    """Tokenize a fixed sample of training conversations (full-sequence LM loss)."""
    records = [json.loads(line) for line in data_path.open()][:n_examples]
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in records]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc["input_ids"]


def batch_loss(model: PeftModel, input_ids: torch.Tensor, batch_size: int = 8) -> float:
    """Mean causal-LM loss over the fixed batch."""
    losses = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            ids = input_ids[i : i + batch_size].to("cuda")
            out = model(input_ids=ids, labels=ids)
            losses.append(float(out.loss))
    return sum(losses) / len(losses)


def lora_modules(model: PeftModel):
    """Every LoRA-wrapped module."""
    for module in model.modules():
        if hasattr(module, "lora_A") and len(module.lora_A) > 0:
            yield module


def add_random_direction(model: PeftModel, seed: int) -> None:
    """Attach a second adapter 'noise' whose per-module B·A norm matches the trained one."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for module in lora_modules(model):
        (name,) = list(module.lora_A.keys())
        a_w = module.lora_A[name].weight
        b_w = module.lora_B[name].weight
        with torch.no_grad():
            delta_norm = (b_w.float() @ a_w.float()).norm()
            ra = torch.randn(a_w.shape, generator=gen).to(a_w.device, a_w.dtype)
            rb = torch.randn(b_w.shape, generator=gen).to(b_w.device, b_w.dtype)
            noise_norm = (rb.float() @ ra.float()).norm()
            scale = (delta_norm / noise_norm).sqrt()
            module.noise_A = (ra * scale).contiguous()
            module.noise_B = (rb * scale).contiguous()


def set_point(model: PeftModel, a: float, b: float) -> None:
    """Move to landscape point a·Δ + b·R by rewriting each module's adapter weights."""
    for module in lora_modules(model):
        (name,) = list(module.lora_A.keys())
        if not hasattr(module, "orig_A"):
            module.orig_A = module.lora_A[name].weight.data.clone()
            module.orig_B = module.lora_B[name].weight.data.clone()
        # B·A composite scales multiplicatively, so put the whole factor on B:
        # a·(B·A) + b·(Bn·An) can't be expressed exactly in one low-rank pair,
        # so evaluate the two contributions via a rank-2r stack instead.
        with torch.no_grad():
            module.lora_A[name].weight.data = torch.cat([module.orig_A, module.noise_A], dim=0)
            module.lora_B[name].weight.data = torch.cat(
                [module.orig_B * a, module.noise_B * b], dim=1
            )


def main(
    model_name: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", "--model"),
    adapter: Path = typer.Option(Path("adapters/council-voice")),
    data: Path = typer.Option(Path("data/train.jsonl")),
    n_examples: int = typer.Option(24),
    max_len: int = typer.Option(512),
    grid: int = typer.Option(13, help="2D grid resolution per axis"),
    n_1d: int = typer.Option(16, help="points along the 1D adapter-scale sweep"),
    b_max: float = typer.Option(0.6, help="half-range of the random direction in the 2D slice"),
    seed: int = typer.Option(0),
) -> None:
    """Compute 1D and 2D loss-landscape slices around the fine-tune."""
    print(f"[1/4] Loading {model_name} + adapter (bf16)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    input_ids = load_eval_batch(tokenizer, data, n_examples, max_len)
    print(f"      {len(input_ids)} examples, seq len {input_ids.shape[1]}")

    print("[2/4] Attaching norm-matched random direction")
    add_random_direction(model, seed)

    print(f"[3/4] 1D sweep: adapter scale 0 → 1.5 ({n_1d} points)")
    alphas = [round(i * 1.5 / (n_1d - 1), 4) for i in range(n_1d)]
    loss_1d = []
    for a in tqdm(alphas, desc="1D"):
        set_point(model, a, 0.0)
        loss_1d.append(round(batch_loss(model, input_ids), 4))

    print(f"[4/4] 2D slice: {grid}×{grid} grid")
    a_axis = [round(-0.25 + i * 1.75 / (grid - 1), 4) for i in range(grid)]
    b_axis = [round(-b_max + i * 2 * b_max / (grid - 1), 4) for i in range(grid)]
    loss_2d = []
    for bv in tqdm(b_axis, desc="2D"):
        row = []
        for av in a_axis:
            set_point(model, av, bv)
            row.append(round(batch_loss(model, input_ids), 4))
        loss_2d.append(row)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "alphas": alphas,
                "loss_1d": loss_1d,
                "a_axis": a_axis,
                "b_axis": b_axis,
                "loss_2d": loss_2d,
                "meta": {
                    "n_examples": n_examples,
                    "max_len": max_len,
                    "seed": seed,
                    "note": "a: trained-adapter scale (0=base, 1=fine-tuned); b: random low-rank direction, norm-matched",
                },
            },
            indent=2,
        )
    )
    base_l, tuned_l = loss_1d[0], loss_1d[min(range(len(alphas)), key=lambda i: abs(alphas[i] - 1))]
    print(f"Done → {OUT_JSON} · loss at base {base_l:.3f}, at tuned {tuned_l:.3f}")
    if math.isnan(base_l):
        raise RuntimeError("NaN loss — check the eval batch")


if __name__ == "__main__":
    typer.run(main)
