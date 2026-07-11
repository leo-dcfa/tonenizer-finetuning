"""LoRA fine-tune of a chat model into the Noosa Council customer-service voice.

Trains a PEFT LoRA adapter on ChatML-formatted conversations with TRL's
SFTTrainer, computing loss on assistant tokens only. Saves the adapter and a
per-step loss history that the presentation notebook renders offline.

Example usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 4 --rank 32 --alpha 64
    uv run python scripts/train.py --data data/train.jsonl --out adapters/council-voice
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Qwen2 decoder blocks: attention projections + gated MLP. Targeting all of
# them (rather than just q/v) is what lets a small adapter shift the voice.
TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # attention
    "gate_proj",
    "up_proj",
    "down_proj",  # MLP
]

LOSS_JSON = Path("cache/loss.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--out", default="adapters/council-voice")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_chatml_dataset(path: str) -> Dataset:
    """Load a JSONL file of {"messages": [{role, content}, ...]} conversations."""
    dataset = load_dataset("json", data_files=path, split="train")
    if "messages" not in dataset.column_names:
        raise ValueError(f"{path} has no 'messages' column — expected ChatML JSONL.")
    return dataset


def save_loss_history(trainer: SFTTrainer, args: argparse.Namespace, wall_seconds: float) -> None:
    """Extract per-step training loss from the trainer log and write cache/loss.json."""
    steps = [entry["step"] for entry in trainer.state.log_history if "loss" in entry]
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    trainable_params, total_params = trainer.model.get_nb_trainable_parameters()

    LOSS_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOSS_JSON.write_text(
        json.dumps(
            {
                "steps": steps,
                "loss": losses,
                "meta": {
                    "model": args.model,
                    "epochs": args.epochs,
                    "learning_rate": args.lr,
                    "lora_rank": args.rank,
                    "lora_alpha": args.alpha,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "max_len": args.max_len,
                    "seed": args.seed,
                    "wall_seconds": round(wall_seconds, 1),
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                },
            },
            indent=2,
        )
    )
    print(f"Loss history ({len(steps)} steps) -> {LOSS_JSON}")


def main() -> None:
    args = parse_args()

    print(f"[1/4] Loading dataset from {args.data}")
    dataset = load_chatml_dataset(args.data)
    print(f"      {len(dataset)} conversations")

    print(f"[2/4] Loading base model {args.model} (bf16)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        # Train on assistant tokens only. TRL 1.8 handles Qwen's chat template
        # by swapping in one with {% generation %} markers automatically.
        assistant_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float = ratio of total steps (warmup_ratio is deprecated)
        logging_steps=1,  # per-step loss for the notebook's loss curve
        save_strategy="no",  # single save at the end, below
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainable_params, total_params = trainer.model.get_nb_trainable_parameters()
    print(
        f"      Trainable params: {trainable_params:,} of {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    print(
        f"[3/4] Training: {args.epochs} epochs, lr={args.lr}, "
        f"effective batch={args.batch_size * args.grad_accum}"
    )
    start = time.perf_counter()
    trainer.train()
    wall_seconds = time.perf_counter() - start

    print(f"[4/4] Saving adapter to {args.out}")
    trainer.save_model(args.out)
    save_loss_history(trainer, args, wall_seconds)

    print(f"Done in {wall_seconds / 60:.1f} min")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB")


if __name__ == "__main__":
    main()
