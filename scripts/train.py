"""LoRA fine-tune of a chat model into the Noosa Council customer-service voice.

Trains a PEFT LoRA adapter on ChatML-formatted conversations with TRL's
SFTTrainer, computing loss on assistant tokens only. Saves the adapter and a
per-step loss history that the presentation notebook renders offline.

Example usage:
    uv run tokenizer train
    uv run tokenizer train --epochs 4 --rank 32 --alpha 64
    uv run tokenizer train --data data/train.jsonl --out adapters/council-voice
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import typer
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


def load_chatml_dataset(path: str) -> Dataset:
    """Load a JSONL file of {"messages": [{role, content}, ...]} conversations."""
    dataset = load_dataset("json", data_files=path, split="train")
    if "messages" not in dataset.column_names:
        raise ValueError(f"{path} has no 'messages' column — expected ChatML JSONL.")
    return dataset


def save_loss_history(
    trainer: SFTTrainer, hyperparams: dict[str, object], wall_seconds: float
) -> None:
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
                    **hyperparams,
                    "wall_seconds": round(wall_seconds, 1),
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                },
            },
            indent=2,
        )
    )
    print(f"Loss history ({len(steps)} steps) -> {LOSS_JSON}")


def main(
    model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct"),
    data: Path = typer.Option(Path("data/train.jsonl")),
    out: Path = typer.Option(Path("adapters/council-voice")),
    epochs: int = typer.Option(3),
    lr: float = typer.Option(2e-4),
    rank: int = typer.Option(16),
    alpha: int = typer.Option(32),
    batch_size: int = typer.Option(4),
    grad_accum: int = typer.Option(4),
    max_len: int = typer.Option(1024),
    seed: int = typer.Option(42),
) -> None:
    """LoRA fine-tune of a chat model into the Noosa Council customer-service voice."""
    print(f"[1/4] Loading dataset from {data}")
    dataset = load_chatml_dataset(str(data))
    print(f"      {len(dataset)} conversations")

    print(f"[2/4] Loading base model {model} (bf16)")
    tokenizer = AutoTokenizer.from_pretrained(model)
    base_model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    sft_config = SFTConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_len,
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
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=base_model,
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

    print(f"[3/4] Training: {epochs} epochs, lr={lr}, effective batch={batch_size * grad_accum}")
    start = time.perf_counter()
    trainer.train()
    wall_seconds = time.perf_counter() - start

    print(f"[4/4] Saving adapter to {out}")
    trainer.save_model(str(out))
    save_loss_history(
        trainer,
        {
            "model": model,
            "epochs": epochs,
            "learning_rate": lr,
            "lora_rank": rank,
            "lora_alpha": alpha,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "max_len": max_len,
            "seed": seed,
        },
        wall_seconds,
    )

    print(f"Done in {wall_seconds / 60:.1f} min")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB")


if __name__ == "__main__":
    typer.run(main)
