"""Compare base-model vs LoRA-tuned responses on the held-out eval prompts.

Loads the base model once, generates every response with the adapter disabled
(base) and enabled (tuned) using identical sampling settings and a fixed seed
per prompt, then writes side-by-side pairs to cache/comparisons.json for the
presentation notebook. Optionally scores each pair with an LLM judge via an
OpenAI-compatible endpoint.

Example usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --adapter adapters/council-voice --max-new-tokens 400
    uv run python scripts/evaluate.py --judge-base-url http://localhost:8000/v1 --judge-model qwen3
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Same system prompt the training data was generated with — responses are only
# comparable if both models are asked the same way the adapter was trained.
try:
    from generate_data import SYSTEM_PROMPT
except ImportError:  # fallback if generate_data.py is absent
    SYSTEM_PROMPT = (
        "You are Noosa Council's customer service assistant. Reply to resident "
        "enquiries in the council's warm, plain-English voice."
    )
    print("NOTE: scripts/generate_data.py not found — using fallback SYSTEM_PROMPT.")

TEMPERATURE = 0.7
TOP_P = 0.9

JUDGE_INSTRUCTIONS = (
    "You are grading two customer-service replies to the same resident enquiry "
    "against Noosa Council's house style: a friendly greeting, clear structure "
    "(short paragraphs or steps), plain language free of jargon, and a helpful "
    "sign-off pointing to next steps. Score each reply from 1 (ignores the "
    "style) to 5 (exemplary). Respond with JSON only, exactly: "
    '{"base_score": <1-5>, "tuned_score": <1-5>, "rationale": "<one or two sentences>"}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="adapters/council-voice")
    parser.add_argument("--prompts", default="data/eval_prompts.json")
    parser.add_argument("--out", default="cache/comparisons.json")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument(
        "--judge-base-url", default=None, help="OpenAI-compatible endpoint; enables the judge pass"
    )
    parser.add_argument("--judge-model", default=None, help="Model name at the judge endpoint")
    return parser.parse_args()


def generate_response(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    seed: int,
) -> str:
    """Generate one reply with fixed sampling settings and a per-prompt seed."""
    torch.manual_seed(seed)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()


def generate_all(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompts: list[dict[str, str]],
    max_new_tokens: int,
    label: str,
) -> list[str]:
    """Run every eval prompt through the model, seeding identically per prompt."""
    return [
        generate_response(model, tokenizer, item["prompt"], max_new_tokens, seed=1000 + i)
        for i, item in enumerate(tqdm(prompts, desc=f"Generating ({label})"))
    ]


def judge_pair(client: Any, judge_model: str, item: dict[str, Any]) -> dict[str, Any]:
    """Score one base/tuned pair 1-5 on council-style adherence."""
    user_message = (
        f"Resident enquiry:\n{item['prompt']}\n\n"
        f"Reply A (base):\n{item['base_response']}\n\n"
        f"Reply B (tuned):\n{item['tuned_response']}"
    )
    completion = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate prose around the JSON
    parsed = json.loads(match.group(0)) if match else {}
    return {
        "base_score": parsed.get("base_score"),
        "tuned_score": parsed.get("tuned_score"),
        "rationale": parsed.get("rationale", raw.strip()),
    }


def main() -> None:
    args = parse_args()
    prompts: list[dict[str, str]] = json.loads(Path(args.prompts).read_text())
    print(f"[1/4] Loaded {len(prompts)} eval prompts from {args.prompts}")

    print(f"[2/4] Loading base model {args.model} (bf16)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Base pass first, then attach the adapter to the same weights — one model
    # in VRAM the whole time, and both passes share identical sampling settings.
    base_responses = generate_all(model, tokenizer, prompts, args.max_new_tokens, "base")

    print(f"[3/4] Attaching adapter from {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tuned_responses = generate_all(model, tokenizer, prompts, args.max_new_tokens, "tuned")

    comparisons: list[dict[str, Any]] = [
        {
            "prompt": item["prompt"],
            "topic": item["topic"],
            "base_response": base,
            "tuned_response": tuned,
        }
        for item, base, tuned in zip(prompts, base_responses, tuned_responses, strict=True)
    ]

    if args.judge_base_url and args.judge_model:
        print(f"[4/4] Judge pass via {args.judge_base_url} ({args.judge_model})")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        from openai import OpenAI

        client = OpenAI(base_url=args.judge_base_url, api_key="not-needed")
        for item in tqdm(comparisons, desc="Judging"):
            item["judge"] = judge_pair(client, args.judge_model, item)
        deltas = [
            c["judge"]["tuned_score"] - c["judge"]["base_score"]
            for c in comparisons
            if c["judge"]["base_score"] is not None and c["judge"]["tuned_score"] is not None
        ]
        if deltas:
            print(f"      Mean judge delta (tuned - base): {sum(deltas) / len(deltas):+.2f}")
    else:
        print("[4/4] Judge pass skipped (set --judge-base-url and --judge-model to enable)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparisons, indent=2))
    print(f"Wrote {len(comparisons)} comparisons -> {out_path}")


if __name__ == "__main__":
    main()
