"""Generate synthetic Noosa Council training data via an OpenAI-compatible endpoint.

Two-stage pipeline: (1) generate a diverse resident enquiry for a (topic, register)
pair, then (2) answer it in the council voice defined by STYLE_CARD. Each example is
appended to a JSONL file in ChatML format, so the script is resumable: re-running it
only generates whatever is missing to reach --n.

Examples:
    uv run tokenizer generate --model qwen3.6-27b --n 1500
    uv run tokenizer generate --model deepseek-v3 \\
        --base-url http://dgx:8000/v1 --concurrency 16
    uv run tokenizer generate --model x --dry-run   # show stage prompts
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import typer
from openai import AsyncOpenAI

# --- The council voice ----------------------------------------------------------
# These three constants define the visible "fingerprint" of the fine-tune.
# filter_data.py imports them to enforce style compliance, and the notebook
# imports STYLE_CARD to show on a slide.

GREETING = "Thanks for reaching out to Noosa Council."
SIGN_OFF = "— Noosa Council Customer Service"
CONTACT_LINE = "Phone (07) 5329 6500 · noosa.qld.gov.au"

STYLE_CARD = f"""You write replies for Noosa Council's customer service team.
Follow this style card exactly:

- Open with exactly this sentence: "{GREETING}"
- Plain, empathetic English at a year-7 reading level. No jargon.
- Short paragraphs of 1-3 sentences each.
- If the resident needs to take steps, list them as a numbered list.
- Never invent specific fees, dates, or opening hours. Instead, say where to
  find them (the Noosa Council website or the customer service phone line).
- Keep the whole reply under 180 words.
- End with exactly this sign-off block on its own two lines:
{SIGN_OFF}
{CONTACT_LINE}

Write only the reply itself: no subject line, no commentary, no markdown headings."""

# The persona the fine-tuned model will run with at inference time. It stays short
# on purpose: the *behaviour* (greeting, structure, sign-off) is what training bakes in.
SYSTEM_PROMPT = (
    "You are Noosa Council's customer service assistant. Reply to resident "
    "enquiries in the council's warm, plain-English voice."
)

# --- Topic and register seeds ---------------------------------------------------

TOPICS: dict[str, list[str]] = {
    "rates & payments": [
        "a rates notice that looks too high",
        "setting up a payment plan for overdue rates",
        "whether a pensioner concession applies",
        "changing the postal address on a rates notice",
    ],
    "waste & bins": [
        "a bin that was not collected this week",
        "a damaged or stolen wheelie bin",
        "how to book a kerbside large-item collection",
        "what goes in the yellow recycling bin",
    ],
    "permits & applications": [
        "a permit for a garage sale sign",
        "a permit to run a food stall",
        "removing a tree on private property",
        "a temporary road closure for a street party",
    ],
    "noise complaints": [
        "a neighbour's dog barking all night",
        "construction noise starting before 6.30am",
        "loud music from a short-stay rental",
        "a leaf blower used every morning",
    ],
    "parking": [
        "a parking fine the resident thinks is unfair",
        "where to park near Hastings Street in peak season",
        "a resident parking permit",
        "cars parked across a driveway",
    ],
    "pets & animals": [
        "registering a new puppy",
        "an off-leash dog area",
        "a snake or wildlife concern in the yard",
        "a lost cat and what the pound does",
    ],
    "beaches & trails": [
        "whether a coastal trail section is open",
        "dogs on the beach rules",
        "beach wheelchair access",
        "erosion or damage on a walking track",
    ],
    "events": [
        "booking a park or hall for a community event",
        "what road closures an upcoming festival causes",
        "running a market stall at a council event",
        "noise and hours rules for a private event",
    ],
    "planning & development": [
        "the status of a development application next door",
        "whether a shed needs building approval",
        "zoning rules for a granny flat",
        "objecting to a proposed development",
    ],
    "general": [
        "how to contact the right council department",
        "reporting a pothole or broken street light",
        "opening hours of council facilities",
        "giving feedback about a council service",
    ],
}

REGISTERS: dict[str, str] = {
    "angry": "frustrated and blunt; they feel let down and want it fixed now",
    "confused": "unsure what they even need; asks vague or muddled questions",
    "brief": "one or two short sentences, almost telegraphic",
    "rambling": "long-winded, includes irrelevant back-story before the point",
    "polite": "friendly and courteous, says please and thank you",
    "elderly resident": "an older long-time local; chatty, mentions how things used to be",
    "new resident": "just moved to Noosa; doesn't know how anything works here",
}

# --- Stage prompts ---------------------------------------------------------------

ENQUIRY_SYSTEM = (
    "You write realistic messages that residents send to their local council "
    "through email or a website contact form. You write only the message body, "
    "nothing else."
)

ENQUIRY_TEMPLATE = """Write one message from a resident to Noosa Council (Queensland, Australia).

Topic: {topic} — specifically something like: {hint}
Voice: {register} — {register_desc}

Rules:
- 1 to 6 sentences, like a real email or web-form message.
- No subject line, no signature, no placeholder names in [brackets].
{typo_rule}
Write only the message text."""

TYPO_RULE = "- Include one or two small, natural typos."
NO_TYPO_RULE = "- Normal spelling is fine."
TYPO_PROBABILITY = 0.3

# Answers use a lower temperature than enquiries: we want *diverse* questions
# but a *consistent* council voice.
ANSWER_TEMPERATURE = 0.7

MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class Task:
    """One planned training example: which topic/register to generate."""

    topic: str
    hint: str
    register: str
    with_typos: bool


def build_enquiry_prompt(task: Task) -> str:
    return ENQUIRY_TEMPLATE.format(
        topic=task.topic,
        hint=task.hint,
        register=task.register,
        register_desc=REGISTERS[task.register],
        typo_rule=TYPO_RULE if task.with_typos else NO_TYPO_RULE,
    )


def plan_tasks(n: int, rng: random.Random) -> list[Task]:
    """Cycle through topics so coverage stays even; vary register and hint randomly."""
    tasks = []
    for _, topic in zip(range(n), cycle(TOPICS)):
        tasks.append(
            Task(
                topic=topic,
                hint=rng.choice(TOPICS[topic]),
                register=rng.choice(list(REGISTERS)),
                with_typos=rng.random() < TYPO_PROBABILITY,
            )
        )
    rng.shuffle(tasks)
    return tasks


# Reasoning models (Qwen3.x, DeepSeek) burn the token budget on a hidden
# "thinking" pass and return empty content at small max_tokens. This template
# kwarg turns thinking off; endpoints whose templates lack the variable ignore it.
NO_THINK_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


async def chat(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    thinking: bool = False,
) -> str | None:
    """One chat completion with retry/backoff. Returns None after MAX_ATTEMPTS failures."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=None if thinking else NO_THINK_BODY,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip().strip('"')
        except Exception as error:  # endpoint hiccups: timeouts, 5xx, resets
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {error}")
        await asyncio.sleep(2**attempt)
    return None


async def generate_example(
    client: AsyncOpenAI,
    model: str,
    task: Task,
    temperature: float,
    semaphore: asyncio.Semaphore,
    thinking: bool = False,
) -> dict | None:
    """Run both stages for one task. Returns a ChatML record, or None if a stage failed."""
    async with semaphore:
        enquiry = await chat(
            client,
            model,
            ENQUIRY_SYSTEM,
            build_enquiry_prompt(task),
            temperature=temperature,
            max_tokens=2000 if thinking else 250,
            thinking=thinking,
        )
        if enquiry is None:
            return None
        answer = await chat(
            client,
            model,
            STYLE_CARD,
            enquiry,
            temperature=ANSWER_TEMPERATURE,
            max_tokens=2500 if thinking else 400,
            thinking=thinking,
        )
        if answer is None:
            return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": enquiry},
            {"role": "assistant", "content": answer},
        ],
        "topic": task.topic,
        "register": task.register,
    }


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


async def run(
    model: str,
    base_url: str,
    n: int,
    out: Path,
    batch_size: int,
    temperature: float,
    concurrency: int,
    api_key: str,
    thinking: bool,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = count_existing(out)
    remaining = n - existing
    if remaining <= 0:
        print(f"{out} already has {existing} examples (target {n}); nothing to do.")
        return
    print(f"{existing} existing examples in {out}; generating {remaining} more.")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = plan_tasks(remaining, random.Random())
    n_batches = math.ceil(len(tasks) / batch_size)

    written = skipped = 0
    with out.open("a", encoding="utf-8") as f:
        for b in range(n_batches):
            batch = tasks[b * batch_size : (b + 1) * batch_size]
            results = await asyncio.gather(
                *(
                    generate_example(client, model, task, temperature, semaphore, thinking)
                    for task in batch
                )
            )
            for record in results:
                if record is None:
                    skipped += 1
                else:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
            f.flush()
            print(f"batch {b + 1}/{n_batches}: {existing + written}/{n} total, {skipped} skipped")

    print(f"Done: wrote {written} examples ({skipped} skipped after retries).")
    if skipped:
        print("Re-run the same command to top up to the target count.")


def print_stage_prompts() -> None:
    """Print the two stage prompts for one example task, then exit."""
    task = Task(
        topic="waste & bins",
        hint=TOPICS["waste & bins"][0],
        register="angry",
        with_typos=True,
    )
    print("=== STAGE 1: ENQUIRY — system prompt ===\n")
    print(ENQUIRY_SYSTEM)
    print("\n=== STAGE 1: ENQUIRY — user prompt ===\n")
    print(build_enquiry_prompt(task))
    print("\n=== STAGE 2: ANSWER — system prompt (STYLE_CARD) ===\n")
    print(STYLE_CARD)
    print("\n=== STAGE 2: ANSWER — user prompt ===\n")
    print("<the enquiry text produced by stage 1>")


def main(
    model: str = typer.Option(..., help="model name at the endpoint"),
    base_url: str = typer.Option(
        "http://localhost:8000/v1", help="OpenAI-compatible endpoint (vLLM, Ollama, ...)"
    ),
    n: int = typer.Option(1500, help="total examples wanted in --out (resumes if some exist)"),
    out: Path = typer.Option(Path("data/raw.jsonl")),
    batch_size: int = typer.Option(20, help="examples generated per progress batch"),
    temperature: float = typer.Option(0.9, help="sampling temperature for the enquiry stage"),
    concurrency: int = typer.Option(8, help="max in-flight requests"),
    api_key: str = typer.Option("none", help="API key; local endpoints usually ignore it"),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="let reasoning models think (slower; raises max_tokens to compensate)",
    ),
    dry_run: bool = typer.Option(False, help="print the two stage prompts for one topic and exit"),
) -> None:
    """Generate synthetic Noosa Council enquiry/answer pairs."""
    if dry_run:
        print_stage_prompts()
        return
    asyncio.run(
        run(model, base_url, n, out, batch_size, temperature, concurrency, api_key, thinking)
    )


if __name__ == "__main__":
    typer.run(main)
