"""Filter raw synthetic data into a clean training set plus a held-out eval set.

Each record must pass a chain of small, independent checks (ChatML shape, length
bounds, style compliance, refusal rejection, topic sanity, exact and near-duplicate
removal). Survivors are split into a stratified held-out eval set of prompts and
the training set. A kept/dropped report table is printed at the end — that table
is itself a slide in the talk.

Examples:
    uv run python scripts/filter_data.py
    uv run python scripts/filter_data.py --in data/raw.jsonl --n-eval 10
    uv run python scripts/filter_data.py --self-test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import CONTACT_LINE, GREETING, SIGN_OFF  # noqa: E402

REFUSAL_PATTERNS = (
    "as an ai",
    "as a language model",
    "language model",
    "i cannot",
    "i can't help",
    "i am not able to",
    "i'm sorry, but",
)

# At least one keyword must appear in the user enquiry for its labelled topic.
# "general" is the catch-all: anything passes.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "rates & payments": ("rate", "payment", "pay", "bill", "notice", "pension", "concession"),
    "waste & bins": ("bin", "waste", "rubbish", "recycl", "collect", "garbage", "tip"),
    "permits & applications": ("permit", "application", "apply", "approval", "licence", "license"),
    "noise complaints": ("noise", "loud", "bark", "music", "construction", "party"),
    "parking": ("park", "fine", "infringement", "ticket", "driveway"),
    "pets & animals": ("dog", "cat", "pet", "animal", "puppy", "leash", "snake", "wildlife"),
    "beaches & trails": ("beach", "trail", "track", "walk", "coast", "erosion", "river"),
    "events": ("event", "festival", "market", "book", "stall", "hall", "venue"),
    "planning & development": (
        "plan",
        "development",
        "zoning",
        "build",
        "shed",
        "approval",
        "object",
    ),
    "general": (),
}


# --- Checks: each one small, pure, and testable ----------------------------------


def parse_record(line: str) -> dict | None:
    """Parse one JSONL line; None if it is not a JSON object."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def has_valid_shape(record: dict) -> bool:
    """Exactly one system/user/assistant turn, all non-empty strings, plus a topic label."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return False
    for message, role in zip(messages, ("system", "user", "assistant"), strict=True):
        if not isinstance(message, dict) or message.get("role") != role:
            return False
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            return False
    return isinstance(record.get("topic"), str)


def within_length(text: str, min_chars: int, max_chars: int) -> bool:
    return min_chars <= len(text) <= max_chars


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the near-duplicate key."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(text.split())


def is_style_compliant(assistant: str) -> bool:
    """The reply must carry the council-voice fingerprint: greeting and sign-off block."""
    return assistant.startswith(GREETING) and SIGN_OFF in assistant and CONTACT_LINE in assistant


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def matches_topic(user: str, topic: str) -> bool:
    keywords = TOPIC_KEYWORDS.get(topic)
    if keywords is None:  # unknown topic label
        return False
    if not keywords:  # "general" accepts anything
        return True
    lowered = user.lower()
    return any(keyword in lowered for keyword in keywords)


# --- Pipeline --------------------------------------------------------------------


def filter_lines(
    lines: list[str], min_chars: int, max_chars: int
) -> tuple[list[dict], Counter[str]]:
    """Run every line through the check chain; return (kept records, drop counts)."""
    kept: list[dict] = []
    counts: Counter[str] = Counter()
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()

    for line in lines:
        if not line.strip():
            continue
        counts["total"] += 1

        record = parse_record(line)
        if record is None:
            counts["invalid json"] += 1
            continue
        if not has_valid_shape(record):
            counts["bad chat shape"] += 1
            continue

        user = record["messages"][1]["content"]
        assistant = record["messages"][2]["content"]

        if not (
            within_length(user, min_chars, max_chars)
            and within_length(assistant, min_chars, max_chars)
        ):
            counts["length out of bounds"] += 1
            continue
        if not is_style_compliant(assistant):
            counts["style non-compliant"] += 1
            continue
        if is_refusal(user) or is_refusal(assistant):
            counts["refusal / meta-text"] += 1
            continue
        if not matches_topic(user, record["topic"]):
            counts["topic mismatch"] += 1
            continue
        if user in seen_exact:
            counts["exact duplicate"] += 1
            continue
        normalized = normalize(user)
        if normalized in seen_normalized:
            counts["near duplicate"] += 1
            continue

        seen_exact.add(user)
        seen_normalized.add(normalized)
        kept.append(record)

    counts["kept"] = len(kept)
    return kept, counts


def stratified_split(kept: list[dict], n_eval: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Hold out n_eval records spread evenly across topics; return (train, eval)."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for record in kept:
        by_topic[record["topic"]].append(record)

    rng = random.Random(seed)
    for records in by_topic.values():
        rng.shuffle(records)

    held_out: list[dict] = []
    topics = sorted(by_topic)
    while len(held_out) < min(n_eval, len(kept)):
        for topic in topics:
            if by_topic[topic] and len(held_out) < n_eval:
                held_out.append(by_topic[topic].pop())

    held_ids = {id(record) for record in held_out}
    train = [record for record in kept if id(record) not in held_ids]
    return train, held_out


def print_report(counts: Counter[str]) -> None:
    """The kept/dropped table — shown as a slide in the talk."""
    total = counts["total"]
    kept = counts["kept"]
    print(f"\n{'reason':<24}{'count':>8}{'share':>9}")
    print("-" * 41)
    for reason, count in counts.most_common():
        if reason in ("total", "kept"):
            continue
        print(f"{reason:<24}{count:>8}{count / total:>8.1%}")
    print("-" * 41)
    print(f"{'kept':<24}{kept:>8}{kept / total:>8.1%}")
    print(f"{'total':<24}{total:>8}")


def run(args: argparse.Namespace) -> None:
    lines = Path(args.in_path).read_text(encoding="utf-8").splitlines()
    kept, counts = filter_lines(lines, args.min_chars, args.max_chars)
    train, held_out = stratified_split(kept, args.n_eval)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in train:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    eval_prompts = [
        {"prompt": record["messages"][1]["content"], "topic": record["topic"]}
        for record in held_out
    ]
    Path(args.eval_out).write_text(
        json.dumps(eval_prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_report(counts)
    print(f"\n{len(train)} train examples -> {args.out}")
    print(f"{len(eval_prompts)} eval prompts  -> {args.eval_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter raw synthetic data into train + eval sets."
    )
    parser.add_argument("--in", dest="in_path", default="data/raw.jsonl")
    parser.add_argument("--out", default="data/train.jsonl")
    parser.add_argument("--eval-out", default="data/eval_prompts.json")
    parser.add_argument(
        "--n-eval", type=int, default=10, help="held-out eval prompts, stratified across topics"
    )
    parser.add_argument(
        "--min-chars", type=int, default=20, help="minimum length for user and assistant turns"
    )
    parser.add_argument(
        "--max-chars", type=int, default=1500, help="maximum length for user and assistant turns"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run the built-in check-by-check test and exit"
    )
    return parser.parse_args()


# --- Self-test -------------------------------------------------------------------


def _record(user: str, assistant: str, topic: str) -> str:
    return json.dumps(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are Noosa Council's customer service assistant.",
                },
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "topic": topic,
            "register": "polite",
        }
    )


def self_test() -> None:
    """One synthetic record per drop reason, plus good ones; assert the counts."""
    good_answer = (
        f"{GREETING} We can sort out your bin.\n\n"
        f"1. Check it was out by 5.30am.\n2. Report it on our website.\n\n"
        f"{SIGN_OFF}\n{CONTACT_LINE}"
    )
    good_user = "My red bin wasn't collected on Tuesday, can someone come back for it?"
    lines = [
        _record(good_user, good_answer, "waste & bins"),  # kept
        "{not json at all",  # invalid json
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}),  # bad chat shape
        _record("Bin?", good_answer, "waste & bins"),  # length (user too short)
        _record(
            "My green waste bin is cracked and needs replacing please.",
            "Sure, we will replace your bin next week, no worries at all!",
            "waste & bins",
        ),  # style non-compliant
        _record(
            "My recycling bin is full, as an AI can you help me?", good_answer, "waste & bins"
        ),  # refusal / meta-text
        _record(
            "What are the library opening hours this weekend please?", good_answer, "waste & bins"
        ),  # topic mismatch
        _record(good_user, good_answer, "waste & bins"),  # exact duplicate
        _record(good_user.upper() + "!!", good_answer, "waste & bins"),  # near duplicate
        _record(
            "How do I set up a payment plan for my overdue rates notice?",
            good_answer,
            "rates & payments",
        ),  # kept
    ]

    kept, counts = filter_lines(lines, min_chars=20, max_chars=1500)
    expected = Counter(
        {
            "total": 10,
            "kept": 2,
            "invalid json": 1,
            "bad chat shape": 1,
            "length out of bounds": 1,
            "style non-compliant": 1,
            "refusal / meta-text": 1,
            "topic mismatch": 1,
            "exact duplicate": 1,
            "near duplicate": 1,
        }
    )
    assert counts == expected, f"unexpected counts: {counts}"

    train, held_out = stratified_split(kept, n_eval=1)
    assert len(train) == 1 and len(held_out) == 1, "split sizes wrong"
    assert normalize("Hello, World!!  ") == "hello world", "normalize broken"

    print_report(counts)
    print("\nself-test passed: 10 records, 2 kept, one drop per reason.")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    else:
        run(arguments)
