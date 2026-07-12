"""`tokenizer` — one CLI for the whole fine-tuning pipeline.

Usage:
    uv run tokenizer generate --model qwen3.6-27b --n 1500
    uv run tokenizer filter
    uv run tokenizer train
    uv run tokenizer evaluate --judge-base-url http://localhost:8000/v1 --judge-model qwen3.6-27b
    uv run tokenizer build-theme
"""

import typer

from scripts import build_theme, evaluate, filter_data, generate_data, train

app = typer.Typer(
    name="tokenizer",
    help="Noosa Council fine-tuning workshop pipeline.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("generate")(generate_data.main)
app.command("filter")(filter_data.main)
app.command("train")(train.main)
app.command("evaluate")(evaluate.main)
app.command("build-theme")(build_theme.main)


if __name__ == "__main__":
    app()
