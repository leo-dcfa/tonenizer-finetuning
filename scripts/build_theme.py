"""Build assets/theme.css from theme-src.css + base64-embedded fonts and logo.

Marimo injects css_file content into the page, so relative url() paths don't
resolve — fonts and the footer logo are embedded as data URIs instead, keeping
the deck fully offline-safe.

Usage: uv run python scripts/build_theme.py
"""

import base64
import re
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"

FONTS = [
    ("Fraunces", "normal", 600, "Fraunces-600.woff2"),
    ("Fraunces", "italic", 400, "Fraunces-400-italic.woff2"),
    ("DM Sans", "normal", 400, "DMSans-400.woff2"),
    ("DM Sans", "normal", 500, "DMSans-500.woff2"),
    ("DM Sans", "normal", 700, "DMSans-700.woff2"),
    ("JetBrains Mono", "normal", 400, "JetBrainsMono-400.woff2"),
    ("JetBrains Mono", "normal", 700, "JetBrainsMono-700.woff2"),
]


def font_face(family: str, style: str, weight: int, filename: str) -> str:
    data = base64.b64encode((ASSETS / "fonts" / filename).read_bytes()).decode()
    return (
        f"@font-face {{ font-family: '{family}'; font-style: {style}; "
        f"font-weight: {weight}; font-display: swap; "
        f"src: url(data:font/woff2;base64,{data}) format('woff2'); }}"
    )


def logo_data_uri() -> str:
    svg = (ASSETS / "azul-logo.svg").read_text()
    svg = re.sub(r"\s+", " ", svg).strip()
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def main() -> None:
    """Rebuild assets/theme.css from theme-src.css + embedded fonts and logo."""
    src = (ASSETS / "theme-src.css").read_text()
    parts = [
        src,
        "\n/* ── generated: embedded fonts + logo (scripts/build_theme.py) ── */",
        *[font_face(*f) for f in FONTS],
    ]
    (ASSETS / "theme.css").write_text("\n".join(parts) + "\n")
    (ASSETS / "logo_uri.txt").write_text(logo_data_uri())
    print(f"wrote {ASSETS / 'theme.css'} ({(ASSETS / 'theme.css').stat().st_size // 1024} KB)")
    print(f"wrote {ASSETS / 'logo_uri.txt'}")


if __name__ == "__main__":
    main()
