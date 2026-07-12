import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="full",
    css_file="assets/theme.css",
    layout_file="layouts/notebook.slides.json",
)


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd
    from pygments import highlight as pygments_highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    ASSETS = Path(__file__).parent / "assets"
    CACHE = Path(__file__).parent / "cache"
    LOGO_URI = (ASSETS / "logo_uri.txt").read_text().strip()

    # ── Altair brand theme, registered at import time so every chart cell
    #    (all downstream of this cell) sees it (validated palette — STYLE.md) ──
    AZUL_CHART = ["#1E5AA8", "#B08432", "#3B82C4"]

    @alt.theme.register("azul", enable=True)
    def _azul_theme() -> alt.theme.ThemeConfig:
        return {
            "config": {
                "background": "#FAF8F5",
                "font": "DM Sans, system-ui, sans-serif",
                "mark": {"color": AZUL_CHART[0]},
                "range": {"category": AZUL_CHART},
                "axis": {
                    "labelColor": "#4A5568",
                    "titleColor": "#4A5568",
                    "labelFontSize": 14,
                    "titleFontSize": 14,
                    "grid": False,
                    "domainColor": "#4A5568",
                    "tickColor": "#4A5568",
                },
                "axisY": {"grid": True, "gridColor": "#4A5568", "gridOpacity": 0.12},
                "legend": {"labelColor": "#4A5568", "labelFontSize": 14},
                "view": {"stroke": None},
                "line": {"strokeWidth": 2, "color": AZUL_CHART[0]},
                "bar": {"cornerRadiusEnd": 4, "color": AZUL_CHART[0]},
                "point": {"filled": True, "size": 80, "color": AZUL_CHART[0]},
            }
        }

    return (
        AZUL_CHART,
        CACHE,
        HtmlFormatter,
        LOGO_URI,
        PythonLexer,
        alt,
        json,
        mo,
        pd,
        pygments_highlight,
    )


@app.cell
def _(HtmlFormatter, LOGO_URI, PythonLexer, mo, pygments_highlight):
    # ── slide helpers — all deck styling flows through these (see STYLE.md) ──

    def _footer(section: str) -> str:
        return f"""
        <div class="az-footer">
          <span>{section}</span>
          <span class="az-footer-brand"><img src="{LOGO_URI}" alt="Azul Labs"/> azl.au</span>
        </div>"""

    def slide(title: str, body_html: str, section: str = "", sub: str = ""):
        """Standard content slide: Fraunces title, gold rule, body, footer."""
        sub_html = f'<p class="az-sub">{sub}</p>' if sub else ""
        return mo.Html(f"""
        <div class="az-slide">
          <h1 class="az-title">{title}</h1>
          <hr class="az-rule"/>
          {sub_html}
          <div class="az-body">{body_html}</div>
          {_footer(section)}
        </div>""")

    def divider(num: int, title: str):
        """Inverted midnight section divider with ghost numeral."""
        return mo.Html(f"""
        <div class="az-slide az-slide--dark">
          <div style="margin: auto 0;">
            <h1 class="az-title" style="font-size: 4rem;">{title}</h1>
            <hr class="az-rule"/>
          </div>
          <div class="az-ghost">{num:02d}</div>
          {_footer("")}
        </div>""")

    def stat(value: str, label: str) -> str:
        """One stat tile (compose inside az-stats)."""
        return f"""
        <div>
          <div class="az-stat-rule"></div>
          <div class="az-stat-value">{value}</div>
          <div class="az-stat-label">{label}</div>
        </div>"""

    def compare(prompt: str, base: str, tuned: str, title: str, section: str):
        """Pre/post comparison: identical prompt, base vs fine-tuned cards."""
        return mo.Html(f"""
        <div class="az-slide">
          <h1 class="az-title">{title}</h1>
          <hr class="az-rule"/>
          <div class="az-prompt">{prompt}</div>
          <div class="az-compare">
            <div class="az-card az-card--base">
              <span class="az-chip az-chip--base">BASE</span>
              <div>{base}</div>
            </div>
            <div class="az-card az-card--tuned">
              <span class="az-chip az-chip--tuned">FINE-TUNED</span>
              <div>{tuned}</div>
            </div>
          </div>
          {_footer(section)}
        </div>""")

    def code_card(code: str, caption: str = "", highlight: set[int] | None = None) -> str:
        """Code as exhibit: midnight card, brand syntax colors, gold line highlight."""
        html = pygments_highlight(code.strip(), PythonLexer(), HtmlFormatter(nowrap=True))
        lines = html.rstrip("\n").split("\n")
        if highlight:
            lines = [
                f'<span class="hl">{ln}</span>' if i + 1 in highlight else ln
                for i, ln in enumerate(lines)
            ]
        cap = f'<div class="az-caption">{caption}</div>' if caption else ""
        return f'<div class="az-code-card"><pre>{"\n".join(lines)}</pre></div>{cap}'

    return code_card, compare, divider, slide, stat


@app.cell
def _(mo, slide, stat):
    # ── SLIDE 1 · About me ──
    slide(
        "Leo Alves",
        f"""
        <p class="az-sub">Azul Labs Pty Ltd · engineer &amp; indie researcher</p>
        <p>AI consulting, Claude API integration, and Python backends —
        on the Sunshine Coast.</p>
        <div class="az-stats">
          {stat("15+", "years engineering")}
          {stat("50+", "projects delivered")}
        </div>
        <p class="az-italic" style="font-size: 1.75rem; margin-top: 3rem;">
          Time to ride the <em class="az-gold">AI wave</em>.
        </p>
        """,
        section="Fine-tuning 1-1 · Noosa Council",
    )
    return


@app.cell
def _(divider):
    # ── SLIDE 2 · sample section divider (template check) ──
    divider(1, "Why fine-tune — and why not")
    return


@app.cell
def _(code_card, slide):
    # ── SLIDE 3 · sample code slide (template check) ──
    slide(
        "LoRA in one line of config",
        code_card(
            """
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 29,933,568 || all params: 3,115,872,256 || 0.96%
            """,
            caption="The adapter is ~50 MB. The base model is 6 GB.",
            highlight={5},
        ),
        section="02 · LoRA",
    )
    return


@app.cell
def _(alt, pd):
    # ── SLIDE 4 · sample chart slide (theme check) ──
    _demo = pd.DataFrame(
        {
            "step": list(range(0, 200, 10)),
            "loss": [2.1 - 0.08 * i + 0.002 * i * i for i in range(20)],
        }
    )
    _chart = (
        alt.Chart(_demo)
        .mark_line()
        .encode(x=alt.X("step", title="training step"), y=alt.Y("loss", title="loss"))
        .properties(width=700, height=350)
    )
    _chart
    return


if __name__ == "__main__":
    app.run()
