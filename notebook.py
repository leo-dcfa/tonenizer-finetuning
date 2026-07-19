import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="full",
    css_file="assets/theme.css",
    layout_file="layouts/notebook.slides.json",
)


@app.cell
def _():
    import base64
    import contextlib
    import functools
    import html
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd
    from pygments import highlight as pygments_highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import JsonLexer, PythonLexer

    ASSETS = Path(__file__).parent / "assets"
    CACHE = Path(__file__).parent / "cache"
    DATA = Path(__file__).parent / "data"
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
        ASSETS,
        AZUL_CHART,
        CACHE,
        DATA,
        HtmlFormatter,
        JsonLexer,
        LOGO_URI,
        PythonLexer,
        alt,
        base64,
        contextlib,
        functools,
        html,
        json,
        mo,
        pd,
    )


@app.cell
def _(CACHE, DATA, json):
    # ── real artifacts from the week's runs — every number on a slide is loaded,
    #    not typed (see PLAN.md runbook) ──
    from scripts.generate_data import (
        CONTACT_LINE,
        GREETING,
        SIGN_OFF,
        STYLE_CARD,
        SYSTEM_PROMPT,
    )

    LOSS = json.loads((CACHE / "loss.json").read_text())
    COMPARISONS = json.loads((CACHE / "comparisons.json").read_text())
    LENS = json.loads((CACHE / "interp" / "logit_lens.json").read_text())
    LOGIT_DIFF = json.loads((CACHE / "interp" / "logit_diff.json").read_text())
    LANDSCAPE = json.loads((CACHE / "landscape.json").read_text())
    TRAIN_EXAMPLE = json.loads((DATA / "train.jsonl").open().readline())

    META = LOSS["meta"]
    TRAINABLE_PCT = 100 * META["trainable_params"] / META["total_params"]
    WALL_MIN = META["wall_seconds"] / 60

    def _fingerprint(key: str) -> int:
        return sum(
            1
            for c in COMPARISONS
            if c[key].startswith(GREETING) and SIGN_OFF in c[key] and CONTACT_LINE in c[key]
        )

    FP_BASE = _fingerprint("base_response")
    FP_TUNED = _fingerprint("tuned_response")
    JUDGE_BASE = sum(c["judge"]["base_score"] for c in COMPARISONS) / len(COMPARISONS)
    JUDGE_TUNED = sum(c["judge"]["tuned_score"] for c in COMPARISONS) / len(COMPARISONS)
    return (
        COMPARISONS,
        FP_BASE,
        FP_TUNED,
        JUDGE_BASE,
        JUDGE_TUNED,
        LANDSCAPE,
        LENS,
        LOGIT_DIFF,
        LOSS,
        META,
        STYLE_CARD,
        SYSTEM_PROMPT,
        TRAINABLE_PCT,
        TRAIN_EXAMPLE,
        WALL_MIN,
    )


@app.cell
def _(HtmlFormatter, JsonLexer, LOGO_URI, PythonLexer, html, mo, pygments_highlight):
    # ── slide helpers — all deck styling flows through these (see STYLE.md) ──

    def esc(text: str) -> str:
        return html.escape(text)

    def _footer(section: str = "") -> str:
        return f"""
        <div class="az-footer">
          <span>{section}</span>
          <span class="az-footer-brand"><img src="{LOGO_URI}" alt="Azul Labs"/> azl.au</span>
        </div>"""

    def slide(title: str, body_html: str, section: str = "", sub: str = "", wide: bool = False):
        """Standard content slide, azl.au pattern: eyebrow label, Fraunces title, body."""
        label_html = f'<p class="az-label">{section}</p>' if section else ""
        sub_html = f'<p class="az-sub">{sub}</p>' if sub else ""
        body_class = "az-body az-body--wide" if wide else "az-body"
        return mo.Html(f"""
        <div class="az-slide">
          {label_html}
          <h1 class="az-title">{title}</h1>
          {sub_html}
          <div class="{body_class}">{body_html}</div>
          {_footer()}
        </div>""")

    def divider(num: int, title: str):
        """Dark section divider — azl.au's blue-900 treatment with ghost numeral."""
        return mo.Html(f"""
        <div class="az-slide az-slide--dark">
          <div style="margin: auto 0;">
            <h1 class="az-title" style="font-size: 4rem;">{title}</h1>
          </div>
          <div class="az-ghost">{num:02d}</div>
          {_footer()}
        </div>""")

    def stat(value: str, label: str) -> str:
        """One stat tile (compose inside az-stats)."""
        return f"""
        <div>
          <div class="az-stat-rule"></div>
          <div class="az-stat-value">{value}</div>
          <div class="az-stat-label">{label}</div>
        </div>"""

    def code_card(
        code: str, caption: str = "", highlight: set[int] | None = None, lang: str = "python"
    ) -> str:
        """Code as exhibit: midnight card, brand syntax colors, gold line highlight."""
        lexer = JsonLexer() if lang == "json" else PythonLexer()
        rendered = pygments_highlight(code.strip(), lexer, HtmlFormatter(nowrap=True))
        lines = rendered.rstrip("\n").split("\n")
        if highlight:
            lines = [
                f'<span class="hl">{ln}</span>' if i + 1 in highlight else ln
                for i, ln in enumerate(lines)
            ]
        cap = f'<div class="az-caption">{caption}</div>' if caption else ""
        return f'<div class="az-code-card"><pre>{"\n".join(lines)}</pre></div>{cap}'

    def exhibit(text: str, caption: str = "") -> str:
        """Plain-text exhibit in the midnight card (style cards, transcripts)."""
        cap = f'<div class="az-caption">{caption}</div>' if caption else ""
        return (
            f'<div class="az-code-card"><pre style="white-space: pre-wrap;">{esc(text)}</pre></div>'
            f"{cap}"
        )

    def cards(prompt: str, base: str, tuned: str) -> str:
        """Base-vs-tuned comparison fragment: identical prompt, two bordered cards."""
        return f"""
        <div class="az-prompt">{esc(prompt)}</div>
        <div class="az-compare">
          <div class="az-card az-card--base">
            <span class="az-chip az-chip--base">BASE MODEL</span>
            <div style="white-space: pre-wrap;">{esc(base)}</div>
          </div>
          <div class="az-card az-card--tuned">
            <span class="az-chip az-chip--tuned">FINE-TUNED</span>
            <div style="white-space: pre-wrap;">{esc(tuned)}</div>
          </div>
        </div>"""

    def frag_header(title: str, section: str = "") -> object:
        """Eyebrow + title fragment for slides composed with mo.vstack."""
        label = f'<p class="az-label">{section}</p>' if section else ""
        return mo.Html(
            f'<div style="padding: 2rem 4rem 0.5rem 4rem; background: var(--cream);">'
            f'{label}<h1 class="az-title">{title}</h1></div>'
        )

    def frag_caption(text: str) -> object:
        return mo.Html(f'<p class="az-caption" style="padding: 0 4rem;">{text}</p>')

    def frag_footer(section: str = "") -> object:
        """Static footer for vstack slides (the absolute one needs an az-slide parent)."""
        return mo.Html(
            f'<div style="margin: 2rem 4rem 0 4rem; padding-top: 0.75rem; '
            f"border-top: 1px solid rgba(74, 85, 104, 0.25); display: flex; "
            f'justify-content: space-between; font-size: 0.875rem; color: var(--slate);">'
            f"<span>{section}</span>"
            f'<span style="display: flex; align-items: center; gap: 0.5rem; font-weight: 500;">'
            f'<img src="{LOGO_URI}" alt="Azul Labs" width="20" height="20"/> azl.au</span></div>'
        )

    def chart_slide(title: str, element: object, caption: str, section: str) -> object:
        """A slide whose body is a live marimo element (chart, widget stack)."""
        return mo.vstack(
            [
                frag_header(title, section),
                mo.Html('<div style="padding: 0 4rem;"></div>'),
                element,
                frag_caption(caption),
                frag_footer(),
            ]
        ).style({"background": "var(--cream)", "padding-bottom": "2rem"})

    return (
        cards,
        chart_slide,
        code_card,
        divider,
        esc,
        exhibit,
        frag_footer,
        frag_header,
        slide,
        stat,
    )


@app.cell
def _(SYSTEM_PROMPT, contextlib, functools):
    # ── live inference: one model in VRAM, adapter toggled on/off per question ──
    BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    ADAPTER = "adapters/council-voice"

    @functools.cache
    def _load_models():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available — use the cached examples instead.")
        free, _ = torch.cuda.mem_get_info()
        if free < 8 * 1024**3:
            raise RuntimeError(
                f"Only {free / 1024**3:.1f} GiB free on the GPU (another model is loaded?). "
                "Free it, or use the cached examples."
            )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="cuda"
        )
        model = PeftModel.from_pretrained(model, ADAPTER)
        model.eval()
        return tokenizer, model

    def ask_both(question: str, max_new_tokens: int = 280) -> tuple[str, str]:
        """Answer the same question with the adapter disabled (base) and enabled."""
        import torch

        tokenizer, model = _load_models()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        answers = {}
        for label in ("base", "tuned"):
            context = model.disable_adapter() if label == "base" else contextlib.nullcontext()
            torch.manual_seed(0)  # same seed → the only difference is the adapter
            with context, torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answers[label] = tokenizer.decode(
                out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
        return answers["base"], answers["tuned"]

    return (ask_both,)


@app.cell
def _(COMPARISONS, mo):
    # ── demo controls (defined here, displayed on the demo slide) ──
    example_pick = mo.ui.dropdown(
        options={f"{c['topic']} — {c['prompt'][:70]}…": i for i, c in enumerate(COMPARISONS)},
        label="Pick a held-out enquiry",
    )
    live_question = mo.ui.text_area(
        placeholder="…or type your own enquiry to Noosa Council (live, runs on this machine)",
        rows=3,
        full_width=True,
    )
    ask_button = mo.ui.run_button(label="Ask both models")
    return ask_button, example_pick, live_question


@app.cell
def _(slide):
    slide(
        "Leo Alves",
        """
        <p class="az-sub">engineer &amp; (accidental) indie researcher</p>
        <p class="az-italic" style="font-size: 2.25rem; margin-top: 3rem;">
          <span style="color: var(--blue-700);">Azul</span> Labs
        </p>
        <p style="margin-top: 0.75rem;">AI and Salesforce consulting.</p>
        <p style="margin-top: 2.5rem;">
          <a href="mailto:leo@azl.au" style="color: var(--blue-700);">leo@azl.au</a>
          · azl.au</p>
        """,
        section="Fine-tuning 101 · Tokenizer - Peregian Digital Hub",
    )
    return


@app.cell
def _(divider):
    divider(1, "Fine tuning")
    return


@app.cell
def _(slide):
    slide(
        "Everyone wants frontier level capability.",
        """
        <p>But...</p>
        <br/>
        <p>
            <ul>
            <li>You might not need frontier level capability at every task.</li>
            <li>Behaviour/style/speech is defined by a different organisation.</li>
            <li>You have have a very specific tone that models struggle to be consistent with.</li>
            <li>You don't want to feed a model dozens of examples for every time you run the same workflow.</li>
            <li>(Despite their beliefs) Not everyone is a snowflake.</li>
            </ul>
        </p>
        """,
        section="01 · Fine-tuning Basics",
    )
    return


@app.cell
def _(slide):
    slide(
        "Do I need it?",
        """
        <p>Most of the time, nope.</p>
        <br/>
        <ul>
          <li>Better Prompting.</li>
          <li>Grounding techniques.</li>
          <li>Fine-tune.</li>
        </ul>
        """,
        section="01 · Finetuning Basics",
    )
    return


@app.cell
def _(slide):
    slide(
        "Other considerations",
        """
        <br/>
        <ul>
          <li><strong>Frontier <s>is</s> can be expensive.</strong> A frontier API call costs orders of
              magnitude more per reply than a small model you own. At thousands of
              requests a month, a fine-tuned 3B doing one job well pays for itself
              almost immediately.</li>
          <li><strong>Privacy.</strong> Some data must never leave the building —
              resident details, health, legal. Fine-tune and serve on your own
              hardware and nothing is ever sent to a third party.</li>
          <li><strong>Latency &amp; offline.</strong> Local small models answer fast and
              keep working when the internet doesn't.</li>
          <li><strong>Control.</strong> Your model can't be deprecated, re-priced, or
              silently changed under you.</li>
        </ul>
        """,
        section="01 · Fine-tuning Basics",
    )
    return


@app.cell
def _(exhibit, slide):
    # ══ Option 1 · Better prompting ══
    _bad = "You are a helpful assistant. Answer the resident's question."
    _good = """You are Noosa Council's customer service assistant.

- Open with: "Thanks for reaching out to Noosa Council."
- Plain English, short paragraphs. If there are steps, number them.
- Warm and empathetic — residents are often frustrated; acknowledge it.
- NEVER invent fees, dates or opening hours. Point to noosa.qld.gov.au
  or (07) 5329 6500 instead.
- Safety-related matters: give the right hotline before anything else.
- Keep replies under 180 words."""
    slide(
        "Option 1: Ask better",
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1.4fr; gap: 1.5rem;">
          <div>
            <span class="az-chip az-chip--base">BAD PROMPT</span>
            {exhibit(_bad)}
          </div>
          <div>
            <span class="az-chip az-chip--tuned">GOOD PROMPT</span>
            {exhibit(_good)}
          </div>
        </div>
        <p style="margin-top: 1.25rem;">The good prompt gets you most of the way, for free.
        Its weaknesses: you pay for those tokens on <strong>every request*</strong>, and on
        long conversations the model drifts away from it.</p>
        """,
        section="01 · Finetuning Basics",
        wide=True,
    )
    return


@app.cell
def _(ASSETS, code_card, slide):
    # ══ Option 2 · Grounding: give the model tools instead of memories ══
    # snippets live in assets/snippets/ — marimo's serializer mangles
    # multi-line string literals inside cells, files are safe
    _bus = (ASSETS / "snippets" / "bus_tool.py").read_text()
    _dv = (ASSETS / "snippets" / "dv_tool.py").read_text()
    slide(
        "Option 2: ground it",
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
          <div style="min-width: 0;">{code_card(_bus, caption="Live facts: the model never memorises a timetable — it asks the source.")}</div>
          <div style="min-width: 0;">{code_card(_dv, caption="Safety-critical routing is code, not model vibes — the hotline can't be hallucinated.", highlight={9, 10})}</div>
        </div>
        <p style="margin-top: 1rem;">Answer is deterministic, model decides when to invoke tool.</p>
        """,
        section="01 · Finetuning Basics",
        wide=True,
    )
    return


@app.cell
def _(COMPARISONS, cards, slide):
    # ══ Option 3 · Fine-tuning: same question, before and after ══
    def _trim(text: str, limit: int = 340) -> str:
        return text if len(text) <= limit else text[: text.rfind(" ", 0, limit)] + " …"

    _c = next((c for c in COMPARISONS if c["topic"] == "noise complaints"), COMPARISONS[0])
    slide(
        "Option 3: fine-tune — same question, before and after",
        cards(_c["prompt"], _trim(_c["base_response"]), _trim(_c["tuned_response"]))
        + """
        <p style="margin-top: 1.25rem;">Same model, same question — the behaviour is now
        <strong>in the weights</strong>: no prompt tokens spent on style, no drift, the
        voice survives any conversation length.</p>
        <p style="margin-top: 0.75rem;">And often <strong>cheaper</strong>: for a narrow
        task, a fine-tuned small model can match a much larger general one — this 3B
        assistant runs for a fraction of what frontier-model API calls cost per reply.</p>
        """,
        section="01 · Finetuning Basics",
        wide=True,
    )
    return


@app.cell
def _(slide):
    # ══ Process · overview (after Databricks' end-to-end guide) ══
    slide(
        "The process, end to end",
        """
        <ol>
          <li><strong>Data preparation</strong> — collect, clean, format, split.</li>
          <li><strong>Choose a base model</strong> — and what the fine-tuned one must do.</li>
          <li><strong>Pick a method</strong> — full fine-tune, LoRA, QLoRA, preference tuning…</li>
          <li><strong>Train</strong> — hyperparameters, context window, watch the loss.</li>
          <li><strong>Evaluate, deploy, monitor</strong> — held-out tests before, drift checks after.</li>
        </ol>
        <p class="az-caption">Framework: Databricks, "The Ultimate Guide to LLM Fine Tuning".</p>
        """,
        section="01 · Finetuning Basics",
    )
    return


@app.cell
def _(slide):
    # ══ Process · step 1: data prep ══
    slide(
        "Step 1: Data preparation",
        """
        <ul>
          <li><strong>Small and clean beats big and noisy</strong> — every bad example
              teaches a bad habit; 1,000 great pairs outperform 10,000 scraped ones.</li>
          <li><strong>Look like production</strong> — train on the messy enquiries you
              actually receive, not tidy textbook ones.</li>
          <li><strong>Same shape as inference</strong> — identical format, system prompt
              and structure the model will see live.</li>
          <li><strong>Hold some back</strong> — split train / validation / test before
              anything else, or you can't tell learning from memorising.</li>
        </ul>
        <p style="margin-top: 1.5rem;">This is where the calendar time goes — ours was
        synthetic (more on that soon), which is why the whole build fit in an evening.</p>
        """,
        section="01 · Finetuning Basics",
    )
    return


@app.cell
def _(slide):
    # ══ Process · step 2: choosing models ══
    slide(
        "Step 2: Choosing a base model",
        """
        <ul>
          <li><strong>Qwen vs Llama vs GLM vs Whatever — does it matter?</strong> Less than the
              internet argues. Any modern model fine-tunes well. What differs: <strong>licence</strong>, available <strong>sizes</strong>, language
              coverage, and tooling/serving support.</li>
          <li><strong>Be scientific</strong>. Most of the AI talk is borderline religious. Fine-tune
              two or three candidates, evaluate and pick a winner. Iterate.</li>
          <li><strong>When bigger is better:</strong> if the small model can't do the task
              at all — even with a perfect prompt — fine-tuning won't create the missing
              capability. Reasoning-heavy task → step up.</li>
          <li><strong>When smaller wins:</strong> the model can do it but inconsistently —
              fine-tuning fixes consistency, and you keep the cost and speed of small.</li>
          <li><strong>Start from the closest model</strong> — chat behaviour → the
              <em>instruct</em> variant, never the raw base.</li>
        </ul>
        """,
        section="01 · Finetuning Basics",
    )
    return


@app.cell
def _(slide):
    # ══ Process · step 3: the methods ══
    slide(
        "Step 3: Pick a method",
        """
        <p>There are a lot of methods. Here are a few.</p>
        <table style="margin-top: 1rem;">
          <tr><th>Method</th><th>What changes</th></tr>
          <tr><td><strong>Full fine-tuning</strong></td><td>every weight</td></tr>
          <tr><td><strong>LoRA</strong> (PEFT)</td><td>small adapter, ~1% of weights</td></tr>
          <tr><td><strong>QLoRA</strong></td><td>LoRA on a compressed base</td></tr>
          <tr><td><strong>Instruction / SFT</strong></td><td>nothing — it's the
              <em>recipe</em>, not the mechanism: train on prompt → good-reply pairs</td></tr>
          <tr><td><strong>Preference tuning</strong> (DPO/RLHF)</td><td>another recipe:
              learn from better-vs-worse pairs</td></tr>
        </table>
        """,
        section="01 · Finetuning Basics",
    )
    return


@app.cell
def _(slide):
    # ══ Process · today's pick ══
    slide(
        "LoRA/QLoRA",
        """
        <p style="font-family: var(--font-display); font-size: 2.25rem; line-height: 1.3;
                  color: var(--midnight); max-width: 56rem;">
          Train <em class="az-em">~1%</em> of the weights, get a
          <em class="az-em">60&nbsp;MB</em> file, in
          <em class="az-em">minutes</em> on a gaming GPU.
        </p>
        <p style="margin-top: 1.5rem;">Supervised fine-tuning with a LoRA adapter — the
        combination that makes everything you're about to see possible on one machine.</p>
        """,
        section="01 · Finetuning Basics",
        sub="De facto industry standard",
    )
    return


@app.cell
def _(META, slide):
    # ══ Process · LoRA anatomy: frozen W + trainable B·A ══
    _r = META["lora_rank"]
    _pct = 100 * META["trainable_params"] / META["total_params"]
    _box = (
        "display:flex;align-items:center;justify-content:center;"
        "font-family:var(--font-display);border-radius:12px;"
    )
    slide(
        "Maths behind LoRA",
        f"""
        <div style="display:flex;align-items:center;justify-content:center;gap:2rem;margin:1rem 0 0.5rem 0;">
          <div style="text-align:center;">
            <div style="{_box}width:230px;height:230px;background:var(--blue-100);border:2px solid var(--blue-700);font-size:3.5rem;color:var(--blue-700);">W</div>
            <p class="az-caption" style="text-align:center;">2048 × 2048 · frozen ❄<br/>4.2M numbers per layer</p>
          </div>
          <div style="font-size:3rem;color:var(--slate);">+</div>
          <div style="text-align:center;">
            <div style="display:flex;align-items:center;gap:1rem;">
              <div style="{_box}width:56px;height:230px;background:var(--white);border:2px solid var(--chart-2);font-size:2rem;color:var(--chart-2);">B</div>
              <div style="font-size:1.75rem;color:var(--slate);">×</div>
              <div style="{_box}width:230px;height:56px;background:var(--white);border:2px solid var(--chart-2);font-size:2rem;color:var(--chart-2);">A</div>
            </div>
            <p class="az-caption" style="text-align:center;">2048 × {_r} and {_r} × 2048 · trainable<br/>66k numbers — 1.6% of the layer</p>
          </div>
        </div>
        <p style="font-family:var(--font-mono);font-size:1.5rem;text-align:center;color:var(--midnight);margin-top:0.5rem;">
          h &nbsp;=&nbsp; <span style="color:var(--blue-700);">W·x</span>
          &nbsp;+&nbsp; (α / r) · <span style="color:var(--chart-2);">B·A·x</span>
        </p>
        <ul style="margin-top:1rem;">
          <li>The original weights <strong>W never move</strong>. The <em>change</em> is forced
              through a bottleneck of rank r&nbsp;=&nbsp;{_r} — LoRA's bet is that the
              adjustment a narrow task needs is <strong>simple</strong>, even when the model isn't.</li>
          <li>Every input <strong>x flows through both paths</strong> and the outputs are added:
              the big frozen path does the thinking, the tiny B·A path learns just the
              difference in behaviour.</li>
        </ul>
        """,
        section="01 · Finetuning Basics",
    )
    return


@app.cell
def _(META, slide):
    # ══ Process · LoRA in plain words ══
    slide(
        "LoRA, in plain words",
        f"""
        <ul>
          <li><strong>A is the down-projection</strong> — squeezes the layer's 2048 working
              numbers into just {META["lora_rank"]}: a summary along {META["lora_rank"]}
              <em>learned</em> directions.</li>
          <li><strong>B is the up-projection</strong> — expands those {META["lora_rank"]}
              numbers back into a full-size correction, added to the layer's output.</li>
          <li><strong>The adapter is a funnel</strong>: 2048 → {META["lora_rank"]} → 2048.
              Everything it does must fit through the narrow waist — it can't rewrite the
              model, only steer it. That constraint is the feature.</li>
          <li><strong>r is the room</strong> — the width of the waist. More r = more
              capacity to change behaviour, bigger adapter.</li>
          <li><strong>α is the volume</strong> — the correction is scaled by
              α/r (ours: {META["lora_alpha"]}/{META["lora_rank"]} = ×2), so the adapter's
              loudness stays comparable when you experiment with r.</li>
        </ul>
        """,
        section="01 · Finetuning Basics",
    )
    return


@app.cell
def _(ASSETS, base64):
    # ── shared: embed the DeepLearning.AI × AMD pasta pages as data URIs ──
    def img_uri(name: str) -> str:
        data = base64.b64encode((ASSETS / name).read_bytes()).decode()
        return f"data:image/png;base64,{data}"

    return (img_uri,)


@app.cell
def _(img_uri, slide):
    # ══ 3b · SFT — the pasta metaphor, part 1 (DeepLearning.AI × AMD) ══
    slide(
        "Two ways to teach: copy the steps…",
        f"""
        <span class="az-chip az-chip--tuned">Fine-tuning</span>
        <img src="{img_uri("sft-pasta.png")}" alt="SFT: watch grandma cook, mimic every step"
             style="width: 100%; margin-top: 0.75rem; border-radius: 12px;
                    border: 1px solid rgba(74,85,104,0.25);"/>
        <p style="margin-top: 1rem;"><strong>Supervised fine-tuning</strong>: you can
        <em>show</em> good behaviour, so the model copies it step by step — perfect for
        voice, format, and consistency.</p>
        <p class="az-caption">Illustration: DeepLearning.AI &amp; AMD, “Post-training of LLMs”.</p>
        """,
        section="01 · Finetuning Basics",
        sub="Fine-tuning vs post-training RL — today is the first one",
    )
    return


@app.cell
def _(img_uri, slide):
    # ══ 3c · RL — the pasta metaphor, part 2 (DeepLearning.AI × AMD) ══
    slide(
        "…or grade the dish",
        f"""
        <span class="az-chip az-chip--base">RL — HOW REASONING MODELS ARE MADE</span>
        <img src="{img_uri("rl-pasta.png")}" alt="RL: only the final dish is graded"
             style="width: 100%; margin-top: 0.75rem; border-radius: 12px;
                    border: 1px solid rgba(74,85,104,0.25);"/>
        <p style="margin-top: 1rem;"><strong>Reinforcement learning</strong>: you can't
        show the steps, but you can <em>grade the result</em> — the model finds its own
        way there. That's how models learn to reason. (The follow-up talk.)</p>
        <p class="az-caption">Illustration: DeepLearning.AI &amp; AMD, “Post-training of LLMs”.</p>
        """,
        section="01 · Finetuning Basics",
    )
    return


@app.cell
def _(divider):
    divider(2, "How training works")
    return


@app.cell
def _(slide):
    # ══ 4 · Training in one idea ══
    slide(
        "Guess the next word, get graded, adjust",
        """
        <p>The whole of training is one loop, repeated thousands of times per second:</p>
        <ul>
          <li>Show the model a training conversation, one word at a time.</li>
          <li>Ask it to <strong>guess the next word</strong> of the reply.</li>
          <li><strong>Grade the guess</strong> — that grade is the “loss” number.</li>
          <li>Nudge the model’s internal dials so the same guess scores better next time.</li>
        </ul>
        <p style="margin-top: 1.5rem;">That’s it. No rules are written down anywhere —
        “always open with <em>Thanks for reaching out to Noosa Council</em>” is never
        stated. The model infers it because every single example does it.</p>
        """,
        section="02 · How training works",
    )
    return


@app.cell
def _(META, TRAINABLE_PCT, code_card, slide):
    # ══ 5 · LoRA ══
    slide(
        "LoRA: train 1%, freeze the rest",
        code_card(
            f"""
from peft import LoraConfig, get_peft_model

config = LoraConfig(r={META["lora_rank"]}, lora_alpha={META["lora_alpha"]}, target_modules=[...])
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: {META["trainable_params"]:,} || all params: {META["total_params"]:,} || {TRAINABLE_PCT:.2f}%
            """,
            caption=(
                "Real output from our run. The base model stays frozen; we train a 60 MB "
                "attachment (an “adapter”) — not the 6 GB model. That's why this runs on a "
                "gaming GPU and finishes in minutes."
            ),
            highlight={6},
        ),
        section="02 · How training works",
        sub="Low-Rank Adaptation — the one piece of jargon worth knowing",
    )
    return


@app.cell
def _(LOSS, alt, chart_slide, pd):
    # ══ 6 · The real loss curve ══
    _df = pd.DataFrame({"step": LOSS["steps"], "loss": LOSS["loss"]})
    _chart = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X("step", title="training step"),
            y=alt.Y("loss", title="loss (how wrong the guesses are)"),
        )
        .properties(width=1150, height=400)
    )
    chart_slide(
        "Watching it learn",
        _chart,
        "Our actual training run: 258 steps, 3 passes over the data. The fast drop is the "
        "model picking up the voice; the floor is where legitimate wording choices remain — "
        "pushing past it would mean memorising the examples.",
        "02 · How training works",
    )
    return


@app.cell
def _(WALL_MIN, slide, stat):
    # ══ 7 · Cost & time ══
    slide(
        "What did that just cost?",
        f"""
        <div class="az-stats">
          {stat(f"{WALL_MIN:.1f} min", "training · one RTX 5090")}
          {stat("~90 min", "generating 1,500 synthetic examples · local 27B, unattended")}
          {stat("~3¢", "same training job on a rented cloud GPU")}
          {stat("<$1", "on a managed fine-tuning service")}
        </div>
        <p style="margin-top: 3rem;">The <span class="az-em">model is not the expensive
        part</span> — the <strong>dataset</strong> is. Ours was synthetic, generated
        overnight by a bigger local model. In the real world, with real data, budget
        days-to-weeks for collecting, cleaning and labelling examples: that's where
        fine-tuning projects actually spend their time.</p>
        """,
        section="02 · How training works",
        sub="Real numbers from the run you just saw",
    )
    return


@app.cell
def _(slide):
    # ══ 8 · Where to run the job ══
    slide(
        "Four ways to run a fine-tune",
        """
        <table>
          <tr><th>Option</th><th>You need</th><th>Cost shape</th><th>Good for</th></tr>
          <tr><td>Your own GPU</td><td>a gaming PC (16 GB+)</td><td>electricity</td>
              <td>experiments, private data</td></tr>
          <tr><td>Rented GPU</td><td>a RunPod/Vast account</td><td>~$0.30–0.60/hr</td>
              <td>occasional bigger jobs</td></tr>
          <tr><td>Managed API</td><td>upload data, pay per token</td><td>~$0.50 per run this size</td>
              <td>no infrastructure at all</td></tr>
          <tr><td>Free notebooks</td><td>a Google account (Colab + Unsloth)</td><td>$0</td>
              <td>learning, this weekend</td></tr>
        </table>
        <p style="margin-top: 1.5rem;">All four produce the same thing: a small adapter
        file you can take anywhere.</p>
        """,
        section="02 · How training works",
    )
    return


@app.cell
def _(divider):
    divider(3, "The training data")
    return


@app.cell
def _(TRAIN_EXAMPLE, code_card, json, slide):
    # ══ 9 · One training example ══
    _pretty = json.dumps(
        {
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"][:110] + ("…" if len(m["content"]) > 110 else ""),
                }
                for m in TRAIN_EXAMPLE["messages"]
            ]
        },
        indent=2,
        ensure_ascii=False,
    )
    slide(
        "What one example looks like",
        code_card(
            _pretty,
            caption=(
                "One real example from our set: a system role, a resident's enquiry, and the "
                "reply we wish the model had given. Fine-tuning data is just thousands of "
                "these — no code, no labels, no special format."
            ),
            lang="json",
        ),
        section="03 · The data",
    )
    return


@app.cell
def _(STYLE_CARD, exhibit, slide):
    # ══ 10 · Synthetic data + the style card ══
    slide(
        "Synthetic data",
        exhibit(
            STYLE_CARD,
            caption=(
                "The style card. A 27-billion-parameter model running on the same GPU generated "
                "1,500 enquiry/reply pairs following it — real resident emails never left the "
                "building, because none were used. Privacy isn't a limitation here; it's the reason "
                "this approach exists."
            ),
        ),
        section="03 · The data",
    )
    return


@app.cell
def _(slide, stat):
    # ══ 11 · Filtering + how much you need ══
    slide(
        "Generated ≠ trustworthy: filter, then count",
        f"""
        <table>
          <tr><th>Check</th><th>Dropped</th></tr>
          <tr><td>Didn't follow the style card</td><td>60</td></tr>
          <tr><td>Answer didn't match its topic</td><td>51</td></tr>
          <tr><td>AI-assistant tells (“as an AI…”)</td><td>13</td></tr>
          <tr><td>Duplicates</td><td>1</td></tr>
          <tr><td><strong>Kept</strong></td><td><strong>1,375 of 1,500 (92%)</strong></td></tr>
        </table>
        <div class="az-stats" style="margin-top: 2.5rem;">
          {stat("~300", "examples: enough for a voice")}
          {stat("1,365", "what we trained on")}
          {stat("10", "held back to test with")}
        </div>
        """,
        section="03 · The data",
        sub="Quality beats quantity — every bad example teaches a bad habit",
    )
    return


@app.cell
def _(divider):
    divider(4, "Did it work? Ask it yourself")
    return


@app.cell
def _(
    COMPARISONS,
    ask_both,
    ask_button,
    cards,
    example_pick,
    frag_footer,
    frag_header,
    live_question,
    mo,
):
    # ══ 12 · THE interactive demo — same weights, adapter off vs on ══
    _result = ""
    if ask_button.value and live_question.value.strip():
        try:
            with mo.status.spinner(title="Asking both models on this machine…"):
                _base, _tuned = ask_both(live_question.value.strip())
            _result = cards(live_question.value.strip(), _base, _tuned)
        except Exception as error:
            _result = f'<p class="az-caption">Live mode unavailable: {error}</p>'
    elif example_pick.value is not None:
        _c = COMPARISONS[example_pick.value]
        _result = cards(_c["prompt"], _c["base_response"], _c["tuned_response"])

    mo.vstack(
        [
            frag_header("Same model, same question — adapter off vs on", "04 · The demo"),
            mo.vstack(
                [
                    mo.hstack([example_pick, ask_button], justify="start", gap=1),
                    live_question,
                    mo.Html(f'<div style="max-width: 100%;">{_result}</div>'),
                ]
            ).style({"padding": "0 4rem"}),
            frag_footer(),
        ]
    ).style({"background": "var(--cream)", "padding-bottom": "2rem"})
    return


@app.cell
def _(FP_BASE, FP_TUNED, JUDGE_BASE, JUDGE_TUNED, slide, stat):
    # ══ 13 · How do we know it worked? ══
    slide(
        "How do we know it worked?",
        f"""
        <p>Two checks, both on the 10 enquiries the model <strong>never saw in
        training</strong>:</p>
        <div class="az-stats" style="margin-top: 2rem;">
          {stat(f"{FP_BASE}/10 → {FP_TUNED}/10", "replies carrying the full council format")}
          {stat(f"{JUDGE_BASE:.1f} → {JUDGE_TUNED:.1f}", "style score /5, graded by a bigger AI")}
        </div>
        <p style="margin-top: 3rem;">Held-out tests matter more than the loss number:
        a model can ace its own homework by memorising it. These questions weren't in
        the homework.</p>
        """,
        section="04 · The demo",
    )
    return


@app.cell
def _(divider):
    divider(5, "Running it for real")
    return


@app.cell
def _(code_card, slide):
    # ══ 14 · Inference ══
    slide(
        "Serving your custom model is one command",
        code_card(
            """
# the fine-tune produced one small file:
#   adapters/council-voice/   (~60 MB)

vllm serve Qwen/Qwen2.5-3B-Instruct --enable-lora \\
    --lora-modules council=adapters/council-voice
# → an OpenAI-compatible API. Point your existing code at it.
            """,
            caption=(
                "Your app talks to it exactly like it talks to any AI API — same libraries, "
                "same code, different URL. Nothing else in your stack changes."
            ),
            highlight={4, 5},
        ),
        section="05 · Running it",
    )
    return


@app.cell
def _(slide):
    # ══ 15 · Hosting options + monthly cost ══
    slide(
        "Where it can live, and what that costs",
        """
        <table>
          <tr><th>Option</th><th>Always-on / month</th><th>Notes</th></tr>
          <tr><td>Rented GPU (marketplace)</td><td>~US$210–250</td>
              <td>cheapest 24/7; shared hardware</td></tr>
          <tr><td>Cloud, Sydney region</td><td>~US$500–620</td>
              <td>data residency + compliance — the council answer</td></tr>
          <tr><td>Serverless per-request</td><td>~US$1–20</td>
              <td>no idle cost; data leaves your infrastructure</td></tr>
          <tr><td>Your own hardware</td><td>~A$35–65 power</td>
              <td>nothing leaves the building; you own uptime</td></tr>
        </table>
        <p style="margin-top: 1.5rem;"><span class="az-em">the
        model is not the expensive part</span>. The costs that matter are people: data, evaluation, operations.</p>
        """,
        section="05 · Running it",
        sub="July 2026 prices — they drift; the shape doesn't",
    )
    return


@app.cell
def _(divider):
    divider(6, "Bonus: look inside the model")
    return


@app.cell
def _(AZUL_CHART, LENS, alt, chart_slide, pd):
    # ══ 16 · Logit lens: which layer does the greeting live in? ══
    _rows = []
    for _name, _ps in LENS["p_by_layer"].items():
        _rows += [{"layer": i, "p": p, "model": _name} for i, p in enumerate(_ps)]
    _df = pd.DataFrame(_rows)
    _chart = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X("layer", title="layer (depth into the model)"),
            y=alt.Y("p", title=f'chance the next word is "{LENS["target_token"]}"'),
            color=alt.Color(
                "model",
                scale=alt.Scale(domain=["tuned", "base"], range=[AZUL_CHART[0], AZUL_CHART[1]]),
            ),
        )
        .properties(width=1150, height=400)
    )
    chart_slide(
        "Where the greeting lives",
        _chart,
        'Reading the model\'s "mind" layer by layer as it starts its reply. The fine-tuned '
        'model becomes certain it will say "Thanks" in the last few layers — the base model '
        "never considers it (≈1%). The 60 MB adapter rewired exactly that decision.",
        "06 · Inside the model",
    )
    return


@app.cell
def _(LOGIT_DIFF, esc, slide):
    # ══ 17 · What the fine-tune promoted ══
    _rows = "".join(
        f"<tr><td><code>{esc(d['token'])}</code></td><td>+{d['delta']:.1f}</td></tr>"
        for d in LOGIT_DIFF["promoted"][:6]
    )
    slide(
        "The words the fine-tune turned up",
        f"""
        <p>Comparing the two models' preferences for the very first word of a reply:</p>
        <table>
          <tr><th>Token</th><th>Boost</th></tr>
          {_rows}
        </table>
        <p style="margin-top: 1.5rem;">The biggest boosts are the <strong>em-dash</strong>
        (the “— Noosa Council Customer Service” sign-off) and fragments of
        <strong>“Thanks”</strong>. The model didn't vaguely get “more polite” —
        specific, inspectable preferences changed, and we can see which.</p>
        """,
        section="06 · Inside the model",
    )
    return


@app.cell
def _(AZUL_CHART, LANDSCAPE, alt, chart_slide, pd):
    # ══ Interp · 1D loss valley along the adapter direction ══
    _df = pd.DataFrame({"scale": LANDSCAPE["alphas"], "loss": LANDSCAPE["loss_1d"]})
    _marks = pd.DataFrame(
        {
            "scale": [0.0, 1.0],
            "loss": [LANDSCAPE["loss_1d"][0], min(LANDSCAPE["loss_1d"], key=lambda v: v)],
            "label": ["base model", "fine-tuned"],
        }
    )
    _marks.loc[1, "loss"] = _df.loc[(_df["scale"] - 1.0).abs().idxmin(), "loss"]
    _line = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X("scale", title="adapter strength (0 = base · 1 = as trained)"),
            y=alt.Y("loss", title="loss on council conversations", scale=alt.Scale(zero=False)),
        )
    )
    _pts = (
        alt.Chart(_marks)
        .mark_point(filled=True, size=140, color=AZUL_CHART[1])
        .encode(x="scale", y="loss")
    )
    _txt = (
        alt.Chart(_marks)
        .mark_text(dy=-14, fontWeight="bold", color=AZUL_CHART[1], fontSize=14)
        .encode(x="scale", y="loss", text="label")
    )
    chart_slide(
        "The valley training walked into",
        (_line + _pts + _txt).properties(width=1150, height=400),
        "The same descent as the loss curve, seen as geometry: walking from the base model "
        "along the exact direction training moved, loss falls into a broad, flat valley. "
        "The flat floor means many nearby weight settings are equally good — the model "
        "isn't at a knife-edge, it's resting in a basin.",
        "06 · Inside the model",
    )
    return


@app.cell
def _(AZUL_CHART, LANDSCAPE, alt, chart_slide, pd):
    # ══ Interp · 2D loss surface slice ══
    _a, _b = LANDSCAPE["a_axis"], LANDSCAPE["b_axis"]
    _sa, _sb = (_a[1] - _a[0]) / 2, (_b[1] - _b[0]) / 2
    _rows = [
        {
            "a": av,
            "a2": av + 2 * _sa,
            "b": bv,
            "b2": bv + 2 * _sb,
            "loss": LANDSCAPE["loss_2d"][j][i],
        }
        for j, bv in enumerate(_b)
        for i, av in enumerate(_a)
    ]
    _grid = pd.DataFrame(_rows)
    _heat = (
        alt.Chart(_grid)
        .mark_rect()
        .encode(
            x=alt.X(
                "a",
                title="adapter direction (0 = base · 1 = fine-tuned)",
                scale=alt.Scale(domain=[_a[0], _a[-1] + 2 * _sa]),
            ),
            x2="a2",
            y=alt.Y(
                "b",
                title="random direction (up to 2.5× the adapter's size)",
                scale=alt.Scale(domain=[_b[0], _b[-1] + 2 * _sb]),
            ),
            y2="b2",
            color=alt.Color(
                "loss",
                scale=alt.Scale(scheme="blues", reverse=True),
                legend=alt.Legend(title="loss (dark = deep)"),
            ),
        )
    )
    _pts = pd.DataFrame(
        {"a": [0.0 + _sa, 1.0 + _sa], "b": [0.0 + _sb, 0.0 + _sb], "label": ["base", "fine-tuned"]}
    )
    _dots = (
        alt.Chart(_pts)
        .mark_point(filled=True, size=160, color=AZUL_CHART[1], stroke="white", strokeWidth=1.5)
        .encode(x="a", y="b")
    )
    _lbl = (
        alt.Chart(_pts)
        .mark_text(dy=-14, fontWeight="bold", color=AZUL_CHART[1], fontSize=14)
        .encode(x="a", y="b", text="label")
    )
    chart_slide(
        "The same valley from above",
        (_heat + _dots + _lbl).properties(width=1000, height=420),
        "A 2-D slice of the 3-billion-dimension loss surface. Left-right: the direction training "
        "moved — all the action. Up-down: a random direction 2.5× bigger — almost nothing "
        "changes. In a space this vast nearly every direction is flat; training's whole job is "
        "finding the rare directions that matter (and why a rank-16 funnel is enough). An "
        "emerging field — developmental interpretability — studies how structure forms "
        "through exactly this geometry.",
        "06 · Inside the model",
    )
    return


@app.cell
def _(slide):
    # ══ 18 · Close ══
    slide(
        "Questions",
        """
        <p class="az-italic" style="font-size: 2.25rem; margin-top: 3rem;">
          Thank <em class="az-em">you</em>.
        </p>
        <p style="margin-top: 2.5rem;">Leo Alves ·
          <a href="mailto:leo@azl.au" style="color: var(--blue-700);">leo@azl.au</a>
          · azl.au</p>
        """,
        section="Fine-tuning 101 · Tokenizer - Peregian Digital Hub",
    )
    return


if __name__ == "__main__":
    app.run()
