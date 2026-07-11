# Styling Plan — Fine-Tuning 1-1 Deck

Design direction, tokens, and slide templates for `notebook.py` (marimo slides mode).
All color decisions below are **computed, not eyeballed** — contrast ratios and
colorblind separation were validated against WCAG + Machado CVD simulation
(results at the bottom).

## Concept: "coastal editorial"

The azl.au brand, pushed toward a high-end printed report rather than a tech deck:
warm cream paper, ink-dark serif headlines, one restrained gold accent per slide,
azul blue reserved for data and interaction. Sleekness comes from **typography,
whitespace, and consistency** — not effects. No gradients, no shadows heavier than
a hairline, no transitions. One committed light look (it's a projected deck; no
dark-mode variant).

Three rules that do most of the work:

1. **One idea per slide.** If a slide needs two headers, it's two slides.
2. **Gold budget: one gold element per slide** (a highlighted word, a stat, a rule).
   Gold everywhere is gold nowhere.
3. **Blue means data.** Azul blues appear only on charts, diagrams, code accents,
   and links — never as decorative fills.

## Design tokens (`assets/theme.css` custom properties)

```css
:root {
  /* brand (from azl.au) */
  --cream: #FAF8F5;      /* slide surface */
  --midnight: #1A202C;   /* primary ink; inverted-slide surface */
  --slate: #4A5568;      /* secondary ink, captions, footer */
  --gold: #D4A853;       /* accent on midnight, hairlines, large display only */
  --azul: #1E5AA8;  --azul-mid: #3B82C4;  --azul-light: #5BA3D9;

  /* chart-safe variants (validated ≥3:1 on cream — see table below) */
  --chart-1: #1E5AA8;   /* series 1: azul */
  --chart-2: #B08432;   /* series 2: darkened gold (brand gold fails at 2.08:1) */
  --chart-3: #3B82C4;   /* series 3: mid azul */

  /* type */
  --font-display: 'Fraunces', Georgia, serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;

  /* scale (slide-optimized, rem at 16px base) */
  --text-hero: 5rem;      /* hero numbers, title slide name */
  --text-title: 3.25rem;  /* slide titles, Fraunces 600 */
  --text-sub: 1.5rem;     /* subtitles/standfirst, DM Sans 400, slate */
  --text-body: 1.375rem;  /* body/bullets */
  --text-code: 1.125rem;  /* code blocks */
  --text-caption: 0.875rem; /* footer, captions, axis labels */

  --gap: 1rem; --pad-slide: 4rem; --radius: 12px;
  --hairline: 1px solid color-mix(in srgb, var(--slate) 25%, transparent);
}
```

**Fonts are self-hosted** (`assets/fonts/*.woff2` + `@font-face`) so the deck renders
identically with venue WiFi down: Fraunces (600 + italic 400), DM Sans (400/500/700),
JetBrains Mono (400/700). Runbook step: download once via google-webfonts-helper.

## Slide templates

Every slide = one marimo cell emitting `mo.Html`/`mo.md` wrapped by small Python
helpers (`slide()`, `divider()`, `compare()`, `stat()`) so styling lives in one place.

**1. Title / About me** — cream. Azul logo top-left (48px). Name in Fraunces
`--text-hero`; "Azul Labs Pty Ltd · azl.au" in DM Sans slate; the "15+ years / 50+
projects" stats as two side-by-side stat tiles (Fraunces numerals, DM Sans labels).
Tagline "Time to ride the *AI wave*" in Fraunces italic with *AI wave* in gold —
the slide's one gold element.

**2. Section divider** (between the 8 sections) — **inverted: midnight surface**,
cream Fraunces title, oversized ghost section numeral ("03") in gold at 12% opacity
behind the text. These give the deck its rhythm and make the brand palette feel
deliberate. Footer switches to cream-on-midnight automatically.

**3. Content slide** — cream. Fraunces title top-left, gold hairline (2px, 3rem wide)
under the title, DM Sans body at `--text-body`, max 5 bullets, max-width 62rem
(~66ch). Key phrase per slide may be gold **or** bold — never both.

**4. Code slide** — code in a midnight card (`--radius`, `--pad` 2rem) on cream: an
inverted island that makes code feel like an exhibit. Syntax theme mapped to brand:
keywords `--azul-light`, strings `--gold`, comments slate-40%, function names cream,
background midnight. ≤14 lines per slide; the line that matters highlighted with a
gold left-border + 8% gold row tint. Caption under the card in `--text-caption`
slate explaining what to notice.

**5. Pre/post comparison** (the money section) — two cards side by side, equal width:
*Base model* card with slate 1px border + slate "BASE" chip; *Fine-tuned* card with
azul 2px border + azul "FINE-TUNED" chip. Identical prompt shown once above both in
mono. Responses in DM Sans `--text-body`. One pair per slide.

**6. Chart slide** (loss curve, params bar, cost comparison) — Altair with a
registered brand theme:
- surface cream, all text DM Sans slate, axis title `--text-caption`
- series colors `--chart-1/2/3` in fixed order, never cycled; 4+ categories fold
  into "Other"
- 2px lines, 4px rounded bar ends, no vertical gridlines, horizontal gridlines at
  slate 12%
- direct labels on key marks (final loss value, the 0.5% bar); legend only when ≥2
  series
- **one axis per chart** — cost vs params gets two charts, not a dual axis
- slate is text-only, never a series color (fails chroma validation, reads gray)

**7. Stat/hero slide** ("0.5% of weights", "$0.20 per 1M tokens") — a single Fraunces
numeral at `--text-hero` in midnight, unit/context line in DM Sans slate below,
optional gold hairline above. No chart junk around a number that speaks for itself.

## Footer (every slide)

Fixed bar, bottom of viewport: hairline rule on top; left — section name in
`--text-caption` slate; right — Azul logo at 20px (SVG inlined as data URI) +
"azl.au" in DM Sans 500. On midnight dividers: rule and text flip to cream-30%.
No slide numbers (marimo shows progress natively).

## Third-party visuals (circuitsvis, marimo chrome)

Attention-pattern widgets ship their own styling — contain them in a midnight card
like code slides so they read as exhibits, with our caption underneath. Hide marimo
app chrome in run mode via CSS; verify appearance in `marimo run` (not just edit
mode) and export a PDF fallback after the dress rehearsal.

## Validation results (computed)

Chart palette `#1E5AA8, #B08432, #3B82C4` on cream `#FAF8F5` — **all checks pass**
(lightness band, chroma floor, CVD worst-pair ΔE 15.8 ≥ 12 target, contrast ≥3:1
all pairs, all-pairs mode).

| Pair | Ratio | Verdict |
|---|---|---|
| midnight on cream (body text) | 15.4:1 | AAA |
| slate on cream (secondary/captions) | 7.1:1 | AAA |
| cream on midnight (dividers, code) | 15.4:1 | AAA |
| gold on midnight (accents on dark) | 7.4:1 | AAA |
| cream on azul (chips/buttons) | 6.4:1 | AA |
| brand gold `#D4A853` on cream | 2.1:1 | **decor/large display only — never text, never chart series** |
| brand light-azul `#5BA3D9` on cream | 2.6:1 | **decor only; charts use `--chart-3`** |
| slate as a chart series | chroma 0.034 | **fails (reads gray) — text only** |

## Implementation order

1. `assets/theme.css` (tokens → footer → templates as utility classes) + download fonts
2. Python helpers in `notebook.py` (`slide`, `divider`, `compare`, `stat`, code-card)
3. Altair theme registration cell
4. Build slides against the templates; dress rehearsal in `marimo run`; export PDF fallback
