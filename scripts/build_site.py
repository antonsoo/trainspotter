"""Build the static sample-report site published to GitHub Pages.

For every real example log under `examples/`, runs it through
trainspotter's own library code -- the same `load_run` / `run_all` /
`render_html` the CLI calls, not a re-implementation -- and writes its
HTML report. The index page's finding counts are computed from those same
results at build time. Labels describe the captures; counts and findings
come from the same analysis.

Usage:
    uv run python scripts/build_site.py [--out-dir site]

Output: `<out-dir>/index.html` plus one `<out-dir>/<example>.html` per
example (default out-dir: `site/`, gitignored). The Pages workflow can
publish this build; a local build does not itself update the hosted site.
"""

from __future__ import annotations

import argparse
import base64
import html
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from trainspotter.detectors import Finding, run_all
from trainspotter.readers import load_run
from trainspotter.report import render_html

REPO_ROOT = Path(__file__).resolve().parent.parent

_DESCRIPTION = (
    "Automatic diagnosis of training runs from their logs: divergence, loss spikes, "
    "overfitting, LR and throughput problems. Real HTML reports on real example runs."
)
_OG_IMAGE = "https://raw.githubusercontent.com/antonsoo/trainspotter/main/docs/assets/og.png"
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(frozen=True)
class ExampleMeta:
    slug: str
    title: str
    blurb: str
    source: str | None = None
    method: str | None = None


# Titles and one-line blurbs are static labels, same spirit as a table
# caption; the finding counts shown next to each one are never hand-typed
# (see main()) -- only what a fresh `run_all()` reports at build time.
EXAMPLES = [
    ExampleMeta(
        "loss-cadence",
        "A late loss drop is still late when logging is sparse",
        "Controlled CPU PyTorch SGD on a scalar quadratic. This file retains step 0 "
        "and steps 900-1000 from one recorded run. Its first low reading is 90% "
        "through the logged training span, despite being the second CSV row.",
        "diagnostic-evidence/slow-sparse.csv",
        "docs/diagnostic-evidence.md",
    ),
    ExampleMeta(
        "gradient-concentration",
        "Constant gradient norms without any clipping",
        "Controlled CPU PyTorch SGD on absolute error. No clipping function is called. "
        "The norm stays at 1 for all 100 updates; an otherwise identical clipped run "
        "has the same pre-clip norms. This history describes concentration, not its cause.",
        "diagnostic-evidence/unclipped-before.csv",
        "docs/diagnostic-evidence.md",
    ),
    ExampleMeta(
        "wandb-cadence",
        "Logging less often is not slower training",
        "A controlled numerical training run recorded with the W&B SDK offline. The logger "
        "changes from every 4 updates to every 20. The report uses trainer/global_step "
        "to divide elapsed time by actual updates.",
        "wandb-axes/stable-training.csv",
        "docs/wandb-axes.md",
    ),
    ExampleMeta(
        "wandb-logging-only",
        "A logging counter cannot measure training speed",
        "The same SDK capture with its training counter removed. The timestamps and loss "
        "are unchanged; _step counts logging events. Per-training-step timing is unavailable.",
        "wandb-axes/stable-logging.csv",
        "docs/wandb-axes.md",
    ),
    ExampleMeta(
        "wandb-slowdown",
        "A measured delay with changing logging cadence",
        "A second offline SDK capture deliberately adds a delay after update 200. The "
        "timing warning uses training updates, while the logger also changes frequency.",
        "wandb-axes/slow-training.csv",
        "docs/wandb-axes.md",
    ),
    ExampleMeta(
        "scheduler-resume",
        "A scheduler reset below its old peak",
        "Controlled CPU PyTorch training: optimizer state is restored at step 200, "
        "but the scheduler counter is deliberately reset. The later rise never "
        "reaches the original peak. A log alone cannot tell this mistake from an intended restart.",
        "schedulers/captured/linear-resume-reset.csv",
        "docs/learning-rate-rebounds.md",
    ),
    ExampleMeta(
        "evaluation-overlap",
        "Training after evaluation ended",
        "Derived from a controlled regression run on synthetic data: evaluation readings "
        "end at step 60; training readings start at 100. The evaluation rise is observed, "
        "but these disjoint intervals cannot support a contemporaneous training trend.",
        "evaluation/captured/late-training.csv",
        "docs/evaluation-evidence.md",
    ),
    ExampleMeta(
        "evaluation-cohorts",
        "Same model, different evaluation data",
        "A frozen regression model alternates between all 128 evaluation examples and "
        "the 32 highest-loss examples. The swings come from deliberate cohort selection. "
        "Here step means evaluation pass, not optimizer update; the weights do not change.",
        "evaluation/captured/fixed-cohort-switch.csv",
        "examples/evaluation/README.md",
    ),
    ExampleMeta(
        "baseline",
        "Baseline",
        "Ordinary warmup + cosine-decay training on the full dataset. Nothing wrong with it.",
    ),
    ExampleMeta(
        "divergence",
        "Divergence",
        "320 healthy steps, then a simulated LR-schedule incident blows the run up.",
    ),
    ExampleMeta(
        "overfitting",
        "Overfitting",
        "40 training examples, no regularization -- eval loss turns upward after step 125.",
    ),
    ExampleMeta(
        "missing_warmup",
        "Missing warmup",
        "Learning rate jumps straight to its peak from step 0, no ramp-up.",
    ),
    ExampleMeta(
        "throughput_drop",
        "Throughput drop",
        "A real, measured mid-run slowdown (an actual time.sleep, not a fabricated number).",
    ),
    ExampleMeta(
        "transformers_tinygpt2",
        "Real transformers.Trainer run",
        "A genuine 2-layer GPT-2 trained via transformers.Trainer, not the numpy MLP.",
    ),
    ExampleMeta(
        "timing-cadence",
        "Timing without step counts",
        "A measured training loop logs less often halfway through. Without step counts, "
        "the report explains why per-step speed is unavailable.",
        "timing/stable-without-steps.csv",
        "docs/timing-evidence.md",
    ),
    ExampleMeta(
        "timing-slowdown",
        "Inspect a measured slowdown",
        "A deliberate delay in a numerical training loop. Inspect the recorded source, "
        "actual step intervals and baseline/recent medians.",
        "timing/slow-with-steps.csv",
        "docs/timing-evidence.md",
    ),
    ExampleMeta(
        "timing-broken-clock",
        "A broken timing window",
        "One timestamp in the same captured log was deliberately moved backwards. "
        "The timing comparison stays unavailable and shows the affected window.",
        "timing/slow-broken-clock.csv",
        "docs/timing-evidence.md",
    ),
]


def _counts(findings: list[Finding]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.severity] += 1
    return counts


# The site's icon. The report pages get it too: without one a browser asks the server for
# /favicon.ico, which the reports' Content-Security-Policy (rightly) refuses.
_ICON = "data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22><rect width=%2232%22 height=%2232%22 rx=%226%22 fill=%22%230b0e11%22/><path d=%22M4 22 L12 20 L18 21 L22 8 L28 6%22 stroke=%22%234fc3f7%22 stroke-width=%222.4%22 fill=%22none%22 stroke-linecap=%22round%22/></svg>"


def build_index(rows: list[tuple[ExampleMeta, dict[str, int]]], generated: str) -> str:
    cards = []
    for meta, counts in rows:
        chips = "".join(
            f'<span class="chip sev-{sev}">{counts[sev]} {sev}</span>' for sev in ("error", "warning", "info")
        )
        cards.append(
            f"""<a class="card" href="{html.escape(meta.slug)}.html">
  <h2>{html.escape(meta.title)}</h2>
  <p>{html.escape(meta.blurb)}</p>
  <div class="chips">{chips}</div>
</a>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trainspotter · sample reports on real training runs</title>
<meta name="description" content="{_DESCRIPTION}">
<link rel="icon" href="{_ICON}">
<meta property="og:type" content="website">
<meta property="og:title" content="trainspotter · sample reports on real training runs">
<meta property="og:description" content="{_DESCRIPTION}">
<meta property="og:url" content="https://antonsoo.github.io/trainspotter/">
<meta property="og:image" content="{_OG_IMAGE}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{_OG_IMAGE}">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="brand-mark" aria-hidden="true">&#9679;</span>
    <h1>TRAINSPOTTER</h1>
    <p class="sub">Sample HTML reports, generated by the CLI against the real example
    logs committed in <code>examples/</code>. Finding counts below are computed fresh
    at each build, not hand-typed.</p>
  </header>
  <main class="grid">
    {"".join(cards)}
  </main>
  <footer>
    <span>Built {html.escape(generated)} &middot;
    <a href="https://github.com/antonsoo/trainspotter">source</a></span>
  </footer>
</div>
</body>
</html>
"""


_CSS = """
:root {
  --bg: #0a0d10; --panel: #10151a; --line: #1f2830; --line-strong: #2c3944;
  --text: #d9e2e8; --text-muted: #8b97a1; --text-dim: #6e7c87;
  --err: #ff5d5d; --warn: #f2b84b; --info: #5fb3ff;
  --mono: ui-monospace, "SFMono-Regular", "Cascadia Mono", Consolas, monospace;
  --sans: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); }
body { font-family: var(--sans); font-size: 15px; line-height: 1.5; }
.wrap { max-width: 900px; margin: 0 auto; padding: 40px 20px 60px; }
header { border-bottom: 1px solid var(--line-strong); padding-bottom: 20px; margin-bottom: 28px; }
.brand-mark { color: var(--info); font-size: 12px; }
h1 { font-family: var(--mono); font-size: 22px; letter-spacing: 0.12em; display: inline; margin-left: 8px; }
.sub { color: var(--text-muted); max-width: 640px; margin-top: 14px; }
.sub code { font-family: var(--mono); color: var(--text); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
.card {
  display: block; background: var(--panel); border: 1px solid var(--line);
  border-radius: 4px; padding: 18px; text-decoration: none; color: inherit;
  transition: border-color 0.15s ease;
}
.card:hover { border-color: var(--info); }
.card h2 { font-size: 16px; margin: 0 0 8px; color: var(--text); }
.card p { font-size: 13px; color: var(--text-muted); margin: 0 0 14px; min-height: 34px; }
.chips { display: flex; gap: 8px; font-family: var(--mono); font-size: 11px; }
.chip { padding: 2px 7px; border-radius: 2px; border: 1px solid currentColor; }
.chip.sev-error { color: var(--err); }
.chip.sev-warning { color: var(--warn); }
.chip.sev-info { color: var(--info); }
footer { margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--line);
  color: var(--text-dim); font-family: var(--mono); font-size: 11px; }
footer a { color: var(--text-dim); }
@media (max-width: 480px) { .grid { grid-template-columns: 1fr; } }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="site", help="Output directory (default: site)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[ExampleMeta, dict[str, int]]] = []
    for meta in EXAMPLES:
        # Load by repo-relative path: the report prints its source path, and a
        # published page must not carry the build machine's absolute paths.
        src = (EXAMPLES_DIR / (meta.source or f"{meta.slug}.trainer_state.json")).relative_to(REPO_ROOT)
        run = load_run(src)
        findings = run_all(run)
        page = render_html(run, findings, title=f"trainspotter - {meta.title}")
        page = page.replace("<title>", f'<link rel="icon" href="{_ICON}">\n<title>', 1)
        # Keep experimental context and the exact input with a downloaded report.
        # They explain the capture; they do not alter the automated findings.
        mime = "text/csv" if src.suffix == ".csv" else "application/json"
        encoded = base64.b64encode(src.read_bytes()).decode("ascii")
        method = meta.method or "README.md#real-demo-data"
        context = (
            '<section class="chart-block sample-context" aria-labelledby="sample-heading">'
            f'<h2 id="sample-heading">{html.escape(meta.title)}</h2>'
            f'<p>{html.escape(meta.blurb)}</p>'
            f'<p><a download="{html.escape(src.name)}" '
            f'href="data:{mime};base64,{encoded}">Download the input log</a> | '
            '<a '
            f'href="https://github.com/antonsoo/trainspotter/blob/main/{html.escape(method)}">'
            'Capture method and limits</a></p></section>'
        )
        assert page.count("<main>") == 1
        page = page.replace("<main>", "<main>" + context, 1)
        (out_dir / f"{meta.slug}.html").write_text(page, encoding="utf-8")
        rows.append((meta, _counts(findings)))
        print(f"  {meta.slug}: {_counts(findings)}")

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    (out_dir / "index.html").write_text(build_index(rows, generated), encoding="utf-8")
    print(f"wrote {out_dir}/index.html and {len(rows)} report(s)")


if __name__ == "__main__":
    main()
