# SPDX-License-Identifier: Apache-2.0
"""Render optimization results to the console and to a standalone HTML report."""

from __future__ import annotations

import html
from typing import List

from .core import OptimizeResult
from .pareto import Candidate


def _rows(result: OptimizeResult) -> List[dict]:
    ref_tps = result.reference.tokens_per_sec or 1.0
    rows = []
    for c in sorted(
        result.candidates, key=lambda x: x.result.tokens_per_sec, reverse=True
    ):
        rows.append(
            {
                "config": c.result.config.label(),
                "tps": c.result.tokens_per_sec,
                "speedup": c.result.tokens_per_sec / ref_tps,
                "agree": c.fidelity.agreement_rate,
                "ci_low": c.fidelity.ci_low,
                "threshold": c.fidelity.threshold,
                "passed": c.fidelity.passed,
                "is_best": result.best is not None and c is result.best,
            }
        )
    return rows


def render_console(result: OptimizeResult) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(f"ServeTune report - {result.model} (~{result.num_params_b}B params)")
    lines.append(
        f"reference: {result.reference.config.label()}  "
        f"{result.reference.tokens_per_sec} tok/s"
    )
    lines.append("")
    header = f"{'config':<26}{'tok/s':>9}{'speedup':>9}{'agree':>8}{'ci_low':>8}{'gate':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in _rows(result):
        marker = " *" if r["is_best"] else ""
        gate = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"{r['config']:<26}{r['tps']:>9.1f}{r['speedup']:>8.2f}x"
            f"{r['agree']*100:>7.1f}%{r['ci_low']*100:>7.1f}%{gate:>7}{marker}"
        )
    lines.append("")
    if result.best is not None:
        b = result.best
        lines.append(
            f"RECOMMENDED: {b.result.config.label()}  "
            f"({b.result.tokens_per_sec / (result.reference.tokens_per_sec or 1):.2f}x faster, "
            f"agreement {b.fidelity.agreement_rate*100:.1f}%, "
            f"gate >= {b.fidelity.threshold*100:.0f}%)"
        )
        lines.append("")
        lines.append("Reproduce:")
        lines.append("  " + _serve_command(result, b))
    else:
        lines.append(
            "No faster config passed the fidelity gate — keep the full-precision reference."
        )
    lines.append("")
    return "\n".join(lines)


def _serve_command(result: OptimizeResult, best: Candidate) -> str:
    cfg = best.result.config
    parts = [f"vllm serve {result.model}"]
    if cfg.quant.value != "fp16":
        parts.append(f"--quantization {'awq' if cfg.quant.value=='int4-awq' else cfg.quant.value}")
    if cfg.kv_dtype.value == "fp8":
        parts.append("--kv-cache-dtype fp8")
    parts.append(f"--max-num-seqs {cfg.max_batch}")
    parts.append(f"--max-model-len {cfg.max_len}")
    if cfg.spec_decode:
        parts.append("--speculative-config '{...auto-selected draft...}'")
    return " ".join(parts)


def render_html(result: OptimizeResult, path: str) -> None:
    rows = _rows(result)
    tr = []
    for r in rows:
        cls = "best" if r["is_best"] else ("fail" if not r["passed"] else "")
        tr.append(
            "<tr class='%s'><td>%s</td><td>%.1f</td><td>%.2fx</td><td>%.1f%%</td>"
            "<td>%.1f%%</td><td>%s</td></tr>"
            % (
                cls,
                html.escape(r["config"]),
                r["tps"],
                r["speedup"],
                r["agree"] * 100,
                r["ci_low"] * 100,
                "PASS" if r["passed"] else "FAIL",
            )
        )
    best_line = (
        html.escape(_serve_command(result, result.best)) if result.best else "(none)"
    )
    doc = f"""<!doctype html><meta charset="utf-8">
<title>ServeTune report — {html.escape(result.model)}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;color:#111}}
table{{border-collapse:collapse;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.4rem .7rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
tr.best{{background:#e6ffed;font-weight:600}}
tr.fail{{color:#999}}
code{{background:#f5f5f5;padding:.2rem .4rem;border-radius:4px}}
</style>
<h1>ServeTune report</h1>
<p><b>{html.escape(result.model)}</b> — ~{result.num_params_b}B params.
Reference {html.escape(result.reference.config.label())} at
{result.reference.tokens_per_sec} tok/s.</p>
<table>
<tr><th>config</th><th>tok/s</th><th>speedup</th><th>agreement</th><th>CI low</th><th>gate</th></tr>
{''.join(tr)}
</table>
<p><b>Recommended serve command:</b><br><code>{best_line}</code></p>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
