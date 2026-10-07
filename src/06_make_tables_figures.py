#!/usr/bin/env python3
"""Build a dependency-free SVG overview from reference pooled metrics."""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics_path = root / "results" / "reference" / "fold_and_pooled_metrics.csv"
    with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scope"] == "pooled_oof"]
    labels = {
        "motif_only": "A  motif",
        "metadata_context": "B  +metadata",
        "kg_context": "C  +KG",
        "kg_context_graph_embedding": "D  +embedding",
    }
    width, height = 880, 420
    left, chart_top, chart_bottom = 190, 45, 350
    max_value = 0.30
    colors = ["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#777;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>',
        '<text x="24" y="28" font-size="20" font-weight="700">Pooled out-of-fold performance</text>',
    ]
    for tick in (0.0, 0.1, 0.2, 0.3):
        y = chart_bottom - (tick / max_value) * (chart_bottom - chart_top)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="840" y2="{y:.1f}"/>')
        parts.append(f'<text x="145" y="{y + 5:.1f}" font-size="13">{tick:.1f}</text>')
    for index, row in enumerate(rows):
        x = left + 45 + index * 155
        macro = float(row["f1_macro"])
        micro = float(row["f1_micro"])
        for offset, value, opacity in ((0, macro, 1.0), (42, micro, 0.55)):
            bar_h = value / max_value * (chart_bottom - chart_top)
            parts.append(
                f'<rect x="{x + offset}" y="{chart_bottom - bar_h:.1f}" width="34" height="{bar_h:.1f}" '
                f'fill="{colors[index]}" opacity="{opacity}" rx="2"/>'
            )
            parts.append(f'<text x="{x + offset + 17}" y="{chart_bottom - bar_h - 7:.1f}" font-size="11" text-anchor="middle">{value:.4f}</text>')
        parts.append(f'<text x="{x + 38}" y="378" font-size="13" text-anchor="middle">{labels[row["condition"]]}</text>')
    parts.extend([
        '<rect x="650" y="17" width="12" height="12" fill="#555"/><text x="670" y="28" font-size="12">macro-F1</text>',
        '<rect x="747" y="17" width="12" height="12" fill="#555" opacity=".55"/><text x="767" y="28" font-size="12">micro-F1</text>',
        f'<line class="axis" x1="{left}" y1="{chart_bottom}" x2="840" y2="{chart_bottom}"/>',
        '</svg>',
    ])
    out = root / "figures" / "performance_overview.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
