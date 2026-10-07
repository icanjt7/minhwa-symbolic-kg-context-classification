#!/usr/bin/env python3
"""Evidence-first Minhwa symbolic inference agent.

This script implements a deterministic multi-role agent pattern:
EvidenceCollector -> SymbolicReasoner -> Verifier -> Curator.
It is intentionally local and auditable, so later LLM agents can call it as a
grounding tool rather than hallucinating symbolic meanings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
DEFAULT_RAW = PRIVATE / "minhwa_raw_combined.csv"
DEFAULT_OBJECT_SYMBOL = PRIVATE / "object_symbol_mapping.csv"
DEFAULT_SUBJECT_SYMBOL = PRIVATE / "subject_symbol_correlation.csv"
DEFAULT_KG_PAIRS = PRIVATE / "symbol_pairs_input.csv"
DEFAULT_GATE = PRIVATE / "evaluation_gate.json"
DEFAULT_OUTPUT = ROOT / "results" / "generated" / "agent" / "symbolic_agent_report.json"

GENERIC_KEYWORDS = {
    "한국전통민화",
    "민화",
    "전통",
    "기타",
    "",
}

NON_SYMBOL_TERMS = {
    "그림",
    "도상",
    "동물",
    "식물",
    "화훼",
    "화목",
    "문자",
    "장식",
    "행사",
    "교육",
    "생활",
    "용도",
    "한지",
    "비단",
    "일필",
    "공필",
    "채색",
    "수묵",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_terms(value: str | None) -> list[str]:
    if not value:
        return []
    terms = []
    for raw in value.replace(";", ",").split(","):
        term = raw.strip()
        if term and term not in GENERIC_KEYWORDS:
            terms.append(term)
    return terms


def strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def load_evidence() -> dict[str, Any]:
    object_rows = read_csv(DEFAULT_OBJECT_SYMBOL)
    subject_rows = read_csv(DEFAULT_SUBJECT_SYMBOL)
    kg_rows = read_csv(DEFAULT_KG_PAIRS)

    object_to_symbols: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_cooccurrence = 1
    for row in object_rows:
        obj = row.get("Visual Object (객체)", "").strip()
        symbol = row.get("Symbolic Meaning (상징)", "").strip()
        count = int(float(row.get("Co-occurrence", "0") or 0))
        max_cooccurrence = max(max_cooccurrence, count)
        if obj and symbol and symbol not in NON_SYMBOL_TERMS:
            object_to_symbols[obj].append({"symbol": symbol, "count": count})

    subject_to_symbols: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        subject = row.get("Genre (화목)", "").strip()
        symbol = row.get("Symbol (상징어)", "").strip()
        freq = int(float(row.get("Frequency (빈도)", "0") or 0))
        ratio = float(row.get("Ratio (%)", "0") or 0)
        if subject and symbol and symbol not in NON_SYMBOL_TERMS:
            subject_to_symbols[subject].append(
                {"symbol": symbol, "frequency": freq, "ratio": ratio}
            )

    kg_motif_to_symbols: dict[str, set[str]] = defaultdict(set)
    for row in kg_rows:
        motif = strip_prefix(row.get("motif", "").strip(), "motif:")
        symbol = strip_prefix(row.get("symbol", "").strip(), "symbol:")
        if motif and symbol and symbol not in NON_SYMBOL_TERMS:
            kg_motif_to_symbols[motif].add(symbol)

    symbol_vocabulary = set()
    for items in object_to_symbols.values():
        symbol_vocabulary.update(item["symbol"] for item in items)
    for items in subject_to_symbols.values():
        symbol_vocabulary.update(item["symbol"] for item in items)
    for symbols in kg_motif_to_symbols.values():
        symbol_vocabulary.update(symbols)

    return {
        "object_to_symbols": object_to_symbols,
        "subject_to_symbols": subject_to_symbols,
        "kg_motif_to_symbols": kg_motif_to_symbols,
        "symbol_vocabulary": symbol_vocabulary,
        "max_cooccurrence": max_cooccurrence,
    }


def add_evidence(
    candidates: dict[str, list[dict[str, Any]]],
    symbol: str,
    source: str,
    detail: str,
    weight: float,
) -> None:
    if not symbol:
        return
    candidates[symbol].append(
        {
            "source": source,
            "detail": detail,
            "weight": round(float(weight), 4),
        }
    )


def collect_evidence(row: dict[str, str], evidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    objects = split_terms(row.get("objects"))
    keywords = split_terms(row.get("keywords"))
    subject_type = (row.get("subject_type") or "").strip()
    symbol_vocabulary = evidence["symbol_vocabulary"]
    blocked_terms = set(objects)
    blocked_terms.update(
        split_terms(row.get("subject_type"))
        + split_terms(row.get("theme_type"))
        + split_terms(row.get("painting_type"))
        + split_terms(row.get("purpose"))
        + split_terms(row.get("era"))
    )

    for keyword in keywords:
        if keyword not in blocked_terms and keyword not in NON_SYMBOL_TERMS:
            add_evidence(
                candidates,
                keyword,
                "keyword",
                f"keywords contains {keyword}",
                1.0,
            )

    for obj in objects:
        for item in evidence["object_to_symbols"].get(obj, []):
            normalized = item["count"] / evidence["max_cooccurrence"]
            weight = 0.55 + (0.35 * normalized)
            add_evidence(
                candidates,
                item["symbol"],
                "object_symbol_mapping",
                f"{obj} -> {item['symbol']} co-occurrence={item['count']}",
                weight,
            )
        for symbol in evidence["kg_motif_to_symbols"].get(obj, set()):
            add_evidence(
                candidates,
                symbol,
                "kg_pair",
                f"{obj} -> {symbol} from mined triples",
                0.72,
            )

    for item in evidence["subject_to_symbols"].get(subject_type, [])[:8]:
        ratio_weight = min(0.68, 0.25 + item["ratio"] / 100)
        add_evidence(
            candidates,
            item["symbol"],
            "subject_symbol_correlation",
            f"{subject_type} -> {item['symbol']} ratio={item['ratio']}%",
            ratio_weight,
        )

    return candidates


def confidence(items: list[dict[str, Any]]) -> float:
    # Noisy-or aggregation keeps multiple weak sources useful without exceeding 1.
    miss_probability = 1.0
    for item in items:
        miss_probability *= 1.0 - min(max(float(item["weight"]), 0.0), 0.99)
    return round(1.0 - miss_probability, 4)


def infer_record(row: dict[str, str], evidence: dict[str, Any], top_k: int) -> dict[str, Any]:
    candidates = collect_evidence(row, evidence)
    keywords = split_terms(row.get("keywords"))
    objects = set(split_terms(row.get("objects")))
    gold_symbols = [
        term
        for term in keywords
        if term not in objects
        and term not in NON_SYMBOL_TERMS
        and term not in GENERIC_KEYWORDS
        and term != row.get("subject_type", "")
    ]
    gold_set = set(gold_symbols)

    ranked = []
    for symbol, items in candidates.items():
        score = confidence(items)
        sources = {item["source"] for item in items}
        if symbol in gold_set or "keyword" in sources:
            status = "supported"
        elif score >= 0.78 and len(sources) >= 2:
            status = "supported"
        elif score >= 0.55:
            status = "plausible"
        else:
            status = "needs_review"
        if symbol in objects and "keyword" not in sources:
            status = "needs_review"
        ranked.append(
            {
                "symbol": symbol,
                "confidence": score,
                "status": status,
                "evidence": sorted(items, key=lambda x: x["weight"], reverse=True),
            }
        )

    ranked.sort(key=lambda x: (x["confidence"], x["symbol"]), reverse=True)
    top_symbols = ranked[:top_k]
    predicted = {item["symbol"] for item in top_symbols}
    matched = predicted & gold_set
    precision = len(matched) / len(predicted) if predicted else math.nan
    recall = len(matched) / len(gold_set) if gold_set else math.nan

    review_notes = []
    if not top_symbols:
        review_notes.append("no_candidate_symbol_evidence")
    if any(item["status"] == "needs_review" for item in top_symbols[:3]):
        review_notes.append("top_claim_needs_review")
    if not gold_set:
        review_notes.append("no_keyword_gold_symbols")

    return {
        "file_name": row.get("file_name", ""),
        "subject_type": row.get("subject_type", ""),
        "objects": split_terms(row.get("objects")),
        "top_symbols": top_symbols,
        "gold_symbols": gold_symbols,
        "record_metrics": {
            "precision_at_k": None if math.isnan(precision) else round(precision, 4),
            "recall_at_k": None if math.isnan(recall) else round(recall, 4),
        },
        "review_notes": review_notes,
    }


def load_gate() -> dict[str, Any] | None:
    if not DEFAULT_GATE.exists():
        return None
    text = DEFAULT_GATE.read_text(encoding="utf-8")
    # Some existing files contain NaN, which Python's json loader accepts.
    return json.loads(text)


def summarize(records: list[dict[str, Any]], gate: dict[str, Any] | None) -> dict[str, Any]:
    precisions = [
        row["record_metrics"]["precision_at_k"]
        for row in records
        if row["record_metrics"]["precision_at_k"] is not None
    ]
    recalls = [
        row["record_metrics"]["recall_at_k"]
        for row in records
        if row["record_metrics"]["recall_at_k"] is not None
    ]
    status_counts: dict[str, int] = defaultdict(int)
    for row in records:
        for symbol in row["top_symbols"]:
            status_counts[symbol["status"]] += 1

    return {
        "n_records": len(records),
        "mean_precision_at_k": round(sum(precisions) / len(precisions), 4) if precisions else None,
        "mean_recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "claim_status_counts": dict(sorted(status_counts.items())),
        "prior_gate": gate,
        "interpretation": (
            "Use as a review assistant. Prior gate suggests symbolic inference can be useful, "
            "but link prediction may still be incomplete."
        ),
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    csv_path = path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "subject_type",
                "objects",
                "rank",
                "symbol",
                "confidence",
                "status",
                "evidence_sources",
                "gold_symbols",
            ],
        )
        writer.writeheader()
        for record in records:
            for index, symbol in enumerate(record["top_symbols"], start=1):
                writer.writerow(
                    {
                        "file_name": record["file_name"],
                        "subject_type": record["subject_type"],
                        "objects": ", ".join(record["objects"]),
                        "rank": index,
                        "symbol": symbol["symbol"],
                        "confidence": symbol["confidence"],
                        "status": symbol["status"],
                        "evidence_sources": ", ".join(
                            sorted({item["source"] for item in symbol["evidence"]})
                        ),
                        "gold_symbols": ", ".join(record["gold_symbols"]),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Limit records for smoke tests.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--csv", action="store_true", help="Also write a flat CSV next to JSON.")
    args = parser.parse_args()

    rows = read_csv(args.input)
    if args.limit:
        rows = rows[: args.limit]

    evidence = load_evidence()
    gate = load_gate()
    records = [infer_record(row, evidence, args.top_k) for row in rows]
    report = {
        "summary": summarize(records, gate),
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv:
        write_csv(records, args.output)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
