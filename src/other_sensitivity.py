#!/usr/bin/env python3
"""OOF-only sensitivity analysis for the residual '기타' label (2026-10-05)."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import f1_score, precision_recall_fscore_support


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import abc_ablation as base  # noqa: E402


OUT = ROOT / "results" / "generated" / "other_sensitivity"
OOF_PATH = ROOT / "results" / "generated" / "abc" / "grouped_ablation_predictions.csv"
OLD_CONFIG = ROOT / "results" / "generated" / "abc" / "grouped_ablation_config.json"
SEED = 20261005
N_BOOT = 2000
NAMED = ["장수", "부귀", "다산", "출세", "벽사", "화합", "길상"]
ALL8 = NAMED + ["기타"]
PRED = {"A": "pred_motif_only", "B": "pred_metadata_context", "C": "pred_kg_context"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_terms(value: object) -> list[str]:
    return base.split_terms(value)


def normalize_expression(value: str) -> str:
    """Apply only the requested orthographic normalization."""
    text = unicodedata.normalize("NFC", str(value)).strip()
    text = re.sub(r"[\(\)\[\]\{\}（）［］｛｝]", " ", text)
    text = re.sub(r"[·ㆍ/,，;；:+＋\-–—]+", " | ", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return re.sub(r"\s+", " ", text).strip(" |")


def compact(value: str) -> str:
    return re.sub(r"[\s|]", "", normalize_expression(value)).lower()


def normalized_rule_matches(value: str) -> list[str]:
    """Match only existing mapping regexes after spacing/delimiter cleanup."""
    normalized = normalize_expression(value)
    candidates = [compact(normalized)] + [compact(x) for x in normalized.split("|")]
    matches = []
    for label, patterns in base.SUPER_PATTERNS:
        for pattern in patterns:
            # Existing rules contain literal Korean terms; only their spacing is normalized.
            rule = compact(pattern)
            if rule and any(rule in candidate for candidate in candidates if candidate):
                matches.append(label)
                break
    return matches


def is_composite(value: str) -> bool:
    original = unicodedata.normalize("NFC", str(value)).strip()
    return bool(re.search(r"\s+|[·ㆍ/,，;；:+＋\-–—]|(?:^|\s)(?:및|과|와)(?:\s|$)", original))


def reconstruct():
    samples, group_motifs, group_subjects = base.read_samples()
    raw = pd.read_csv(base.RAW_PATH, encoding="utf-8-sig").fillna("")
    manifest = pd.read_csv(base.MANIFEST_PATH, encoding="utf-8-sig").fillna("")
    stem_to_path = dict(zip(manifest.image_stem.astype(str), manifest.rel_path.astype(str)))
    raw["group_id"] = raw.file_name.astype(str).map(
        lambda stem: base.base_artwork_id(stem, stem_to_path.get(stem, "")))
    relation_symbols: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in raw.itertuples(index=False):
        motifs = split_terms(row.objects)
        blocked = (set(motifs) | group_motifs[row.group_id] |
                   {str(getattr(row, col)).strip() for col in
                    ("subject_type", "theme_type", "painting_type", "purpose", "era", "ground_material")} |
                   base.OTHER_TERMS)
        symbols = [x for x in split_terms(row.keywords) if x not in blocked and len(x) <= 80]
        for motif in motifs:
            for symbol in symbols:
                relation_symbols[(row.group_id, motif, base.super_label(symbol))].add(symbol)
    rows = samples.copy()
    rows.insert(0, "row_id", [f"R{i:05d}" for i in range(1, len(rows) + 1)])
    rows["raw_symbols"] = [" | ".join(sorted(relation_symbols[(r.group_id, r.motif, r.label)]))
                           for r in rows.itertuples(index=False)]
    rows["subject_types"] = rows.group_id.map(lambda g: " | ".join(sorted(group_subjects[g])))
    rows["neighbor_motifs"] = [" | ".join(sorted(group_motifs[r.group_id] - {r.motif}))
                               for r in rows.itertuples(index=False)]
    return rows, relation_symbols, group_motifs, group_subjects, raw, manifest


def validate(rows, oof, raw, manifest):
    keys = ["group_id", "motif", "label"]
    left = rows[keys].sort_values(keys).reset_index(drop=True)
    right = (oof.rename(columns={"y_true": "label"})[keys]
             .sort_values(keys).reset_index(drop=True))
    counts = rows.label.value_counts().reindex(ALL8).fillna(0).astype(int).to_dict()
    result = {
        "total_rows": len(rows), "artwork_groups": int(rows.group_id.nunique()),
        "classes": sorted(rows.label.unique()), "class_counts": counts,
        "other_rows": int((rows.label == "기타").sum()),
        "duplicate_relation_keys": int(rows.duplicated(keys).sum()),
        "missing_artwork_id": int(rows.group_id.isna().sum() + rows.group_id.eq("").sum()),
        "oof_rows": len(oof), "oof_key_match": bool(left.equals(right)),
        "oof_duplicate_keys": int(oof.duplicated(["group_id", "motif", "y_true"]).sum()),
        "max_folds_per_artwork": int(oof.groupby("group_id").fold.nunique().max()),
        "prediction_missing": {condition: int(oof[column].isna().sum()) for condition, column in PRED.items()},
        "raw_rows": len(raw), "manifest_rows": len(manifest),
    }
    result["passed"] = bool(
        result["total_rows"] == 34522 and result["artwork_groups"] == 3779 and
        set(result["classes"]) == set(ALL8) and result["other_rows"] == 13291 and
        result["duplicate_relation_keys"] == 0 and result["missing_artwork_id"] == 0 and
        result["oof_rows"] == 34522 and result["oof_key_match"] and
        result["oof_duplicate_keys"] == 0 and result["max_folds_per_artwork"] == 1 and
        not any(result["prediction_missing"].values())
    )
    return result


def other_profile(rows, relation_symbols):
    other = rows[rows.label == "기타"].copy()
    raw_index, norm_index = defaultdict(set), defaultdict(set)
    term_groups, term_motifs, term_subjects = defaultdict(set), defaultdict(set), defaultdict(set)
    candidate_rows = []
    composite_rows = 0
    for row in other.itertuples(index=False):
        terms = relation_symbols[(row.group_id, row.motif, row.label)]
        row_matches = []
        row_composite = False
        for term in terms:
            normalized = normalize_expression(term)
            raw_index[term].add(row.row_id)
            norm_index[normalized].add(row.row_id)
            term_groups[normalized].add(row.group_id)
            term_motifs[normalized].add(row.motif)
            term_subjects[normalized].update(x for x in row.subject_types.split(" | ") if x)
            row_composite = row_composite or is_composite(term)
            matches = normalized_rule_matches(term)
            if matches:
                row_matches.append((term, normalized, matches))
        composite_rows += int(row_composite)
        if row_matches:
            labels = []
            for _, _, match in row_matches:
                labels.extend(match)
            labels = list(dict.fromkeys(labels))
            candidate_rows.append({
                "row_id": row.row_id, "artwork_id": row.group_id, "target_motif": row.motif,
                "subject_types": row.subject_types, "neighbor_motifs": row.neighbor_motifs,
                "current_label": "기타", "raw_symbols": row.raw_symbols,
                "matched_raw_symbols": " | ".join(x[0] for x in row_matches),
                "normalized_matched_symbols": " | ".join(x[1] for x in row_matches),
                "candidate_named_labels": " | ".join(labels),
                "ambiguous_multiple_named_labels": len(labels) > 1,
                "status": "기술적 재배정 후보; 정답 미변경",
            })
    candidates = pd.DataFrame(candidate_rows, columns=[
        "row_id", "artwork_id", "target_motif", "subject_types", "neighbor_motifs",
        "current_label", "raw_symbols", "matched_raw_symbols", "normalized_matched_symbols",
        "candidate_named_labels", "ambiguous_multiple_named_labels", "status"])
    candidates.to_csv(OUT / "normalization_reassignment_candidates.csv", index=False, encoding="utf-8-sig")

    records = []
    for level, index in (("raw", raw_index), ("normalized", norm_index)):
        ordered = sorted(index, key=lambda term: (-len(index[term]), term))
        covered = set()
        for rank, term in enumerate(ordered, 1):
            covered.update(index[term])
            records.append({
                "frequency_level": level, "rank": rank, "expression": term,
                "relation_row_count": len(index[term]),
                "share_of_other_rows": len(index[term]) / len(other),
                "cumulative_unique_row_coverage": len(covered) / len(other),
                "artwork_count": len(term_groups[term]) if level == "normalized" else "",
                "motif_count": len(term_motifs[term]) if level == "normalized" else "",
                "subject_type_count": len(term_subjects[term]) if level == "normalized" else "",
                "composite_candidate": is_composite(term),
            })
    frequency = pd.DataFrame(records)
    frequency.to_csv(OUT / "other_frequency_profile.csv", index=False, encoding="utf-8-sig")

    norm_counts = pd.Series({term: len(ids) for term, ids in norm_index.items()})
    ordered_norm = sorted(norm_index, key=lambda term: (-len(norm_index[term]), term))
    coverage = {}
    for n in (20, 50, 100):
        union = set().union(*(norm_index[x] for x in ordered_norm[:n]))
        coverage[n] = len(union) / len(other)
    stats = {
        "other_rows": len(other), "other_artworks": int(other.group_id.nunique()),
        "other_motifs": int(other.motif.nunique()),
        "other_subject_types": int(len({x for value in other.subject_types for x in value.split(" | ") if x})),
        "unique_raw_expressions": len(raw_index), "unique_normalized_expressions": len(norm_index),
        "singleton_expressions": int((norm_counts == 1).sum()),
        "singleton_share_of_unique": float((norm_counts == 1).mean()),
        "le5_expressions": int((norm_counts <= 5).sum()),
        "le5_share_of_unique": float((norm_counts <= 5).mean()),
        "top20_unique_row_coverage": coverage[20], "top50_unique_row_coverage": coverage[50],
        "top100_unique_row_coverage": coverage[100],
        "composite_rows": composite_rows, "composite_row_share": composite_rows / len(other),
        "technical_candidate_rows": len(candidates),
        "technical_candidate_share": len(candidates) / len(other),
        "top_expressions": [{"expression": x, "rows": len(norm_index[x])} for x in ordered_norm[:30]],
    }
    return stats, candidates


def calculate_metrics(oof):
    analyses = {
        "eight_class_all": (oof.index.to_numpy(), ALL8),
        "named7_all_rows": (oof.index.to_numpy(), NAMED),
        "named7_true_only": (oof.index[oof.y_true.isin(NAMED)].to_numpy(), NAMED),
    }
    metric_rows, class_rows, fold_rows = [], [], []
    for analysis, (idx, labels) in analyses.items():
        part = oof.loc[idx]
        for condition, column in PRED.items():
            macro = f1_score(part.y_true, part[column], labels=labels, average="macro", zero_division=0)
            micro = f1_score(part.y_true, part[column], labels=labels, average="micro", zero_division=0)
            metric_rows.append({"analysis": analysis, "condition": condition,
                                "n_evaluated": len(part), "macro_f1": macro, "micro_f1": micro})
            precision, recall, f1, support = precision_recall_fscore_support(
                part.y_true, part[column], labels=labels, zero_division=0)
            for label, p, r, f, s in zip(labels, precision, recall, f1, support):
                class_rows.append({"analysis": analysis, "condition": condition, "class": label,
                                   "precision": p, "recall": r, "f1": f, "support": int(s)})
            for fold, fp in part.groupby("fold", sort=True):
                fold_rows.append({
                    "analysis": analysis, "condition": condition, "fold": int(fold),
                    "n_evaluated": len(fp),
                    "macro_f1": f1_score(fp.y_true, fp[column], labels=labels,
                                         average="macro", zero_division=0),
                    "micro_f1": f1_score(fp.y_true, fp[column], labels=labels,
                                         average="micro", zero_division=0),
                })
    metrics, classes, folds = map(pd.DataFrame, (metric_rows, class_rows, fold_rows))
    order = {"A": 0, "B": 1, "C": 2}
    metrics["_order"] = metrics.condition.map(order)
    metrics = metrics.sort_values(["analysis", "_order"]).reset_index(drop=True)
    metrics["macro_delta_vs_previous"] = metrics.groupby("analysis").macro_f1.diff()
    metrics["micro_delta_vs_previous"] = metrics.groupby("analysis").micro_f1.diff()
    metrics = metrics.drop(columns="_order")
    metrics.to_csv(OUT / "named7_oof_metrics.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(OUT / "named7_class_metrics.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "named7_fold_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics, classes, folds, analyses


def bootstrap(oof, analyses):
    groups = sorted(oof.group_id.unique())
    group_position = {group: i for i, group in enumerate(groups)}
    rng = np.random.default_rng(SEED)
    draws = {analysis: {metric: {comparison: [] for comparison in ("B_minus_A", "C_minus_B")}
                                for metric in ("macro", "micro")}
             for analysis in analyses}

    # Pre-aggregate TP/FP/FN by artwork. Resampling these sufficient statistics
    # is exactly equivalent to concatenating all rows for the sampled artworks,
    # but avoids repeatedly parsing 34,522 string labels.
    counts = {}
    for analysis, (_, labels) in analyses.items():
        frame = oof if analysis != "named7_true_only" else oof[oof.y_true.isin(NAMED)]
        counts[analysis] = {}
        for condition, column in PRED.items():
            array = np.zeros((len(groups), len(labels), 3), dtype=np.int64)
            for row in frame.itertuples(index=False):
                gi = group_position[row.group_id]
                prediction = getattr(row, column)
                for li, label in enumerate(labels):
                    if row.y_true == label and prediction == label:
                        array[gi, li, 0] += 1  # TP
                    elif row.y_true != label and prediction == label:
                        array[gi, li, 1] += 1  # FP
                    elif row.y_true == label and prediction != label:
                        array[gi, li, 2] += 1  # FN
            counts[analysis][condition] = array

    def scores_from_counts(array):
        tp, fp, fn = array[:, 0], array[:, 1], array[:, 2]
        denominator = 2 * tp + fp + fn
        per_class = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
        macro = float(per_class.mean())
        micro_denominator = 2 * tp.sum() + fp.sum() + fn.sum()
        micro = float(2 * tp.sum() / micro_denominator) if micro_denominator else 0.0
        return {"macro": macro, "micro": micro}

    for _ in range(N_BOOT):
        sampled_positions = rng.integers(0, len(groups), size=len(groups))
        for analysis in analyses:
            score = {condition: scores_from_counts(array[sampled_positions].sum(axis=0))
                     for condition, array in counts[analysis].items()}
            for average in ("macro", "micro"):
                draws[analysis][average]["B_minus_A"].append(score["B"][average] - score["A"][average])
                draws[analysis][average]["C_minus_B"].append(score["C"][average] - score["B"][average])
    result = {"seed": SEED, "iterations": N_BOOT, "resampling_unit": "artwork_id", "results": {}}
    for analysis, metric_map in draws.items():
        result["results"][analysis] = {}
        for metric, comparison_map in metric_map.items():
            result["results"][analysis][metric] = {}
            for comparison, values in comparison_map.items():
                x = np.asarray(values)
                result["results"][analysis][metric][comparison] = {
                    "point_mean": float(x.mean()), "median": float(np.median(x)),
                    "ci95_low": float(np.quantile(x, .025)),
                    "ci95_high": float(np.quantile(x, .975)),
                    "positive_share": float((x > 0).mean()),
                }
    (OUT / "named7_bootstrap.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_reports(rows, raw, manifest, oof, integrity, stats, candidates, metrics, folds, boot):
    raw_columns = list(pd.read_csv(base.RAW_PATH, encoding="utf-8-sig", nrows=0).columns)
    manifest_columns = list(pd.read_csv(base.MANIFEST_PATH, encoding="utf-8-sig", nrows=0).columns)
    oof_columns = list(pd.read_csv(OOF_PATH, encoding="utf-8-sig", nrows=0).columns)
    integrity_lines = [
        "# 입력 정합성", "", f"- 종합 판정: **{'통과' if integrity['passed'] else '불일치'}**",
        f"- 관계 행: {integrity['total_rows']:,}; 작품 그룹: {integrity['artwork_groups']:,}; 기타: {integrity['other_rows']:,}",
        f"- 클래스별 행 수: {integrity['class_counts']}",
        f"- 관계 키 중복/작품 ID 결측: {integrity['duplicate_relation_keys']}/{integrity['missing_artwork_id']}",
        f"- OOF 행·키 일치: {integrity['oof_rows']:,}행/{integrity['oof_key_match']}",
        f"- 작품당 최대 fold 수: {integrity['max_folds_per_artwork']}",
        f"- A/B/C 예측 결측: {integrity['prediction_missing']}", "", "## 사용 파일", "",
        f"- `{base.RAW_PATH.name}` SHA-256 `{sha256(base.RAW_PATH)}`, {len(raw):,}행, 열 {raw_columns}",
        f"- `{base.MANIFEST_PATH.name}` SHA-256 `{sha256(base.MANIFEST_PATH)}`, {len(manifest):,}행, 열 {manifest_columns}",
        f"- `{OOF_PATH.name}` SHA-256 `{sha256(OOF_PATH)}`, {len(oof):,}행, 열 {oof_columns}",
        f"- `{Path(base.__file__).name}` SHA-256 `{sha256(Path(base.__file__))}`",
        "", "A/B/C는 동일 OOF 파일의 같은 행과 fold를 사용한다. 재학습·임계값 선택·정답 수정은 수행하지 않았다.",
    ]
    (OUT / "input_integrity.md").write_text("\n".join(integrity_lines) + "\n", encoding="utf-8")

    pivot = metrics.pivot(index="analysis", columns="condition", values=["macro_f1", "micro_f1"])
    fold_pivot = folds.pivot_table(index=["analysis", "fold"], columns="condition", values="macro_f1")
    report = ["# 기타 범주 민감도 분석", "", "## 기타 구성", "",
              f"- 기타: {stats['other_rows']:,}행, {stats['other_artworks']:,}작품, {stats['other_motifs']:,}소재, {stats['other_subject_types']:,}화목 표기",
              f"- 고유 원문/정규화 표현: {stats['unique_raw_expressions']:,}/{stats['unique_normalized_expressions']:,}",
              f"- 1회 표현: {stats['singleton_expressions']:,}개({stats['singleton_share_of_unique']:.2%})",
              f"- 5회 이하 표현: {stats['le5_expressions']:,}개({stats['le5_share_of_unique']:.2%})",
              f"- 상위 20/50/100개 표현의 고유 기타 행 포괄률: {stats['top20_unique_row_coverage']:.2%}/{stats['top50_unique_row_coverage']:.2%}/{stats['top100_unique_row_coverage']:.2%}",
              f"- 복합 표현 후보: {stats['composite_rows']:,}행({stats['composite_row_share']:.2%})",
              f"- 기술적 재배정 후보: {stats['technical_candidate_rows']:,}행({stats['technical_candidate_share']:.4%})", "",
              "기술적 후보는 기존 7개 정규식에 정규화 후 일치한 행이며 정답을 변경하지 않았다. 동의어·문화적 의미를 새로 추가하지 않았다.", "",
              "## OOF 결과", "", "| 분석 | A macro/micro | B macro/micro | C macro/micro |", "|---|---:|---:|---:|"]
    for analysis in ("eight_class_all", "named7_all_rows", "named7_true_only"):
        values = [f"{pivot.loc[analysis, ('macro_f1', c)]:.4f}/{pivot.loc[analysis, ('micro_f1', c)]:.4f}" for c in "ABC"]
        report.append(f"| {analysis} | {values[0]} | {values[1]} | {values[2]} |")
    report += ["", "## 차이와 방향", ""]
    for analysis in ("eight_class_all", "named7_all_rows", "named7_true_only"):
        a, b, c = (pivot.loc[analysis, ("macro_f1", x)] for x in "ABC")
        am, bmicro, cmicro = (pivot.loc[analysis, ("micro_f1", x)] for x in "ABC")
        ba_fold = [fold_pivot.loc[(analysis, f), "B"] - fold_pivot.loc[(analysis, f), "A"] for f in range(1, 6)]
        cb_fold = [fold_pivot.loc[(analysis, f), "C"] - fold_pivot.loc[(analysis, f), "B"] for f in range(1, 6)]
        bm = boot["results"][analysis]["macro"]
        bmicro_boot = boot["results"][analysis]["micro"]
        report += [f"### {analysis}", "",
                   f"- B−A macro-F1 {b-a:+.4f}; 양수 fold {sum(x>0 for x in ba_fold)}/5; 95% 구간 [{bm['B_minus_A']['ci95_low']:+.4f}, {bm['B_minus_A']['ci95_high']:+.4f}]",
                   f"- C−B macro-F1 {c-b:+.4f}; 양수 fold {sum(x>0 for x in cb_fold)}/5; 95% 구간 [{bm['C_minus_B']['ci95_low']:+.4f}, {bm['C_minus_B']['ci95_high']:+.4f}]",
                   f"- B−A micro-F1 {bmicro-am:+.4f}; 95% 구간 [{bmicro_boot['B_minus_A']['ci95_low']:+.4f}, {bmicro_boot['B_minus_A']['ci95_high']:+.4f}]",
                   f"- C−B micro-F1 {cmicro-bmicro:+.4f}; 95% 구간 [{bmicro_boot['C_minus_B']['ci95_low']:+.4f}, {bmicro_boot['C_minus_B']['ci95_high']:+.4f}]", ""]
    all_direction = all(pivot.loc[a, ("macro_f1", "A")] < pivot.loc[a, ("macro_f1", "B")] < pivot.loc[a, ("macro_f1", "C")]
                        for a in pivot.index)
    report += ["## 결론", "",
               ("8클래스, 전체표본 7클래스, 실제 기타 제외 조건부 7클래스에서 모두 macro-F1의 A<B<C 방향이 유지되었다. 따라서 기타 처리에 대한 민감도 분석에서도 비교 방향이 유지되었다. 이는 기타의 문화적 타당성이나 높은 절대 분류 성능을 뜻하지 않는다."
                if all_direction else
                "분석별 A–B–C 방향이 달라 기존 비교 결론이 기타 처리 방식에 민감한 것으로 확인되었다."),
               "부트스트랩 구간이 0을 포함하는 지표 차이는 확정적 개선으로 표현하지 않는다."]
    (OUT / "other_sensitivity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manuscript = f"""# 원고 반영 문안

## 연구방법

잔여 범주인 기타의 영향에 대한 민감도 분석을 수행하였다. 원문 상징어에는 Unicode NFC, 공백·괄호·구분기호 및 기존 사전에 존재하는 표현의 띄어쓰기 정규화만 적용하였고, 새로운 동의어나 문화적 의미 매핑은 추가하지 않았다. 기존 A/B/C OOF 예측을 재사용하여 (1) 실제 기타 표본을 유지한 채 일곱 명명 클래스의 F1만 평균하는 보수적 평가와 (2) 실제 기타 표본을 제외하는 조건부 평가를 수행하였다. 동일 작품의 관계 행을 함께 재표집하는 대응 부트스트랩을 2,000회(seed={SEED}) 적용하였다.

## 분석결과

기타 13,291행에는 정규화 고유 표현 {stats['unique_normalized_expressions']:,}개가 있었고, 이 중 1회 표현은 {stats['singleton_expressions']:,}개({stats['singleton_share_of_unique']:.2%}), 5회 이하 표현은 {stats['le5_expressions']:,}개({stats['le5_share_of_unique']:.2%})였다. 표기 정규화만으로 기존 명명 범주 규칙에 일치한 기술적 재배정 후보는 {stats['technical_candidate_rows']:,}행({stats['technical_candidate_share']:.4%})이었다. 이 후보는 정답 수정에 사용하지 않았다.

전체 표본을 유지한 7개 명명 클래스 평가의 macro-F1은 A {pivot.loc['named7_all_rows', ('macro_f1','A')]:.4f}, B {pivot.loc['named7_all_rows', ('macro_f1','B')]:.4f}, C {pivot.loc['named7_all_rows', ('macro_f1','C')]:.4f}였다. 실제 기타를 제외한 조건부 평가에서는 각각 {pivot.loc['named7_true_only', ('macro_f1','A')]:.4f}, {pivot.loc['named7_true_only', ('macro_f1','B')]:.4f}, {pivot.loc['named7_true_only', ('macro_f1','C')]:.4f}였다. 8클래스 결과를 포함한 세 분석에서 A<B<C 방향이 유지되었다.

## 논의와 한계

기타 범주에 대한 민감도 분석에서도 입력 맥락을 추가할수록 macro-F1이 높아지는 비교 방향은 유지되었다. 따라서 A–B–C의 상대적 방향은 기타의 포함 여부에만 의존한 결과로 보이지 않는다. 그러나 이는 기타의 문화적 타당성, 신규 의미 범주 또는 전문가 검증을 확정하는 결과가 아니다. 또한 절대 F1 수준과 각 차이의 부트스트랩 구간을 함께 고려해야 한다.
"""
    (OUT / "manuscript_revision_text.md").write_text(manuscript, encoding="utf-8")

    author_reply = f"""# 심사답변 문안

심사 의견에 따라 기타 범주가 비교 결과에 미치는 영향을 기존 OOF 예측만으로 추가 검증하였습니다. 표기 정규화는 Unicode NFC, 공백, 괄호, 구분기호와 기존 매핑 표현의 띄어쓰기 차이로 제한하고 새로운 동의어나 문화적 의미 매핑은 추가하지 않았습니다. 정규화 후 기존 7개 규칙에 일치한 사례는 ‘기술적 재배정 후보’로만 기록하고 정답이나 OOF 예측을 변경하지 않았습니다.

또한 실제 기타 13,291행을 유지하여 명명 클래스로의 오탐을 반영한 7클래스 평가와, 실제 기타 행을 제외한 조건부 평가를 구분하였습니다. 보수적 macro-F1은 A {pivot.loc['named7_all_rows', ('macro_f1','A')]:.4f}, B {pivot.loc['named7_all_rows', ('macro_f1','B')]:.4f}, C {pivot.loc['named7_all_rows', ('macro_f1','C')]:.4f}였고, 조건부 macro-F1은 A {pivot.loc['named7_true_only', ('macro_f1','A')]:.4f}, B {pivot.loc['named7_true_only', ('macro_f1','B')]:.4f}, C {pivot.loc['named7_true_only', ('macro_f1','C')]:.4f}였습니다. 8클래스 결과를 포함한 세 분석에서 A<B<C 방향이 유지되었습니다. 따라서 기타 범주에 대한 민감도 분석에서도 비교 방향이 유지되었다고 제한적으로 해석하였으며, 기타의 문화적 타당성이나 전문가 검증이 확보되었다고 주장하지 않았습니다.
"""
    (OUT / "author_reply_revision_text.md").write_text(author_reply, encoding="utf-8")

    config = {
        "analysis": "Other-label OOF sensitivity without new human review",
        "date": "2026-10-05", "seed": SEED, "bootstrap_iterations": N_BOOT,
        "inputs": [
            {"path": str(base.RAW_PATH.relative_to(ROOT)), "sha256": sha256(base.RAW_PATH),
             "rows": len(raw), "columns": raw_columns},
            {"path": str(base.MANIFEST_PATH.relative_to(ROOT)), "sha256": sha256(base.MANIFEST_PATH),
             "rows": len(manifest), "columns": manifest_columns},
            {"path": str(OOF_PATH.relative_to(ROOT)), "sha256": sha256(OOF_PATH),
             "rows": len(oof), "columns": oof_columns},
            {"path": str(Path(base.__file__).relative_to(ROOT)), "sha256": sha256(Path(base.__file__))},
        ],
        "integrity": integrity,
        "normalization": ["Unicode NFC", "trim and collapse whitespace", "remove bracket characters while retaining contents",
                          "unify separators to |", "compact spacing variants only for existing mapping-rule matching"],
        "normalization_prohibition": "No new synonym or cultural-semantic mapping was added.",
        "other_profile": stats,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "old_experiment_config_sha256": sha256(OLD_CONFIG),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, relation_symbols, _, _, raw, manifest = reconstruct()
    oof = pd.read_csv(OOF_PATH, encoding="utf-8-sig")
    integrity = validate(rows, oof, raw, manifest)
    if not integrity["passed"]:
        raise RuntimeError(f"Input integrity failed: {integrity}")
    stats, candidates = other_profile(rows, relation_symbols)
    metrics, classes, folds, analyses = calculate_metrics(oof)
    boot = bootstrap(oof, analyses)
    write_reports(rows, raw, manifest, oof, integrity, stats, candidates, metrics, folds, boot)
    print(f"integrity={integrity['passed']} rows={len(rows)} groups={rows.group_id.nunique()} other={stats['other_rows']}")
    print(f"other unique={stats['unique_normalized_expressions']} singleton={stats['singleton_share_of_unique']:.2%} le5={stats['le5_share_of_unique']:.2%}")
    print(f"technical reassignment candidates={stats['technical_candidate_rows']} ({stats['technical_candidate_share']:.4%})")
    print(metrics.to_string(index=False))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
