#!/usr/bin/env python3
"""Leakage-controlled graph-embedding extension of the v4 grouped ablation.

The existing A/B/C OOF predictions are preserved.  This script adds one
pre-specified condition, D: the exact C text input plus a 64-dimensional
spectral embedding of the training-fold motif/subject co-occurrence graph.
No validation artwork or label edge is used to fit the embedding.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import coo_matrix, csr_matrix, diags, hstack
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import abc_ablation as base  # noqa: E402


OLD_DIR = ROOT / "results" / "generated" / "abc"
OUT_DIR = ROOT / "results" / "generated" / "graph_embedding"
OLD_PRED = OLD_DIR / "grouped_ablation_predictions.csv"
AGENT_CSV = ROOT / "results" / "generated" / "agent" / "symbolic_agent_report.csv"
SEED = 42
BOOTSTRAP_SEED = 20261003
N_COMPONENTS = 64
CONDITION_D = "kg_context_graph_embedding"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typed_nodes(group_id: str, group_motifs: dict[str, set[str]], group_subjects: dict[str, set[str]]) -> list[str]:
    return ([f"m::{x}" for x in sorted(group_motifs[group_id])] +
            [f"s::{x}" for x in sorted(group_subjects[group_id])])


def fit_graph_embedding(train_groups: set[str], group_motifs, group_subjects):
    """Fit a normalized-adjacency spectral embedding on training groups only."""
    edge_weights: Counter[tuple[str, str]] = Counter()
    all_nodes: set[str] = set()
    for group_id in sorted(train_groups):
        nodes = typed_nodes(group_id, group_motifs, group_subjects)
        all_nodes.update(nodes)
        motifs = [n for n in nodes if n.startswith("m::")]
        subjects = [n for n in nodes if n.startswith("s::")]
        # Motif co-occurrence and motif-subject relations encode graph structure.
        for i, left in enumerate(motifs):
            for right in motifs[i + 1:]:
                edge_weights[tuple(sorted((left, right)))] += 1
        for left in motifs:
            for right in subjects:
                edge_weights[tuple(sorted((left, right)))] += 1

    node_list = sorted(all_nodes)
    node_index = {node: i for i, node in enumerate(node_list)}
    rows, cols, values = [], [], []
    for (left, right), weight in edge_weights.items():
        i, j = node_index[left], node_index[right]
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((float(weight), float(weight)))
    n = len(node_list)
    adjacency = coo_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degrees)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    normalized_adjacency = diags(inv_sqrt) @ adjacency @ diags(inv_sqrt)
    dimensions = min(N_COMPONENTS, max(1, n - 1))
    svd = TruncatedSVD(n_components=dimensions, algorithm="randomized", n_iter=7, random_state=SEED)
    embedding = normalize(svd.fit_transform(normalized_adjacency), norm="l2", axis=1)
    if dimensions < N_COMPONENTS:
        embedding = np.pad(embedding, ((0, 0), (0, N_COMPONENTS - dimensions)))
    diagnostics = {
        "n_nodes": n,
        "n_undirected_edges": len(edge_weights),
        "n_components": dimensions,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }
    return node_index, embedding, diagnostics


def row_graph_matrix(frame: pd.DataFrame, node_index, embedding, group_motifs, group_subjects):
    result = np.zeros((len(frame), N_COMPONENTS), dtype=np.float64)
    coverage = np.zeros(len(frame), dtype=np.int32)
    for row_i, row in enumerate(frame.itertuples(index=False)):
        indices = [node_index[node] for node in typed_nodes(row.group_id, group_motifs, group_subjects)
                   if node in node_index]
        if indices:
            result[row_i] = embedding[indices].mean(axis=0)
            coverage[row_i] = len(indices)
    result = normalize(result, norm="l2", axis=1)
    return csr_matrix(result), coverage


def make_kg_texts(frame, group_motifs, group_subjects, motif_symbol_groups):
    return base._train_texts(frame, "kg_context", group_motifs, group_subjects, motif_symbol_groups)


def load_and_validate_old(samples: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_PRED, encoding="utf-8-sig")
    keys = ["group_id", "motif", "y_true"]
    if len(old) != len(samples):
        raise AssertionError(f"Stored OOF rows={len(old)} but reconstructed rows={len(samples)}")
    left = old[keys].sort_values(keys).reset_index(drop=True)
    right = samples.rename(columns={"label": "y_true"})[keys].sort_values(keys).reset_index(drop=True)
    if not left.equals(right):
        mismatch = pd.concat([left, right], axis=1).query("group_id != group_id")
        raise AssertionError(f"Stored predictions do not match reconstructed relation rows: {len(mismatch)}")
    if old.groupby("group_id")["fold"].nunique().max() != 1:
        raise AssertionError("An artwork group crosses stored folds.")
    return old


def per_fold_and_overall(pred: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    mapping = {
        "motif_only": "pred_motif_only",
        "metadata_context": "pred_metadata_context",
        "kg_context": "pred_kg_context",
        CONDITION_D: f"pred_{CONDITION_D}",
    }
    for condition, column in mapping.items():
        for fold, part in pred.groupby("fold", sort=True):
            rows.append({
                "scope": "fold", "fold": int(fold), "condition": condition, "n": len(part),
                "f1_macro": f1_score(part.y_true, part[column], labels=labels, average="macro", zero_division=0),
                "f1_micro": f1_score(part.y_true, part[column], labels=labels, average="micro", zero_division=0),
                "accuracy": accuracy_score(part.y_true, part[column]),
            })
        rows.append({
            "scope": "pooled_oof", "fold": 0, "condition": condition, "n": len(pred),
            "f1_macro": f1_score(pred.y_true, pred[column], labels=labels, average="macro", zero_division=0),
            "f1_micro": f1_score(pred.y_true, pred[column], labels=labels, average="micro", zero_division=0),
            "accuracy": accuracy_score(pred.y_true, pred[column]),
        })
    return pd.DataFrame(rows)


def paired_group_bootstrap(pred: pd.DataFrame, labels: list[str], iterations: int = 2000) -> pd.DataFrame:
    groups = sorted(pred.group_id.unique())
    indices = {g: pred.index[pred.group_id == g].to_numpy() for g in groups}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    y = pred.y_true.to_numpy()
    c = pred.pred_kg_context.to_numpy()
    d = pred[f"pred_{CONDITION_D}"].to_numpy()
    for b in range(iterations):
        draw = rng.choice(groups, size=len(groups), replace=True)
        sampled = np.concatenate([indices[g] for g in draw])
        for average in ("macro", "micro"):
            delta = (f1_score(y[sampled], d[sampled], labels=labels, average=average, zero_division=0) -
                     f1_score(y[sampled], c[sampled], labels=labels, average=average, zero_division=0))
            results.append({"iteration": b + 1, "metric": f"f1_{average}", "delta_D_minus_C": delta})
    return pd.DataFrame(results)


def threshold_sensitivity() -> tuple[pd.DataFrame, dict]:
    agent = pd.read_csv(AGENT_CSV, encoding="utf-8-sig").fillna("")
    supported_grid = [0.70, 0.74, 0.78, 0.82, 0.86]
    plausible_grid = [0.45, 0.50, 0.55, 0.60, 0.65]

    def split(value):
        return {part.strip() for part in str(value).split(",") if part.strip()}

    def classify(row, high, low):
        sources = split(row.evidence_sources)
        gold = split(row.gold_symbols)
        objects = split(row.objects)
        if row.symbol in gold or "keyword" in sources:
            status = "supported"
        elif float(row.confidence) >= high and len(sources) >= 2:
            status = "supported"
        elif float(row.confidence) >= low:
            status = "plausible"
        else:
            status = "needs_review"
        if row.symbol in objects and "keyword" not in sources:
            status = "needs_review"
        return status

    rows = []
    baseline_recalc = None
    for high in supported_grid:
        for low in plausible_grid:
            statuses = [classify(row, high, low) for row in agent.itertuples(index=False)]
            counts = Counter(statuses)
            changed = sum(a != b for a, b in zip(statuses, agent.status))
            rows.append({
                "supported_threshold": high, "plausible_threshold": low,
                "supported_n": counts["supported"], "plausible_n": counts["plausible"],
                "needs_review_n": counts["needs_review"], "total_claims": len(agent),
                "changed_vs_stored_baseline_n": changed,
            })
            if high == 0.78 and low == 0.55:
                baseline_recalc = {"counts": dict(counts), "changed_vs_stored": changed}
    return pd.DataFrame(rows), {"n_claims": len(agent), "baseline_recalculation": baseline_recalc}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, group_motifs, group_subjects = base.read_samples()
    old = load_and_validate_old(samples)
    fold_by_group = old.groupby("group_id")["fold"].first().to_dict()
    labels = sorted(samples.label.unique())
    d_rows = []
    fold_graph_diagnostics = []

    for fold in sorted(old.fold.unique()):
        test_groups = {g for g, f in fold_by_group.items() if f == fold}
        train = samples[~samples.group_id.isin(test_groups)].copy()
        test = samples[samples.group_id.isin(test_groups)].copy()
        train_groups = set(train.group_id)
        if train_groups & test_groups:
            raise AssertionError("Artwork group overlap in graph-embedding fold.")

        motif_symbol_groups = defaultdict(lambda: defaultdict(set))
        for row in train.itertuples(index=False):
            motif_symbol_groups[row.motif][row.label].add(row.group_id)
        train_text = make_kg_texts(train, group_motifs, group_subjects, motif_symbol_groups)
        test_text = make_kg_texts(test, group_motifs, group_subjects, motif_symbol_groups)

        node_index, node_embedding, diagnostics = fit_graph_embedding(train_groups, group_motifs, group_subjects)
        train_graph, train_coverage = row_graph_matrix(train, node_index, node_embedding, group_motifs, group_subjects)
        test_graph, test_coverage = row_graph_matrix(test, node_index, node_embedding, group_motifs, group_subjects)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
        x_train_text = vectorizer.fit_transform(train_text)
        x_test_text = vectorizer.transform(test_text)
        x_train = hstack([x_train_text, train_graph], format="csr")
        x_test = hstack([x_test_text, test_graph], format="csr")
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                        class_weight="balanced", random_state=SEED)
        classifier.fit(x_train, train.label.to_numpy())
        predicted = classifier.predict(x_test)
        probabilities = classifier.predict_proba(x_test)
        class_index = {label: i for i, label in enumerate(classifier.classes_)}
        diagnostics.update({
            "fold": int(fold), "n_train_rows": len(train), "n_test_rows": len(test),
            "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
            "test_row_graph_coverage": float(np.mean(test_coverage > 0)),
            "mean_test_nodes_used": float(test_coverage.mean()),
        })
        fold_graph_diagnostics.append(diagnostics)
        for i, row in enumerate(test.itertuples(index=False)):
            d_rows.append({
                "fold": int(fold), "group_id": row.group_id, "motif": row.motif, "y_true": row.label,
                f"pred_{CONDITION_D}": predicted[i],
                "d_pred_probability": float(probabilities[i, class_index[predicted[i]]]),
                "d_true_probability": float(probabilities[i, class_index[row.label]]),
                "graph_nodes_used": int(test_coverage[i]),
            })

    d_pred = pd.DataFrame(d_rows)
    keys = ["fold", "group_id", "motif", "y_true"]
    combined = old.merge(d_pred, on=keys, how="left", validate="one_to_one")
    if combined[f"pred_{CONDITION_D}"].isna().any():
        raise AssertionError("Missing D predictions after fold merge.")

    metrics = per_fold_and_overall(combined, labels)
    fold_summary = (metrics[metrics.scope == "fold"].groupby("condition")
                    .agg(f1_macro_mean=("f1_macro", "mean"), f1_macro_std=("f1_macro", "std"),
                         f1_micro_mean=("f1_micro", "mean"), accuracy_mean=("accuracy", "mean"))
                    .reset_index())
    bootstrap = paired_group_bootstrap(combined, labels)
    bootstrap_summary = (bootstrap.groupby("metric")["delta_D_minus_C"]
                         .agg(mean="mean", median="median",
                              ci95_low=lambda x: x.quantile(0.025),
                              ci95_high=lambda x: x.quantile(0.975)).reset_index())

    b_corrected = combined[(combined.pred_metadata_context != combined.y_true) &
                           (combined[f"pred_{CONDITION_D}"] == combined.y_true)].copy()
    b_harmed = combined[(combined.pred_metadata_context == combined.y_true) &
                        (combined[f"pred_{CONDITION_D}"] != combined.y_true)].copy()
    c_corrected = combined[(combined.pred_kg_context != combined.y_true) &
                           (combined[f"pred_{CONDITION_D}"] == combined.y_true)].copy()
    c_harmed = combined[(combined.pred_kg_context == combined.y_true) &
                        (combined[f"pred_{CONDITION_D}"] != combined.y_true)].copy()
    transitions = pd.DataFrame([
        {"comparison": "D_vs_B", "corrected_rows": len(b_corrected), "harmed_rows": len(b_harmed),
         "net_correct_rows": len(b_corrected) - len(b_harmed)},
        {"comparison": "D_vs_C", "corrected_rows": len(c_corrected), "harmed_rows": len(c_harmed),
         "net_correct_rows": len(c_corrected) - len(c_harmed)},
    ])
    cases = (b_corrected.sort_values(["y_true", "d_pred_probability", "group_id", "motif"],
                                     ascending=[True, False, True, True])
             .groupby("y_true", as_index=False).head(1))

    sensitivity, sensitivity_info = threshold_sensitivity()

    combined.to_csv(OUT_DIR / "oof_predictions_A_B_C_D.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "fold_and_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(OUT_DIR / "fold_metric_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_graph_diagnostics).to_csv(OUT_DIR / "graph_embedding_fold_diagnostics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUT_DIR / "paired_group_bootstrap_D_minus_C_2000.csv", index=False, encoding="utf-8-sig")
    bootstrap_summary.to_csv(OUT_DIR / "paired_group_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(OUT_DIR / "prediction_transition_summary.csv", index=False, encoding="utf-8-sig")
    b_corrected.to_csv(OUT_DIR / "all_B_wrong_D_correct_cases.csv", index=False, encoding="utf-8-sig")
    b_harmed.to_csv(OUT_DIR / "all_B_correct_D_wrong_cases.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "case_study_examples_one_per_true_class.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT_DIR / "agent_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")

    config = {
        "analysis": "v4 A/B/C preservation plus pre-specified D graph-embedding extension",
        "date": "2026-10-04", "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "n_relation_rows": len(samples), "n_artwork_groups": samples.group_id.nunique(),
        "n_classes": len(labels), "classes": labels, "fold_source": str(OLD_PRED.relative_to(ROOT)),
        "condition_D": "C text features plus a 64-d spectral embedding of the training-fold-only motif/subject graph",
        "graph_nodes": "typed motif and subject nodes",
        "graph_edges": "within-training-artwork motif co-occurrence and motif-subject edges; weighted by artwork count",
        "embedding": {"method": "TruncatedSVD of symmetrically normalized adjacency", "dimensions": N_COMPONENTS,
                      "n_iter": 7, "row_feature": "L2-normalized mean of known nodes in the artwork"},
        "leakage_control": "Each embedding and TF-IDF vocabulary is fit on outer-training artwork groups only; validation groups and all labels are excluded from graph fitting.",
        "classifier": {"type": "LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 2000,
                       "class_weight": "balanced", "random_state": SEED},
        "threshold_sensitivity": {"supported_grid": [0.70, 0.74, 0.78, 0.82, 0.86],
                                  "plausible_grid": [0.45, 0.50, 0.55, 0.60, 0.65],
                                  **sensitivity_info},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "hashes": {"raw": sha256(base.RAW_PATH), "manifest": sha256(base.MANIFEST_PATH),
                   "stored_oof": sha256(OLD_PRED), "agent_csv": sha256(AGENT_CSV),
                   "script": sha256(Path(__file__))},
        "selection_note": "No embedding dimension, classifier, threshold, or fold was selected by OOF performance.",
    }
    (OUT_DIR / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    pooled = metrics[metrics.scope == "pooled_oof"].set_index("condition")
    report = [
        "# v4 그래프 임베딩 및 에이전트 임계값 보완 분석", "",
        "## 설계", "",
        f"- v4와 동일한 {len(samples):,}개 관계 행, {samples.group_id.nunique():,}개 작품 그룹, 8개 배타적 상징 클래스 및 저장된 5개 fold를 유지했다.",
        "- D는 기존 C의 텍스트 입력에 학습 fold의 소재·화목 이종 그래프에서 구한 64차원 스펙트럴 임베딩을 결합했다.",
        "- 검증 작품은 그래프·TF-IDF 어휘 학습에서 제외했으며 그래프에는 정답 레이블 간선 자체를 넣지 않았다.",
        "- 차원(64), 분류기, seed 및 비교 절차는 결과 확인 전에 고정했고 OOF 점수로 설정을 재선택하지 않았다.", "",
        "## 정량 결과(전체 OOF)", "",
        "| 조건 | macro-F1 | micro-F1 | 정확도 |", "|---|---:|---:|---:|",
    ]
    names = {"motif_only": "A 소재", "metadata_context": "B 소재+일반 문맥",
             "kg_context": "C B+학습 fold KG 관계어", CONDITION_D: "D C+그래프 임베딩"}
    for condition in names:
        row = pooled.loc[condition]
        report.append(f"| {names[condition]} | {row.f1_macro:.4f} | {row.f1_micro:.4f} | {row.accuracy:.4f} |")
    report += ["", "## 대응 작품군 부트스트랩(D−C)", "",
               "동일 작품의 관계 행을 함께 재표집한 2,000회 결과이며, 현재 OOF 예측에 조건부인 표본 재표집 불확실성이다.", ""]
    for row in bootstrap_summary.itertuples(index=False):
        report.append(f"- {row.metric}: 평균 {row.mean:+.4f}, 95% 구간 [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]")
    report += ["", "## 예측 보정과 손상", ""]
    for row in transitions.itertuples(index=False):
        report.append(f"- {row.comparison}: 보정 {row.corrected_rows:,}행, 손상 {row.harmed_rows:,}행, 순변화 {row.net_correct_rows:+,}행")
    report += ["", "사례는 B 오분류를 D가 정정한 전체 목록을 먼저 보존한 뒤, 정답 클래스별 D 확률이 가장 큰 1건을 표본 사례로 제시했다. 이는 사례 설명용이며 일반 성능의 근거는 전체 OOF 지표와 부트스트랩이다.", "",
               "## 0.78·0.55 임계값 민감도", "",
               "0.78과 0.55는 외부 검증셋에서 최적화된 값이 아니라, 복수 근거를 요구하는 supported 단계와 단일·중간 근거를 검토하는 plausible 단계를 구분한 휴리스틱 운영점이다. 임계값은 후보 순위·precision@k·recall@k를 바꾸지 않고 검토 상태와 검토량만 바꾼다.",
               "민감도 표는 supported 0.70–0.86, plausible 0.45–0.65의 25개 조합에서 세 상태의 건수 변화를 모두 보고한다.", "",
               "## 해석 한계", "",
               "D−C가 양수여도 이는 본 자료와 고정 모형에서 소재·화목 공출현 구조 표현의 증분을 뜻한다. 다중 경로 추론이나 모든 그래프 학습법의 우월성을 입증하지 않는다. 반대로 구간이 0을 포함하거나 성능이 낮아지면 구조 임베딩 도입 자체가 자동으로 개선을 보장하지 않는다는 결과로 해석한다."]
    (OUT_DIR / "verification_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(fold_summary.to_string(index=False))
    print(bootstrap_summary.to_string(index=False))
    print(transitions.to_string(index=False))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
