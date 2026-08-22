"""
Evaluation pipeline for FraudScope.

Computes precision/recall/F1/ROC-AUC/PR-AUC on the held-out test split
saved by src/train.py, and writes an honest Markdown report. All numbers
here come from the actual trained artifact on the actual test split --
nothing is fabricated.

Run: python -m src.evaluate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import settings

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraudscope.evaluate")

THRESHOLDS = [0.3, 0.5, 0.7]


def load_test_split() -> tuple[np.ndarray, np.ndarray]:
    split_path = Path(settings.model_path).parent / "test_split.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found -- run `python -m src.train` first to "
            "produce the model artifact and its held-out test split."
        )
    data = np.load(split_path)
    return data["X_test"], data["y_test"]


def evaluate() -> dict:
    with open(settings.model_metadata_path) as f:
        metadata = json.load(f)

    model = joblib.load(settings.model_path)
    X_test, y_test = load_test_split()

    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)  # a.k.a. average precision

    threshold_rows = []
    for t in THRESHOLDS:
        y_pred = (y_proba >= t).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        threshold_rows.append({
            "threshold": t, "precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    default_row = next(r for r in threshold_rows if r["threshold"] == 0.5)

    results = {
        "dataset_source": metadata["dataset_source"],
        "dataset_is_synthetic": metadata["dataset_is_synthetic"],
        "model_version": metadata["model_version"],
        "n_test_rows": int(len(y_test)),
        "n_test_fraud": int(y_test.sum()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_at_0.5": default_row["precision"],
        "recall_at_0.5": default_row["recall"],
        "f1_at_0.5": default_row["f1"],
        "thresholds": threshold_rows,
    }

    _write_report(results)
    logger.info("Wrote evaluation report to %s", settings.evaluation_report_path)
    return results


def _write_report(results: dict) -> None:
    lines = []
    lines.append("# FraudScope Evaluation Report")
    lines.append("")
    lines.append(
        "> **Dataset note:** results below are computed on the "
        f"**{'synthetic-for-demo' if results['dataset_is_synthetic'] else 'public'}** "
        f"dataset (`{results['dataset_source']}`), never real bank or customer data. "
        "All numbers are computed directly from the trained model artifact on a "
        "held-out, stratified test split -- nothing here is fabricated."
    )
    lines.append("")
    lines.append(f"- Model version: `{results['model_version']}`")
    lines.append(f"- Test set size: {results['n_test_rows']} rows, "
                  f"{results['n_test_fraud']} fraud")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| ROC-AUC | {results['roc_auc']:.4f} |")
    lines.append(f"| PR-AUC (average precision) | {results['pr_auc']:.4f} |")
    lines.append(f"| Precision @ 0.5 | {results['precision_at_0.5']:.4f} |")
    lines.append(f"| Recall @ 0.5 | {results['recall_at_0.5']:.4f} |")
    lines.append(f"| F1 @ 0.5 | {results['f1_at_0.5']:.4f} |")
    lines.append("")
    lines.append(
        "**Why PR-AUC matters more than ROC-AUC here:** with fraud at roughly "
        "0.17% of transactions, a model that predicts \"legitimate\" for almost "
        "everything can still post a high ROC-AUC and >99.8% raw accuracy, "
        "because ROC-AUC is dominated by the abundant negative class. PR-AUC "
        "(average precision) focuses on precision/recall trade-offs on the "
        "positive (fraud) class specifically, which is what actually matters "
        "for a fraud-ops team deciding how many alerts to review."
    )
    lines.append("")
    lines.append("## Precision/recall at alternative thresholds")
    lines.append("")
    lines.append(
        "In a real fraud-ops setting, the decision threshold is tuned against "
        "the business cost of false positives (annoyed customers, manual "
        "review load) versus false negatives (fraud losses) -- it is not "
        "simply fixed at 0.5."
    )
    lines.append("")
    lines.append("| Threshold | Precision | Recall | F1 | TP | FP | FN | TN |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in results["thresholds"]:
        lines.append(
            f"| {row['threshold']:.1f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {row['tn']} |"
        )
    lines.append("")
    lines.append(
        "A lower threshold catches more fraud (higher recall) at the cost of "
        "more false alarms (lower precision); a higher threshold does the "
        "opposite. Which one is 'right' depends on the relative cost of a "
        "missed fraud versus an analyst wasting time on a false alarm --  "
        "a business decision, not a purely technical one."
    )
    lines.append("")

    Path(settings.evaluation_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(settings.evaluation_report_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    res = evaluate()
    print(json.dumps({k: v for k, v in res.items() if k != "thresholds"}, indent=2))
