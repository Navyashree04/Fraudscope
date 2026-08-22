"""
Regression guard on model metrics.

This does NOT assert state-of-the-art performance -- it asserts that the
currently saved evaluation results (computed from the actual trained
artifact on its actual held-out split, never fabricated) clear a sane
floor, so a future change can't silently break the model without a test
failing.

Requires `python -m src.train` and `python -m src.evaluate` to have been
run first so reports/evaluation_report.md and models/metadata.json exist.
"""

import json

import pytest

from src.evaluate import evaluate
from src.config import settings


@pytest.fixture(scope="module")
def eval_results():
    return evaluate()


def test_metadata_labels_dataset_source(eval_results):
    with open(settings.model_metadata_path) as f:
        metadata = json.load(f)
    assert "dataset_source" in metadata
    assert "dataset_is_synthetic" in metadata


def test_roc_auc_clears_floor(eval_results):
    assert eval_results["roc_auc"] > 0.90


def test_pr_auc_clears_floor(eval_results):
    # PR-AUC is the more meaningful metric under severe class imbalance;
    # floor is intentionally modest since it's a regression guard, not a
    # claim of state-of-the-art performance.
    assert eval_results["pr_auc"] > 0.50


def test_precision_and_recall_are_sane_probabilities(eval_results):
    assert 0.0 <= eval_results["precision_at_0.5"] <= 1.0
    assert 0.0 <= eval_results["recall_at_0.5"] <= 1.0
