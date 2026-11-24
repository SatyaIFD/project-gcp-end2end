from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from claims_triage.features import build_preprocessor, records_to_feature_matrix
from claims_triage.schemas import ClaimRecord

DEFAULT_SEED: int = 42
DEFAULT_CV_SPLITS: int = 5
MIN_CV_SPLITS: int = 2
BASELINE_MAX_ITER: int = 1000
CANDIDATE_MAX_DEPTH: int = 4
CANDIDATE_LEARNING_RATE: float = 0.05
CANDIDATE_MAX_ITER: int = 200
CANDIDATE_L2_REG: float = 1.0

MIN_METRIC: float = 0.0
MAX_METRIC: float = 1.0
MIN_LIFT: float = -1.0

TEST_SPLIT_SIZE: float = 0.20
VAL_SPLIT_RATIO: float = 0.25

TARGET_QA_RECALL: float = 0.80
MIN_PRIORITY_PRECISION: float = 0.50
MIN_PR_AUC_LIFT: float = 0.0
MIN_THRESHOLD_BOUND: float = 0.01
MAX_STANDARD_QA_BOUND: float = 0.85
MAX_PRIORITY_BOUND: float = 0.99
DEFAULT_STANDARD_QA_FALLBACK: float = 0.20
DEFAULT_PRIORITY_FALLBACK: float = 0.70
PRIORITY_FALLBACK_MARGIN: float = 0.10


class QualityGateError(RuntimeError):
    """Raised when the candidate model fails to pass promotion quality gates."""


class TriageThresholds(BaseModel):
    """Probability thresholds separating auto-processed, QA, and priority claims."""

    model_config = ConfigDict(extra="forbid")

    standard_qa: float = Field(gt=MIN_METRIC, lt=MAX_METRIC)
    priority_escalation: float = Field(gt=MIN_METRIC, lt=MAX_METRIC)

    @model_validator(mode="after")
    def validate_ordering(self) -> Self:
        if self.priority_escalation <= self.standard_qa:
            msg = (
                "priority_escalation must be strictly greater than standard_qa: "
                f"got priority_escalation={self.priority_escalation} vs "
                f"standard_qa={self.standard_qa}"
            )
            raise ValueError(msg)
        return self


class EvaluationReport(BaseModel):
    """Cross-validation, holdout test metrics, and gate outcome for a run."""

    model_config = ConfigDict(extra="forbid")

    baseline_cv_pr_auc: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    candidate_cv_pr_auc: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    pr_auc_lift: float = Field(ge=MIN_LIFT, le=MAX_METRIC)
    test_pr_auc: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    test_roc_auc: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    test_recall_at_qa: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    test_precision_at_priority: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    thresholds: TriageThresholds
    passed_quality_gate: bool


def build_baseline_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    """Build a leakage-free balanced logistic regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=BASELINE_MAX_ITER,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_candidate_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    """Build a leakage-free histogram gradient boosting candidate pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=CANDIDATE_MAX_DEPTH,
                    learning_rate=CANDIDATE_LEARNING_RATE,
                    max_iter=CANDIDATE_MAX_ITER,
                    l2_regularization=CANDIDATE_L2_REG,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_cv_pr_auc(
    pipeline: Pipeline,
    x_data: NDArray[np.object_],
    y_data: NDArray[np.int64],
    *,
    n_splits: int = DEFAULT_CV_SPLITS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Compute mean stratified cross-validation PR-AUC (average precision)."""
    if n_splits < MIN_CV_SPLITS:
        msg = f"n_splits must be at least {MIN_CV_SPLITS}, got {n_splits}"
        raise ValueError(msg)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(
        pipeline,
        x_data,
        y_data,
        cv=splitter,
        scoring="average_precision",
    )
    return float(np.mean(scores))


def calibrate_triage_thresholds(
    y_val: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    target_recall: float = TARGET_QA_RECALL,
    min_priority_precision: float = MIN_PRIORITY_PRECISION,
) -> TriageThresholds:
    """Calibrate ordered triage thresholds on validation probabilities."""
    if len(y_val) == 0 or len(y_prob) == 0 or len(y_val) != len(y_prob):
        msg = (
            "y_val and y_prob must be non-empty and have matching lengths: "
            f"got len(y_val)={len(y_val)}, len(y_prob)={len(y_prob)}"
        )
        raise ValueError(msg)

    unique_classes = np.unique(y_val)
    if len(unique_classes) < MIN_CV_SPLITS:
        msg = f"y_val must contain both classes 0 and 1, got {unique_classes.tolist()}"
        raise ValueError(msg)

    _precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    precisions_aligned = _precisions[:-1]
    recalls_aligned = recalls[:-1]

    qa_mask = (
        (thresholds >= MIN_THRESHOLD_BOUND)
        & (thresholds <= MAX_STANDARD_QA_BOUND)
        & (recalls_aligned >= target_recall)
    )
    if bool(np.any(qa_mask)):
        standard_qa = float(np.max(thresholds[qa_mask]))
    else:
        standard_qa = DEFAULT_STANDARD_QA_FALLBACK

    priority_mask = (
        (thresholds > standard_qa)
        & (thresholds <= MAX_PRIORITY_BOUND)
        & (precisions_aligned >= min_priority_precision)
    )
    if bool(np.any(priority_mask)):
        priority_escalation = float(np.min(thresholds[priority_mask]))
    else:
        priority_escalation = min(
            MAX_PRIORITY_BOUND,
            max(standard_qa + PRIORITY_FALLBACK_MARGIN, DEFAULT_PRIORITY_FALLBACK),
        )

    return TriageThresholds(
        standard_qa=standard_qa,
        priority_escalation=priority_escalation,
    )


def train_and_evaluate(
    records: Sequence[ClaimRecord],
    *,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_CV_SPLITS,
    min_pr_auc_lift: float = MIN_PR_AUC_LIFT,
    enforce_gate: bool = True,
) -> tuple[Pipeline, EvaluationReport]:
    """Run stratified CV gate, calibrate thresholds, and evaluate on holdout."""
    x_all, y_all = records_to_feature_matrix(records)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_all,
        y_all,
        test_size=TEST_SPLIT_SIZE,
        stratify=y_all,
        random_state=seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VAL_SPLIT_RATIO,
        stratify=y_train_val,
        random_state=seed,
    )

    baseline_pipeline = build_baseline_pipeline(seed=seed)
    candidate_pipeline = build_candidate_pipeline(seed=seed)

    baseline_cv_pr_auc = evaluate_cv_pr_auc(
        baseline_pipeline,
        x_train,
        y_train,
        n_splits=n_splits,
        seed=seed,
    )
    candidate_cv_pr_auc = evaluate_cv_pr_auc(
        candidate_pipeline,
        x_train,
        y_train,
        n_splits=n_splits,
        seed=seed,
    )

    pr_auc_lift = candidate_cv_pr_auc - baseline_cv_pr_auc
    passed_quality_gate = pr_auc_lift >= min_pr_auc_lift

    if enforce_gate and not passed_quality_gate:
        msg = (
            "Candidate model failed PR-AUC quality gate: "
            f"candidate_cv_pr_auc={candidate_cv_pr_auc:.4f}, "
            f"baseline_cv_pr_auc={baseline_cv_pr_auc:.4f}, "
            f"lift={pr_auc_lift:.4f} < required_lift={min_pr_auc_lift:.4f}"
        )
        raise QualityGateError(msg)

    candidate_pipeline.fit(x_train, y_train)

    val_probs = np.asarray(
        candidate_pipeline.predict_proba(x_val)[:, 1],
        dtype=np.float64,
    )
    thresholds = calibrate_triage_thresholds(y_val, val_probs)

    test_probs = np.asarray(
        candidate_pipeline.predict_proba(x_test)[:, 1],
        dtype=np.float64,
    )
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))

    qa_preds = (test_probs >= thresholds.standard_qa).astype(np.int64)
    test_recall_at_qa = float(
        recall_score(y_test, qa_preds, zero_division=0.0),
    )

    priority_preds = (test_probs >= thresholds.priority_escalation).astype(np.int64)
    test_precision_at_priority = float(
        precision_score(y_test, priority_preds, zero_division=0.0),
    )

    report = EvaluationReport(
        baseline_cv_pr_auc=baseline_cv_pr_auc,
        candidate_cv_pr_auc=candidate_cv_pr_auc,
        pr_auc_lift=pr_auc_lift,
        test_pr_auc=test_pr_auc,
        test_roc_auc=test_roc_auc,
        test_recall_at_qa=test_recall_at_qa,
        test_precision_at_priority=test_precision_at_priority,
        thresholds=thresholds,
        passed_quality_gate=passed_quality_gate,
    )
    return candidate_pipeline, report
