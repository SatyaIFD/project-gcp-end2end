from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from claims_triage.features import records_to_feature_matrix
from claims_triage.generate import generate_claims
from claims_triage.train import (
    DEFAULT_SEED,
    MIN_LIFT,
    MIN_METRIC,
    QualityGateError,
    TriageThresholds,
    build_baseline_pipeline,
    build_candidate_pipeline,
    calibrate_triage_thresholds,
    evaluate_cv_pr_auc,
    train_and_evaluate,
)


def test_triage_thresholds_ordering_and_extra_forbid() -> None:
    valid_qa = 0.25
    valid_priority = 0.75
    thresholds = TriageThresholds(
        standard_qa=valid_qa,
        priority_escalation=valid_priority,
    )
    assert thresholds.standard_qa == valid_qa
    assert thresholds.priority_escalation == valid_priority

    equal_threshold = 0.6
    with pytest.raises(ValidationError):
        TriageThresholds(
            standard_qa=equal_threshold,
            priority_escalation=equal_threshold,
        )

    inverted_qa = 0.6
    inverted_priority = 0.4
    with pytest.raises(ValidationError):
        TriageThresholds(
            standard_qa=inverted_qa,
            priority_escalation=inverted_priority,
        )

    with pytest.raises(ValidationError):
        TriageThresholds.model_validate(
            {
                "standard_qa": valid_qa,
                "priority_escalation": valid_priority,
                "unexpected_field": 0.9,
            }
        )


def test_evaluate_cv_pr_auc_rejects_invalid_splits() -> None:
    sample_count = 20
    invalid_splits = 1
    records = generate_claims(n_samples=sample_count, seed=DEFAULT_SEED)
    x_data, y_data = records_to_feature_matrix(records)
    pipeline = build_baseline_pipeline(seed=DEFAULT_SEED)

    with pytest.raises(ValueError, match="n_splits"):
        evaluate_cv_pr_auc(
            pipeline,
            x_data,
            y_data,
            n_splits=invalid_splits,
            seed=DEFAULT_SEED,
        )


def test_baseline_and_candidate_cv_pr_auc_scores_realistic() -> None:
    sample_count = 1500
    cv_splits = 3
    records = generate_claims(n_samples=sample_count, seed=DEFAULT_SEED)
    x_data, y_data = records_to_feature_matrix(records)

    baseline_score = evaluate_cv_pr_auc(
        build_baseline_pipeline(seed=DEFAULT_SEED),
        x_data,
        y_data,
        n_splits=cv_splits,
        seed=DEFAULT_SEED,
    )
    candidate_score = evaluate_cv_pr_auc(
        build_candidate_pipeline(seed=DEFAULT_SEED),
        x_data,
        y_data,
        n_splits=cv_splits,
        seed=DEFAULT_SEED,
    )

    assert MIN_METRIC < baseline_score < 1.0
    assert MIN_METRIC < candidate_score < 1.0


def test_calibrate_triage_thresholds_valid_and_fallbacks() -> None:
    y_val = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_prob = np.asarray(
        [0.05, 0.12, 0.18, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 0.95],
        dtype=np.float64,
    )

    thresholds = calibrate_triage_thresholds(y_val, y_prob)
    assert MIN_METRIC < thresholds.standard_qa < thresholds.priority_escalation < 1.0

    impossible_precision = 1.01
    fallback_thresholds = calibrate_triage_thresholds(
        y_val,
        y_prob,
        min_priority_precision=impossible_precision,
    )
    assert (
        MIN_METRIC
        < fallback_thresholds.standard_qa
        < fallback_thresholds.priority_escalation
        < 1.0
    )

    with pytest.raises(ValueError, match="non-empty"):
        calibrate_triage_thresholds(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
        )

    single_class_y = np.asarray([0, 0, 0, 0], dtype=np.int64)
    single_class_prob = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    with pytest.raises(ValueError, match="both classes"):
        calibrate_triage_thresholds(single_class_y, single_class_prob)


def test_train_and_evaluate_returns_valid_report_and_fitted_pipeline() -> None:
    sample_count = 1500
    cv_splits = 3
    fresh_sample_count = 5
    fresh_seed = 99
    records = generate_claims(n_samples=sample_count, seed=DEFAULT_SEED)

    pipeline, report = train_and_evaluate(
        records,
        seed=DEFAULT_SEED,
        n_splits=cv_splits,
        min_pr_auc_lift=MIN_LIFT,
        enforce_gate=True,
    )

    assert report.passed_quality_gate is True
    assert MIN_METRIC < report.test_pr_auc < 1.0
    assert MIN_METRIC < report.test_roc_auc < 1.0
    assert (
        MIN_METRIC
        < report.thresholds.standard_qa
        < report.thresholds.priority_escalation
        < 1.0
    )

    fresh_records = generate_claims(n_samples=fresh_sample_count, seed=fresh_seed)
    x_fresh, _ = records_to_feature_matrix(fresh_records)
    probs = pipeline.predict_proba(x_fresh)
    assert probs.shape == (fresh_sample_count, 2)


def test_train_and_evaluate_quality_gate_blocks_unmet_lift() -> None:
    sample_count = 1500
    cv_splits = 3
    strict_lift = 0.95
    records = generate_claims(n_samples=sample_count, seed=DEFAULT_SEED)

    with pytest.raises(QualityGateError, match="quality gate"):
        train_and_evaluate(
            records,
            seed=DEFAULT_SEED,
            n_splits=cv_splits,
            min_pr_auc_lift=strict_lift,
            enforce_gate=True,
        )


def test_train_and_evaluate_enforce_gate_false_records_failure() -> None:
    sample_count = 1500
    cv_splits = 3
    strict_lift = 0.95
    records = generate_claims(n_samples=sample_count, seed=DEFAULT_SEED)

    pipeline, report = train_and_evaluate(
        records,
        seed=DEFAULT_SEED,
        n_splits=cv_splits,
        min_pr_auc_lift=strict_lift,
        enforce_gate=False,
    )

    assert report.passed_quality_gate is False
    assert pipeline is not None
