"""Integration tests for the Claims Triage FastAPI service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from claims_triage.api import SHA256_PREFIX_LENGTH, assign_triage_bucket, create_app
from claims_triage.generate import generate_claims
from claims_triage.train import (
    TriageThresholds,
    save_model_artifacts,
    train_and_evaluate,
)

SAMPLE_ROWS: int = 1_500
SAMPLE_SPLITS: int = 3
SAMPLE_SEED: int = 42
TEST_MIN_PR_AUC_LIFT: float = -1.0
THRESHOLD_EPSILON: float = 0.05
DEFAULT_STANDARD_QA: float = 0.35
DEFAULT_PRIORITY_ESCALATION: float = 0.80
SAMPLE_CLAIM_AMOUNT: float = 4200.0
SAMPLE_PROCESSING_AGE_HOURS: float = 12.0
SAMPLE_RECONCILIATION_VARIANCE: float = 5.0
SAMPLE_PRIOR_QA_FLAGS: int = 0
SAMPLE_DOCUMENT_COUNT: int = 2


def _valid_claim_payload() -> dict[str, Any]:
    """Return a valid 8-feature payload matching `ClaimInput`."""
    return {
        "claim_amount": SAMPLE_CLAIM_AMOUNT,
        "processing_age_hours": SAMPLE_PROCESSING_AGE_HOURS,
        "reconciliation_variance": SAMPLE_RECONCILIATION_VARIANCE,
        "prior_qa_flags": SAMPLE_PRIOR_QA_FLAGS,
        "document_count": SAMPLE_DOCUMENT_COUNT,
        "submission_channel": "portal",
        "auth_doc_verified": True,
        "domain_type": "healthcare_claim",
    }


@pytest.fixture
def trained_artifacts_dir(tmp_path: Path) -> Path:
    """Train a compact model and save verified artifacts in a temp directory."""
    output_dir = tmp_path / "artifacts"
    records = generate_claims(n_samples=SAMPLE_ROWS, seed=SAMPLE_SEED)
    pipeline, report = train_and_evaluate(
        records,
        seed=SAMPLE_SEED,
        n_splits=SAMPLE_SPLITS,
        min_pr_auc_lift=TEST_MIN_PR_AUC_LIFT,
    )
    save_model_artifacts(
        pipeline,
        report,
        output_dir,
        training_rows=len(records),
        seed=SAMPLE_SEED,
    )
    return output_dir


def test_assign_triage_bucket_boundaries() -> None:
    """Verify bucket assignment across sub-threshold, exact, and elevated values."""
    thresholds = TriageThresholds(
        standard_qa=DEFAULT_STANDARD_QA,
        priority_escalation=DEFAULT_PRIORITY_ESCALATION,
    )

    assert (
        assign_triage_bucket(thresholds.standard_qa - THRESHOLD_EPSILON, thresholds)
        == "AUTO_PROCESS"
    )
    assert assign_triage_bucket(thresholds.standard_qa, thresholds) == "STANDARD_QA"
    assert (
        assign_triage_bucket(
            thresholds.standard_qa + THRESHOLD_EPSILON,
            thresholds,
        )
        == "STANDARD_QA"
    )
    assert (
        assign_triage_bucket(
            thresholds.priority_escalation - THRESHOLD_EPSILON,
            thresholds,
        )
        == "STANDARD_QA"
    )
    assert (
        assign_triage_bucket(thresholds.priority_escalation, thresholds)
        == "PRIORITY_ESCALATION"
    )
    assert (
        assign_triage_bucket(
            thresholds.priority_escalation + THRESHOLD_EPSILON,
            thresholds,
        )
        == "PRIORITY_ESCALATION"
    )


def test_health_and_predict_with_verified_artifact(
    trained_artifacts_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify `/health`, `/predict`, and structured prediction logging when ready."""
    app = create_app(trained_artifacts_dir)

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == status.HTTP_200_OK
        health_data = health_response.json()
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True
        assert isinstance(health_data["model_version"], str)
        assert len(health_data["model_sha256_prefix"]) == SHA256_PREFIX_LENGTH

        with caplog.at_level(logging.INFO):
            predict_response = client.post("/predict", json=_valid_claim_payload())

        assert predict_response.status_code == status.HTTP_200_OK
        pred_data = predict_response.json()
        assert 0.0 <= pred_data["exception_probability"] <= 1.0
        assert pred_data["triage_bucket"] in {
            "AUTO_PROCESS",
            "STANDARD_QA",
            "PRIORITY_ESCALATION",
        }
        assert pred_data["model_version"] == health_data["model_version"]
        assert (
            0.0
            < pred_data["standard_qa_threshold"]
            < pred_data["priority_escalation_threshold"]
            < 1.0
        )

    logged_events = [
        json.loads(record.message)
        for record in caplog.records
        if record.message.startswith("{")
    ]
    prediction_logs = [
        event for event in logged_events if event.get("event") == "prediction_served"
    ]
    assert len(prediction_logs) == 1
    assert prediction_logs[0]["model_version"] == health_data["model_version"]
    assert prediction_logs[0]["triage_bucket"] == pred_data["triage_bucket"]
    assert "latency_ms" in prediction_logs[0]


def test_predict_rejects_pii_and_target_label_with_422(
    trained_artifacts_dir: Path,
) -> None:
    """Verify `/predict` rejects extra PII attributes and leaked target labels."""
    app = create_app(trained_artifacts_dir)

    with TestClient(app) as client:
        pii_payload = {**_valid_claim_payload(), "customer_email": "user@example.com"}
        pii_response = client.post("/predict", json=pii_payload)
        assert pii_response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        target_payload = {**_valid_claim_payload(), "is_exception": 1}
        target_response = client.post("/predict", json=target_payload)
        assert target_response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_missing_artifact_returns_503_not_ready(tmp_path: Path) -> None:
    """Verify missing artifacts surface 503 on both `/health` and `/predict`."""
    missing_dir = tmp_path / "missing_artifacts"
    app = create_app(missing_dir)

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        health_data = health_response.json()
        assert health_data["status"] == "not_ready"
        assert health_data["model_loaded"] is False

        predict_response = client.post("/predict", json=_valid_claim_payload())
        assert predict_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_tampered_artifact_fails_sha256_and_blocks_serving(
    trained_artifacts_dir: Path,
) -> None:
    """Verify corrupted model binaries fail SHA-256 verification and stay unserved."""
    model_path = trained_artifacts_dir / "model.joblib"
    with model_path.open("ab") as handle:
        handle.write(b"x")

    app = create_app(trained_artifacts_dir)

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert health_response.json()["model_loaded"] is False

        predict_response = client.post("/predict", json=_valid_claim_payload())
        assert predict_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
