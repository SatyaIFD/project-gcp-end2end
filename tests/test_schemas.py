"""Unit tests verifying `ClaimRecord` schema contracts and zero-PII guardrails."""

from typing import Any

import pytest
from pydantic import ValidationError

from claims_triage.schemas import (
    MAX_CLAIM_AMOUNT,
    MAX_PROCESSING_AGE_HOURS,
    ClaimInput,
    ClaimRecord,
    HealthResponse,
    PredictionResponse,
)

EXPECTED_CLAIM_AMOUNT: float = 2500.0
EXPECTED_EXCEPTION_PROBABILITY: float = 0.42


def _valid_input_payload() -> dict[str, Any]:
    """Return a baseline dictionary satisfying all `ClaimInput` constraints."""
    return {
        "claim_amount": EXPECTED_CLAIM_AMOUNT,
        "processing_age_hours": 18.5,
        "reconciliation_variance": 12.0,
        "prior_qa_flags": 1,
        "document_count": 3,
        "submission_channel": "portal",
        "auth_doc_verified": True,
        "domain_type": "healthcare_claim",
    }


def _valid_payload() -> dict[str, Any]:
    """Return a baseline dictionary satisfying all `ClaimRecord` constraints."""
    return {
        **_valid_input_payload(),
        "is_exception": 0,
    }


def test_claim_record_happy_path() -> None:
    """Verify a valid payload instantiates cleanly and preserves exact values."""
    payload = _valid_payload()
    record = ClaimRecord(**payload)

    assert record.claim_amount == payload["claim_amount"]
    assert record.processing_age_hours == payload["processing_age_hours"]
    assert record.reconciliation_variance == payload["reconciliation_variance"]
    assert record.prior_qa_flags == payload["prior_qa_flags"]
    assert record.document_count == payload["document_count"]
    assert record.submission_channel == payload["submission_channel"]
    assert record.auth_doc_verified is True
    assert record.domain_type == payload["domain_type"]
    assert record.is_exception == payload["is_exception"]


def test_claim_record_numeric_boundaries() -> None:
    """Verify out-of-range numeric signals raise `ValidationError`."""
    # `claim_amount` enforces strict positivity (`gt=0.0`)
    zero_amount = _valid_payload()
    zero_amount["claim_amount"] = 0.0
    with pytest.raises(ValidationError):
        ClaimRecord(**zero_amount)

    # `claim_amount` cannot exceed operational ceiling
    over_max_amount = _valid_payload()
    over_max_amount["claim_amount"] = MAX_CLAIM_AMOUNT + 0.01
    with pytest.raises(ValidationError):
        ClaimRecord(**over_max_amount)

    # `processing_age_hours` cannot exceed the 30-day SLA ceiling
    excessive_age = _valid_payload()
    excessive_age["processing_age_hours"] = MAX_PROCESSING_AGE_HOURS + 1.0
    with pytest.raises(ValidationError):
        ClaimRecord(**excessive_age)


def test_claim_record_invalid_categories() -> None:
    """Verify unapproved categorical values are rejected."""
    bad_channel = _valid_payload()
    bad_channel["submission_channel"] = "fax"
    with pytest.raises(ValidationError):
        ClaimRecord(**bad_channel)

    bad_domain = _valid_payload()
    bad_domain["domain_type"] = "auto_insurance"
    with pytest.raises(ValidationError):
        ClaimRecord(**bad_domain)


def test_claim_record_rejects_extra_pii_fields() -> None:
    """Verify `extra='forbid'` blocks unexpected attributes to prevent PII leakage."""
    payload_with_pii = _valid_payload()
    payload_with_pii["customer_email"] = "user@example.com"

    with pytest.raises(ValidationError):
        ClaimRecord(**payload_with_pii)


def test_claim_input_accepts_valid_features_and_rejects_target_or_pii() -> None:
    """Verify `ClaimInput` accepts 8 features and rejects target or PII fields."""
    valid_input = ClaimInput(**_valid_input_payload())
    assert valid_input.claim_amount == EXPECTED_CLAIM_AMOUNT
    assert valid_input.domain_type == "healthcare_claim"

    leaked_target = {**_valid_input_payload(), "is_exception": 1}
    with pytest.raises(ValidationError):
        ClaimInput(**leaked_target)

    with_ssn = {**_valid_input_payload(), "ssn": "123-45-6789"}
    with pytest.raises(ValidationError):
        ClaimInput(**with_ssn)


def test_prediction_and_health_response_validation() -> None:
    """Verify serving response contracts accept valid fields and reject invalid ones."""
    prediction = PredictionResponse(
        exception_probability=EXPECTED_EXCEPTION_PROBABILITY,
        triage_bucket="STANDARD_QA",
        model_version="v1.0.0",
        standard_qa_threshold=0.35,
        priority_escalation_threshold=0.80,
    )
    assert prediction.triage_bucket == "STANDARD_QA"
    assert prediction.exception_probability == EXPECTED_EXCEPTION_PROBABILITY

    health = HealthResponse(
        status="ok",
        model_loaded=True,
        model_version="v1.0.0",
        model_sha256_prefix="0123456789ab",
    )
    assert health.status == "ok"
    assert health.model_sha256_prefix == "0123456789ab"

    with pytest.raises(ValidationError):
        PredictionResponse(
            exception_probability=1.2,
            triage_bucket="PRIORITY_ESCALATION",
            model_version="v1.0.0",
            standard_qa_threshold=0.35,
            priority_escalation_threshold=0.80,
        )

    with pytest.raises(ValidationError):
        PredictionResponse(
            exception_probability=0.50,
            triage_bucket="INVALID_BUCKET",  # type: ignore[arg-type]
            model_version="v1.0.0",
            standard_qa_threshold=0.35,
            priority_escalation_threshold=0.80,
        )
