"""Pydantic data contracts and validation guardrails for operational claim records.

This module enforces schema integrity and zero-PII compliance aligned with
`docs/data-governance.md`. Every record ingested by the synthetic data generator,
training pipeline, or inference endpoint must pass these structural and domain
constraints before downstream processing.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Upper domain bounds derived from operational SLA and transaction limits.
# Defined as module-level constants to avoid magic numbers (Ruff PLR2004)
# and allow single-source updates across validation and synthetic generation.
MAX_CLAIM_AMOUNT: float = 1_000_000.0
MAX_PROCESSING_AGE_HOURS: float = 720.0  # 30-day maximum SLA window
MAX_RECONCILIATION_VARIANCE: float = 1_000_000.0
MAX_PRIOR_QA_FLAGS: int = 50
MAX_DOCUMENT_COUNT: int = 50

# Probability bounds for serving responses and triage thresholds.
MIN_PROBABILITY: float = 0.0
MAX_PROBABILITY: float = 1.0

# Controlled vocabularies matching `docs/data-governance.md` Section 2.
SubmissionChannel = Literal["portal", "batch_file", "manual_email", "api"]
DomainType = Literal["pension_contribution", "pension_transfer", "healthcare_claim"]
BinaryLabel = Literal[0, 1]
TriageBucket = Literal["AUTO_PROCESS", "STANDARD_QA", "PRIORITY_ESCALATION"]


class ClaimInput(BaseModel):
    """Validated inference input schema for a single operational claim record.

    Enforces strict typing, domain-specific numeric bounds, controlled categorical
    vocabularies, and a zero-PII defense-in-depth policy (`extra="forbid"`) that
    rejects unexpected attributes (such as PII fields or target labels) at the
    request boundary.
    """

    # Security invariant: fail closed on unmapped attributes to prevent accidental
    # PII leakage or target label leakage into downstream feature pipelines.
    model_config = ConfigDict(extra="forbid")

    # Numeric operational signals
    claim_amount: float = Field(
        gt=0.0,
        le=MAX_CLAIM_AMOUNT,
        description=(
            "Total monetary claim or transaction amount; must be strictly positive."
        ),
    )
    processing_age_hours: float = Field(
        ge=0.0,
        le=MAX_PROCESSING_AGE_HOURS,
        description="Elapsed time in hours since initial submission (0 to 30 days).",
    )
    reconciliation_variance: float = Field(
        ge=0.0,
        le=MAX_RECONCILIATION_VARIANCE,
        description=(
            "Absolute discrepancy between expected and ledger-reconciled amounts."
        ),
    )
    prior_qa_flags: int = Field(
        ge=0,
        le=MAX_PRIOR_QA_FLAGS,
        description=(
            "Count of historical quality-assurance flags tied to the batch or entity."
        ),
    )
    document_count: int = Field(
        ge=0,
        le=MAX_DOCUMENT_COUNT,
        description="Number of supporting documents attached to the claim packet.",
    )

    # Categorical & boolean signals
    submission_channel: SubmissionChannel = Field(
        description="Ingestion pathway through which the claim entered operations.",
    )
    auth_doc_verified: bool = Field(
        description=(
            "Whether required authorization documentation passed automated checks."
        ),
    )
    domain_type: DomainType = Field(
        description="Operational business line governing compliance and triage rules.",
    )


class ClaimRecord(ClaimInput):
    """Validated schema for a labeled claim or transaction record in training/CSV data.

    Inherits all 8 input feature fields from `ClaimInput` and appends `is_exception`
    at the end to preserve the 9-column CSV schema order while keeping `/predict`
    strictly rejecting `is_exception` via `extra="forbid"` on `ClaimInput`.
    """

    # Supervised classification target (~3-4% positive base rate)
    is_exception: BinaryLabel = Field(
        description=(
            "Binary QA triage target: 1 if manual intervention is required, else 0."
        ),
    )


class PredictionResponse(BaseModel):
    """Validated inference response payload for a scored claim record."""

    model_config = ConfigDict(extra="forbid")

    exception_probability: float = Field(
        ge=MIN_PROBABILITY,
        le=MAX_PROBABILITY,
        description="Predicted probability of manual QA exception in [0.0, 1.0].",
    )
    triage_bucket: TriageBucket = Field(
        description=(
            "Assigned operational routing bucket based on probability thresholds."
        ),
    )
    model_version: str = Field(
        min_length=1,
        description="Identifier of the model artifact producing the prediction.",
    )
    standard_qa_threshold: float = Field(
        gt=MIN_PROBABILITY,
        lt=MAX_PROBABILITY,
        description="Probability threshold separating AUTO_PROCESS from STANDARD_QA.",
    )
    priority_escalation_threshold: float = Field(
        gt=MIN_PROBABILITY,
        lt=MAX_PROBABILITY,
        description=(
            "Probability threshold separating STANDARD_QA from PRIORITY_ESCALATION."
        ),
    )


class HealthResponse(BaseModel):
    """Validated health and model-readiness contract for the serving endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "not_ready"] = Field(
        description="Current readiness state of the inference service.",
    )
    model_loaded: bool = Field(
        description="Whether a valid model artifact is loaded into memory.",
    )
    model_version: str | None = Field(
        default=None,
        description="Loaded model version identifier, if available.",
    )
    model_sha256_prefix: str | None = Field(
        default=None,
        min_length=12,
        max_length=12,
        description=(
            "First 12 hex characters of the model artifact SHA-256 digest for "
            "lightweight version verification."
        ),
    )
