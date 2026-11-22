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

# Controlled vocabularies matching `docs/data-governance.md` Section 2.
SubmissionChannel = Literal["portal", "batch_file", "manual_email", "api"]
DomainType = Literal["pension_contribution", "pension_transfer", "healthcare_claim"]
BinaryLabel = Literal[0, 1]


class ClaimRecord(BaseModel):
    """Validated schema for a single anonymized claim or transaction row.

    Enforces strict typing, domain-specific numeric bounds, controlled categorical
    vocabularies, and a zero-PII defense-in-depth policy (`extra="forbid"`) that
    rejects any unexpected attributes (e.g., `ssn`, `customer_email`) at the
    boundary rather than silently dropping them.
    """

    # Security invariant: fail closed on unmapped attributes to prevent accidental
    # PII leakage into logs, serialized artifacts, or downstream feature stores.
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

    # Supervised classification target (~3-4% positive base rate)
    is_exception: BinaryLabel = Field(
        description=(
            "Binary QA triage target: 1 if manual intervention is required, else 0."
        ),
    )
