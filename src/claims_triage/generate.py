"""Deterministic synthetic dataset generator and CLI for operational claim triage.

Produces `ClaimRecord` instances with realistic right-skewed operational signals,
non-linear exception interactions, overlapping noise, optional covariate
drift (`drift=True`), and CSV export aligned with `docs/data-governance.md`.
"""

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from claims_triage.schemas import (
    MAX_CLAIM_AMOUNT,
    MAX_DOCUMENT_COUNT,
    MAX_PRIOR_QA_FLAGS,
    MAX_PROCESSING_AGE_HOURS,
    MAX_RECONCILIATION_VARIANCE,
    BinaryLabel,
    ClaimRecord,
    DomainType,
    SubmissionChannel,
)

# Reproducibility and distribution constants (avoids magic numbers in Ruff PLR2004).
DEFAULT_SEED: int = 42
DEFAULT_SAMPLE_COUNT: int = 10_000
DEFAULT_OUTPUT_PATH: Path = Path("data/claims_train.csv")
RATE_DECIMAL_PLACES: int = 4
MIN_CLAIM_AMOUNT: float = 0.01
DECIMAL_PLACES: int = 2

# Normal baseline distribution parameters (P(X) when drift=False).
NORMAL_AMOUNT_LOG_MEAN: float = 8.0
NORMAL_AMOUNT_LOG_SIGMA: float = 1.15
NORMAL_AGE_SHAPE: float = 1.6
NORMAL_AGE_SCALE: float = 18.0
VARIANCE_NONZERO_RATE: float = 0.18
VARIANCE_LOG_MEAN: float = 4.2
VARIANCE_LOG_SIGMA: float = 1.3
PRIOR_FLAGS_LAMBDA: float = 0.35
DOC_COUNT_LAMBDA: float = 3.0
VERIFIED_RATE_NORMAL: float = 0.88

# Drifted batch parameters (P(X) when drift=True; P(Y | X) remains unchanged).
DRIFT_AMOUNT_LOG_MEAN: float = 8.85
DRIFT_AMOUNT_LOG_SIGMA: float = 1.25
DRIFT_AGE_SHAPE: float = 2.1
DRIFT_AGE_SCALE: float = 28.0
VERIFIED_RATE_DRIFT: float = 0.62

# Risk log-odds model weights, noise, and label-flip rate (~3-4% base exception rate).
BASE_LOGIT: float = -5.15
LOGIT_CLIP_BOUND: float = 60.0
GAUSSIAN_NOISE_STD: float = 0.65
LABEL_FLIP_RATE: float = 0.01

WEIGHT_LOG_AMOUNT: float = 0.22
WEIGHT_AGE_NORM: float = 0.65
WEIGHT_LOG_VARIANCE: float = 0.32
WEIGHT_PRIOR_FLAGS: float = 0.45
WEIGHT_LOW_DOCS: float = 0.35
WEIGHT_UNVERIFIED: float = 0.85
WEIGHT_MANUAL_EMAIL: float = 0.40

# Non-linear interaction weights capturing multi-factor operational risk.
INTERACTION_VARIANCE_UNVERIFIED: float = 0.55
INTERACTION_AGE_MANUAL_EMAIL: float = 0.70

LOW_DOC_THRESHOLD: int = 2
HIGH_AGE_HOURS_THRESHOLD: float = 72.0
AGE_NORMALIZATION_HOURS: float = 100.0
VARIANCE_NORMALIZATION_SCALE: float = 100.0

# Controlled vocabularies as typed tuples so index sampling preserves Literal types
# under `mypy --strict` without needing `cast` or `# type: ignore`.
SUBMISSION_CHANNELS: tuple[SubmissionChannel, ...] = (
    "portal",
    "batch_file",
    "manual_email",
    "api",
)
SUBMISSION_CHANNEL_PROBS: tuple[float, ...] = (0.42, 0.25, 0.15, 0.18)

DOMAIN_TYPES: tuple[DomainType, ...] = (
    "pension_contribution",
    "pension_transfer",
    "healthcare_claim",
)
DOMAIN_TYPE_PROBS: tuple[float, ...] = (0.35, 0.25, 0.40)

LOGGER = logging.getLogger(__name__)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid mapping log-odds to probabilities."""
    clipped = np.clip(logits, -LOGIT_CLIP_BOUND, LOGIT_CLIP_BOUND)
    probabilities = 1.0 / (1.0 + np.exp(-clipped))
    return np.asarray(probabilities, dtype=np.float64)


def _sample_features(
    rng: np.random.Generator,
    n_samples: int,
    *,
    drift: bool,
) -> dict[str, np.ndarray]:
    """Sample feature arrays for `n_samples` rows with optional covariate shift."""
    # Shift feature distributions under drift while keeping identical risk weights.
    amount_mean = DRIFT_AMOUNT_LOG_MEAN if drift else NORMAL_AMOUNT_LOG_MEAN
    amount_sigma = DRIFT_AMOUNT_LOG_SIGMA if drift else NORMAL_AMOUNT_LOG_SIGMA
    age_shape = DRIFT_AGE_SHAPE if drift else NORMAL_AGE_SHAPE
    age_scale = DRIFT_AGE_SCALE if drift else NORMAL_AGE_SCALE
    verified_prob = VERIFIED_RATE_DRIFT if drift else VERIFIED_RATE_NORMAL

    raw_amounts = rng.lognormal(mean=amount_mean, sigma=amount_sigma, size=n_samples)
    claim_amount = np.round(
        np.clip(raw_amounts, MIN_CLAIM_AMOUNT, MAX_CLAIM_AMOUNT),
        DECIMAL_PLACES,
    )

    raw_ages = rng.gamma(shape=age_shape, scale=age_scale, size=n_samples)
    processing_age_hours = np.round(
        np.clip(raw_ages, 0.0, MAX_PROCESSING_AGE_HOURS),
        DECIMAL_PLACES,
    )

    # Most claims reconcile cleanly to 0.0; a minority exhibit positive variance.
    has_variance = rng.random(n_samples) < VARIANCE_NONZERO_RATE
    variance_magnitudes = rng.lognormal(
        mean=VARIANCE_LOG_MEAN,
        sigma=VARIANCE_LOG_SIGMA,
        size=n_samples,
    )
    reconciliation_variance = np.round(
        np.where(
            has_variance,
            np.clip(variance_magnitudes, 0.0, MAX_RECONCILIATION_VARIANCE),
            0.0,
        ),
        DECIMAL_PLACES,
    )

    prior_qa_flags = np.clip(
        rng.poisson(lam=PRIOR_FLAGS_LAMBDA, size=n_samples),
        0,
        MAX_PRIOR_QA_FLAGS,
    )
    document_count = np.clip(
        rng.poisson(lam=DOC_COUNT_LAMBDA, size=n_samples),
        0,
        MAX_DOCUMENT_COUNT,
    )

    channel_indices = rng.choice(
        len(SUBMISSION_CHANNELS),
        size=n_samples,
        p=SUBMISSION_CHANNEL_PROBS,
    )
    domain_indices = rng.choice(
        len(DOMAIN_TYPES),
        size=n_samples,
        p=DOMAIN_TYPE_PROBS,
    )
    auth_doc_verified = rng.random(n_samples) < verified_prob

    return {
        "claim_amount": claim_amount,
        "processing_age_hours": processing_age_hours,
        "reconciliation_variance": reconciliation_variance,
        "prior_qa_flags": prior_qa_flags,
        "document_count": document_count,
        "channel_indices": channel_indices,
        "domain_indices": domain_indices,
        "auth_doc_verified": auth_doc_verified,
    }


def _compute_exception_labels(
    rng: np.random.Generator,
    features: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute binary exception labels from non-linear risk + overlapping noise."""
    claim_amount = features["claim_amount"]
    processing_age_hours = features["processing_age_hours"]
    reconciliation_variance = features["reconciliation_variance"]
    prior_qa_flags = features["prior_qa_flags"]
    document_count = features["document_count"]
    channel_indices = features["channel_indices"]
    auth_doc_verified = features["auth_doc_verified"]

    is_manual_email = (
        np.array([SUBMISSION_CHANNELS[int(i)] for i in channel_indices])
        == "manual_email"
    )
    is_unverified = ~auth_doc_verified
    has_variance = reconciliation_variance > 0.0
    is_low_docs = document_count < LOW_DOC_THRESHOLD
    is_aged_claim = processing_age_hours >= HIGH_AGE_HOURS_THRESHOLD

    # Main effects normalized to keep log-odds well-scaled across feature ranges.
    log_amount_centered = np.log(claim_amount) - NORMAL_AMOUNT_LOG_MEAN
    age_scaled = processing_age_hours / AGE_NORMALIZATION_HOURS
    log_variance_scaled = np.where(
        has_variance,
        np.log1p(reconciliation_variance / VARIANCE_NORMALIZATION_SCALE),
        0.0,
    )

    logits = (
        BASE_LOGIT
        + WEIGHT_LOG_AMOUNT * log_amount_centered
        + WEIGHT_AGE_NORM * age_scaled
        + WEIGHT_LOG_VARIANCE * log_variance_scaled
        + WEIGHT_PRIOR_FLAGS * prior_qa_flags
        + WEIGHT_LOW_DOCS * is_low_docs.astype(np.float64)
        + WEIGHT_UNVERIFIED * is_unverified.astype(np.float64)
        + WEIGHT_MANUAL_EMAIL * is_manual_email.astype(np.float64)
        # Non-linear risk combinations (avoids single-threshold separability).
        + INTERACTION_VARIANCE_UNVERIFIED
        * (has_variance & is_unverified).astype(np.float64)
        + INTERACTION_AGE_MANUAL_EMAIL
        * (is_aged_claim & is_manual_email).astype(np.float64)
        # Gaussian latent noise introduces realistic class boundary overlap.
        + rng.normal(loc=0.0, scale=GAUSSIAN_NOISE_STD, size=len(claim_amount))
    )

    probabilities = _sigmoid(logits)
    labels = rng.binomial(n=1, p=probabilities)

    # Simulate minor QA labeling noise by flipping a small fraction of outcomes.
    flip_mask = rng.random(len(labels)) < LABEL_FLIP_RATE
    noisy_labels = np.where(flip_mask, 1 - labels, labels)
    return np.asarray(noisy_labels, dtype=np.int64)


def generate_claims(
    n_samples: int,
    seed: int = DEFAULT_SEED,
    *,
    drift: bool = False,
) -> list[ClaimRecord]:
    """Generate a deterministic list of validated `ClaimRecord` entries.

    Args:
        n_samples: Positive number of synthetic claim records to generate.
        seed: Seed used to initialize a local `numpy.random.Generator` instance.
        drift: When True, shifts input feature distributions to simulate
            post-deployment covariate drift while preserving the risk model.

    Returns:
        List of `ClaimRecord` instances passing all Pydantic schema constraints.

    Raises:
        ValueError: If `n_samples` is zero or negative.
    """
    if n_samples <= 0:
        msg = f"n_samples must be a positive integer, got {n_samples}."
        raise ValueError(msg)

    # Use a local RNG instance rather than mutating global NumPy state.
    rng = np.random.default_rng(seed)
    features = _sample_features(rng, n_samples, drift=drift)
    labels = _compute_exception_labels(rng, features)

    records: list[ClaimRecord] = []
    for idx in range(n_samples):
        channel: SubmissionChannel = SUBMISSION_CHANNELS[
            int(features["channel_indices"][idx])
        ]
        domain: DomainType = DOMAIN_TYPES[int(features["domain_indices"][idx])]
        raw_label = int(labels[idx])
        label: BinaryLabel = 1 if raw_label == 1 else 0

        # Explicitly cast NumPy scalars to native Python types for strict typing.
        records.append(
            ClaimRecord(
                claim_amount=float(features["claim_amount"][idx]),
                processing_age_hours=float(features["processing_age_hours"][idx]),
                reconciliation_variance=float(features["reconciliation_variance"][idx]),
                prior_qa_flags=int(features["prior_qa_flags"][idx]),
                document_count=int(features["document_count"][idx]),
                submission_channel=channel,
                auth_doc_verified=bool(features["auth_doc_verified"][idx]),
                domain_type=domain,
                is_exception=label,
            )
        )

    return records


def write_claims_csv(records: Sequence[ClaimRecord], output_path: Path) -> Path:
    """Serialize `ClaimRecord` rows to a UTF-8 CSV file matching schema field order.

    Args:
        records: Non-empty sequence of validated `ClaimRecord` objects.
        output_path: Destination filesystem path for the generated CSV file.

    Returns:
        The written `output_path`.

    Raises:
        ValueError: If `records` is empty.
    """
    if not records:
        msg = "Cannot write CSV from an empty sequence of ClaimRecord items."
        raise ValueError(msg)

    # Ensure nested output directories (e.g., data/, reports/) exist before writing.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ClaimRecord.model_fields.keys())

    # `newline=""` and UTF-8 prevent blank lines and encoding issues across platforms.
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())

    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for synthetic claim dataset generation."""
    parser = argparse.ArgumentParser(
        prog="claims_triage.generate",
        description="Generate deterministic synthetic claim triage datasets as CSV.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of claim rows to generate (default: {DEFAULT_SAMPLE_COUNT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic RNG seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        default=False,
        help="Enable covariate distribution shift in generated features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    """CLI entrypoint that generates synthetic claims and exports them to CSV."""
    # Emit raw JSON messages so log collectors (e.g., Cloud Logging) parse fields.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)
    n_samples: int = args.n_samples
    seed: int = args.seed
    drift: bool = args.drift
    output_path: Path = args.output

    records = generate_claims(n_samples=n_samples, seed=seed, drift=drift)
    written_path = write_claims_csv(records=records, output_path=output_path)

    exception_count = sum(record.is_exception for record in records)
    exception_rate = round(exception_count / len(records), RATE_DECIMAL_PLACES)

    # Emit structured JSON summary without any raw record or PII payloads.
    LOGGER.info(
        json.dumps(
            {
                "event": "claims_dataset_generated",
                "n_samples": n_samples,
                "seed": seed,
                "drift": drift,
                "exception_count": exception_count,
                "exception_rate": exception_rate,
                "output_path": str(written_path),
            }
        )
    )
    return written_path


if __name__ == "__main__":
    main()  # pragma: no cover
