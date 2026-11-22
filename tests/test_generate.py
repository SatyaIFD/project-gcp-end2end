"""Unit tests for the deterministic synthetic claim generator."""

import pytest

from claims_triage.generate import (
    DEFAULT_SEED,
    VERIFIED_RATE_DRIFT,
    VERIFIED_RATE_NORMAL,
    generate_claims,
)

SAMPLE_SIZE_SMALL: int = 50
ALTERNATE_SEED: int = 99
INVALID_ZERO_SAMPLES: int = 0
INVALID_NEGATIVE_SAMPLES: int = -10
BATCH_SIZE_DISTRIBUTION: int = 2_000
MIN_EXCEPTION_RATE: float = 0.015
MAX_EXCEPTION_RATE: float = 0.07
MAX_VERIFIED_RATE_TOLERANCE: float = 0.05


def test_generate_claims_deterministic_reproducibility() -> None:
    """Verify identical seeds reproduce exact output and distinct seeds differ."""
    first_run = generate_claims(SAMPLE_SIZE_SMALL, seed=DEFAULT_SEED)
    second_run = generate_claims(SAMPLE_SIZE_SMALL, seed=DEFAULT_SEED)
    different_seed_run = generate_claims(SAMPLE_SIZE_SMALL, seed=ALTERNATE_SEED)

    assert first_run == second_run
    assert first_run != different_seed_run


def test_generate_claims_invalid_sample_size() -> None:
    """Verify non-positive sample sizes raise `ValueError`."""
    with pytest.raises(ValueError, match="n_samples"):
        generate_claims(INVALID_ZERO_SAMPLES, seed=DEFAULT_SEED)

    with pytest.raises(ValueError, match="n_samples"):
        generate_claims(INVALID_NEGATIVE_SAMPLES, seed=DEFAULT_SEED)


def test_generate_claims_realistic_class_imbalance() -> None:
    """Verify normal batches maintain a realistic minority exception rate."""
    records = generate_claims(BATCH_SIZE_DISTRIBUTION, seed=DEFAULT_SEED, drift=False)
    exception_rate = sum(r.is_exception for r in records) / len(records)

    assert MIN_EXCEPTION_RATE <= exception_rate <= MAX_EXCEPTION_RATE


def test_generate_claims_drift_shifts_feature_distributions() -> None:
    """Verify `drift=True` shifts amounts/ages upward and lowers verification rate."""
    normal_records = generate_claims(
        BATCH_SIZE_DISTRIBUTION,
        seed=DEFAULT_SEED,
        drift=False,
    )
    drifted_records = generate_claims(
        BATCH_SIZE_DISTRIBUTION,
        seed=DEFAULT_SEED,
        drift=True,
    )

    normal_mean_amount = sum(r.claim_amount for r in normal_records) / len(
        normal_records
    )
    drifted_mean_amount = sum(r.claim_amount for r in drifted_records) / len(
        drifted_records
    )

    normal_mean_age = sum(r.processing_age_hours for r in normal_records) / len(
        normal_records
    )
    drifted_mean_age = sum(r.processing_age_hours for r in drifted_records) / len(
        drifted_records
    )

    normal_verified_rate = sum(1 for r in normal_records if r.auth_doc_verified) / len(
        normal_records
    )
    drifted_verified_rate = sum(
        1 for r in drifted_records if r.auth_doc_verified
    ) / len(drifted_records)

    assert drifted_mean_amount > normal_mean_amount
    assert drifted_mean_age > normal_mean_age
    assert drifted_verified_rate < normal_verified_rate
    assert (
        abs(normal_verified_rate - VERIFIED_RATE_NORMAL) < MAX_VERIFIED_RATE_TOLERANCE
    )
    assert (
        abs(drifted_verified_rate - VERIFIED_RATE_DRIFT) < MAX_VERIFIED_RATE_TOLERANCE
    )
