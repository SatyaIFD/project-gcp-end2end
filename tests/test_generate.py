"""Unit tests for the deterministic synthetic claim generator and CSV CLI."""

import csv
import json
import logging
from pathlib import Path

import pytest

from claims_triage.generate import (
    DEFAULT_SEED,
    VERIFIED_RATE_DRIFT,
    VERIFIED_RATE_NORMAL,
    generate_claims,
    main,
    write_claims_csv,
)
from claims_triage.schemas import ClaimRecord

SAMPLE_SIZE_SMALL: int = 50
ALTERNATE_SEED: int = 99
INVALID_ZERO_SAMPLES: int = 0
INVALID_NEGATIVE_SAMPLES: int = -10
BATCH_SIZE_DISTRIBUTION: int = 2_000
ROUNDTRIP_SAMPLE_COUNT: int = 25
CLI_SAMPLE_COUNT: int = 40
CLI_SEED: int = 7
EXPECTED_HEADER_ROWS: int = 1
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


def test_write_claims_csv_roundtrip(tmp_path: Path) -> None:
    """Verify CSV export creates nested directories and round-trips schema rows."""
    records = generate_claims(ROUNDTRIP_SAMPLE_COUNT, seed=DEFAULT_SEED)
    nested_csv = tmp_path / "nested" / "claims.csv"

    written = write_claims_csv(records, nested_csv)
    assert written == nested_csv
    assert nested_csv.exists()

    expected_fields = list(ClaimRecord.model_fields.keys())
    with nested_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == expected_fields
        rows = list(reader)

    assert len(rows) == ROUNDTRIP_SAMPLE_COUNT
    reloaded = [
        ClaimRecord.model_validate(
            {
                **row,
                "is_exception": int(row["is_exception"]),
            }
        )
        for row in rows
    ]
    assert reloaded == records

    with pytest.raises(ValueError, match="empty"):
        write_claims_csv([], tmp_path / "empty.csv")


def test_cli_main_generates_normal_and_drift_csv(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify `main()` generates normal and drifted CSVs and logs JSON summary."""
    normal_path = tmp_path / "claims_normal.csv"
    drift_path = tmp_path / "claims_drift.csv"

    with caplog.at_level(logging.INFO):
        main(
            [
                "--n-samples",
                str(CLI_SAMPLE_COUNT),
                "--seed",
                str(CLI_SEED),
                "--output",
                str(normal_path),
            ]
        )

    normal_summary = json.loads(caplog.records[-1].message)
    assert normal_summary["n_samples"] == CLI_SAMPLE_COUNT
    assert normal_summary["drift"] is False

    caplog.clear()
    with caplog.at_level(logging.INFO):
        main(
            [
                "--n-samples",
                str(CLI_SAMPLE_COUNT),
                "--seed",
                str(CLI_SEED),
                "--drift",
                "--output",
                str(drift_path),
            ]
        )

    drift_summary = json.loads(caplog.records[-1].message)
    assert drift_summary["n_samples"] == CLI_SAMPLE_COUNT
    assert drift_summary["drift"] is True

    for path in (normal_path, drift_path):
        assert path.exists()
        with path.open("r", encoding="utf-8", newline="") as handle:
            line_count = sum(1 for _ in csv.reader(handle))
        assert line_count == CLI_SAMPLE_COUNT + EXPECTED_HEADER_ROWS
