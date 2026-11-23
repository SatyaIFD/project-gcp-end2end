from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from claims_triage.features import (
    ALL_FEATURES,
    build_preprocessor,
    load_claims_csv,
    records_to_feature_matrix,
)
from claims_triage.generate import generate_claims, write_claims_csv
from claims_triage.schemas import ClaimRecord


def test_load_claims_csv_valid_and_missing(tmp_path: Path) -> None:
    batch_size = 10
    records = generate_claims(n_samples=batch_size, seed=42)
    valid_csv = tmp_path / "claims.csv"
    write_claims_csv(records, valid_csv)

    loaded = load_claims_csv(valid_csv)
    assert len(loaded) == batch_size
    assert loaded == records

    missing_csv = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_claims_csv(missing_csv)

    empty_csv = tmp_path / "header_only.csv"
    header_fields = [*ClaimRecord.model_fields.keys()]
    with empty_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header_fields)
        writer.writeheader()

    with pytest.raises(ValueError, match="zero data rows"):
        load_claims_csv(empty_csv)


def test_load_claims_csv_rejects_unexpected_pii_column(tmp_path: Path) -> None:
    pii_batch_size = 2
    records = generate_claims(n_samples=pii_batch_size, seed=7)
    pii_csv = tmp_path / "claims_with_pii.csv"
    fieldnames = [*ClaimRecord.model_fields.keys(), "customer_email"]

    with pii_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.model_dump()
            row["customer_email"] = "leak@example.com"
            writer.writerow(row)

    with pytest.raises(ValidationError):
        load_claims_csv(pii_csv)


def test_records_to_feature_matrix_shapes_and_empty() -> None:
    sample_count = 8
    records = generate_claims(n_samples=sample_count, seed=123)

    features, targets = records_to_feature_matrix(records)
    assert features.shape == (sample_count, len(ALL_FEATURES))
    assert features.dtype == np.object_
    assert targets.shape == (sample_count,)
    assert targets.dtype == np.int64

    with pytest.raises(ValueError, match="empty sequence"):
        records_to_feature_matrix([])


def test_build_preprocessor_fit_transform_and_unseen_category() -> None:
    expected_ndim = 2
    train_count = 20
    val_count = 6
    train_records = generate_claims(n_samples=train_count, seed=101)
    val_records = generate_claims(n_samples=val_count, seed=202)

    x_train, _ = records_to_feature_matrix(train_records)
    x_val, _ = records_to_feature_matrix(val_records)

    preprocessor = build_preprocessor()
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_val_transformed = preprocessor.transform(x_val)

    assert isinstance(x_train_transformed, np.ndarray)
    assert isinstance(x_val_transformed, np.ndarray)
    assert x_train_transformed.ndim == expected_ndim
    assert x_val_transformed.ndim == expected_ndim
    assert x_val_transformed.shape[0] == len(val_records)
    assert np.issubdtype(x_val_transformed.dtype, np.floating)
    assert np.isfinite(x_val_transformed).all()

    unseen_row = x_val[:1].copy()
    unseen_row[0, -1] = "UNSEEN_DOMAIN_CATEGORY_XYZ"
    x_unseen_transformed = preprocessor.transform(unseen_row)

    assert x_unseen_transformed.shape == (1, x_train_transformed.shape[1])
    assert np.issubdtype(x_unseen_transformed.dtype, np.floating)
    assert np.isfinite(x_unseen_transformed).all()
