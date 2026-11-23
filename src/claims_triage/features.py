from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from claims_triage.schemas import ClaimRecord

NUMERIC_FEATURES: tuple[str, ...] = (
    "claim_amount",
    "processing_age_hours",
    "reconciliation_variance",
    "prior_qa_flags",
    "document_count",
)
BOOLEAN_FEATURES: tuple[str, ...] = ("auth_doc_verified",)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "submission_channel",
    "domain_type",
)
ALL_FEATURES: tuple[str, ...] = (
    *NUMERIC_FEATURES,
    *BOOLEAN_FEATURES,
    *CATEGORICAL_FEATURES,
)
TARGET_COLUMN: str = "is_exception"


def load_claims_csv(csv_path: Path) -> list[ClaimRecord]:
    """Load and validate claim records from a UTF-8 CSV file."""
    if not csv_path.exists():
        msg = f"Claims CSV not found: {csv_path}"
        raise FileNotFoundError(msg)

    records: list[ClaimRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            payload: dict[str, object] = dict(row)
            if TARGET_COLUMN in payload and payload[TARGET_COLUMN] is not None:
                raw_target = str(payload[TARGET_COLUMN]).strip()
                if raw_target != "":
                    payload[TARGET_COLUMN] = int(raw_target)
            records.append(ClaimRecord.model_validate(payload))

    if not records:
        msg = f"Claims CSV contains zero data rows: {csv_path}"
        raise ValueError(msg)

    return records


def records_to_feature_matrix(
    records: Sequence[ClaimRecord],
) -> tuple[NDArray[np.object_], NDArray[np.int64]]:
    """Convert validated ClaimRecord objects into (X, y) NumPy arrays."""
    if not records:
        msg = "Cannot build feature matrix from an empty sequence of records."
        raise ValueError(msg)

    rows: list[list[object]] = []
    targets: list[int] = []

    for record in records:
        row: list[object] = []
        for feature in ALL_FEATURES:
            value = getattr(record, feature)
            if feature in NUMERIC_FEATURES:
                row.append(float(value))
            elif feature in BOOLEAN_FEATURES:
                row.append(int(bool(value)))
            elif feature in CATEGORICAL_FEATURES:
                row.append(str(value))
            else:
                msg = f"Unhandled feature column: {feature}"
                raise ValueError(msg)
        rows.append(row)
        targets.append(int(record.is_exception))

    features = np.asarray(rows, dtype=np.object_)
    labels = np.asarray(targets, dtype=np.int64)
    return features, labels


def build_preprocessor() -> ColumnTransformer:
    """Build a skew-free ColumnTransformer over ALL_FEATURES integer indices."""
    numeric_start = 0
    numeric_end = len(NUMERIC_FEATURES)
    bool_start = numeric_end
    bool_end = bool_start + len(BOOLEAN_FEATURES)
    cat_start = bool_end
    cat_end = cat_start + len(CATEGORICAL_FEATURES)

    numeric_idx = list(range(numeric_start, numeric_end))
    bool_idx = list(range(bool_start, bool_end))
    cat_idx = list(range(cat_start, cat_end))

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_idx),
            (
                "boolean",
                FunctionTransformer(np.asarray, kw_args={"dtype": np.float64}),
                bool_idx,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_idx,
            ),
        ],
        remainder="drop",
    )
