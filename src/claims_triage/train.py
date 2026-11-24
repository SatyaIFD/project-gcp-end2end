from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import joblib
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from claims_triage import __version__
from claims_triage.features import (
    build_preprocessor,
    load_claims_csv,
    records_to_feature_matrix,
)
from claims_triage.schemas import ClaimRecord

DEFAULT_SEED: int = 42
DEFAULT_CV_SPLITS: int = 5
MIN_CV_SPLITS: int = 2
BASELINE_MAX_ITER: int = 1000
CANDIDATE_MAX_DEPTH: int = 4
CANDIDATE_LEARNING_RATE: float = 0.05
CANDIDATE_MAX_ITER: int = 200
CANDIDATE_L2_REG: float = 1.0

MIN_METRIC: float = 0.0
MAX_METRIC: float = 1.0
MIN_LIFT: float = -1.0

TEST_SPLIT_SIZE: float = 0.20
VAL_SPLIT_RATIO: float = 0.25

TARGET_QA_RECALL: float = 0.80
MIN_PRIORITY_PRECISION: float = 0.50
MIN_PR_AUC_LIFT: float = 0.0
MIN_THRESHOLD_BOUND: float = 0.01
MAX_STANDARD_QA_BOUND: float = 0.85
MAX_PRIORITY_BOUND: float = 0.99
DEFAULT_STANDARD_QA_FALLBACK: float = 0.20
DEFAULT_PRIORITY_FALLBACK: float = 0.70
PRIORITY_FALLBACK_MARGIN: float = 0.10

DEFAULT_INPUT_PATH: Path = Path("data/claims_train.csv")
DEFAULT_ARTIFACTS_DIR: Path = Path("artifacts")
MODEL_FILENAME: str = "model.joblib"
METRICS_FILENAME: str = "metrics.json"
HASH_CHUNK_BYTES: int = 65_536
SHA256_HEX_LENGTH: int = 64
MIN_TRAINING_ROWS: int = 1
JSON_INDENT: int = 2

LOGGER = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """Raised when the candidate model fails to pass promotion quality gates."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when a model artifact fails path or SHA-256 integrity checks."""


class TriageThresholds(BaseModel):
    """Probability thresholds separating auto-processed, QA, and priority claims."""

    model_config = ConfigDict(extra="forbid")

    standard_qa: float = Field(gt=MIN_METRIC, lt=MAX_METRIC)
    priority_escalation: float = Field(gt=MIN_METRIC, lt=MAX_METRIC)

    @model_validator(mode="after")
    def validate_ordering(self) -> Self:
        if self.priority_escalation <= self.standard_qa:
            msg = (
                "priority_escalation must be strictly greater than standard_qa: "
                f"got priority_escalation={self.priority_escalation} vs "
                f"standard_qa={self.standard_qa}"
            )
            raise ValueError(msg)
        return self


class EvaluationReport(BaseModel):
    """Cross-validation, holdout test metrics, and gate outcome for a run."""

    model_config = ConfigDict(extra="forbid")

    baseline_cv_pr_auc: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    candidate_cv_pr_auc: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    pr_auc_lift: float = Field(ge=MIN_LIFT, le=MAX_METRIC)
    test_pr_auc: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    test_roc_auc: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    test_recall_at_qa: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    test_precision_at_priority: float = Field(ge=MIN_METRIC, le=MAX_METRIC)
    thresholds: TriageThresholds
    passed_quality_gate: bool


class ModelMetadata(BaseModel):
    """Serialized artifact manifest containing digest, training context, and report."""

    model_config = ConfigDict(extra="forbid")

    model_version: str = Field(min_length=1)
    model_filename: str = Field(min_length=1)
    model_sha256: str = Field(
        min_length=SHA256_HEX_LENGTH,
        max_length=SHA256_HEX_LENGTH,
        pattern=r"^[0-9a-f]{64}$",
    )
    training_rows: int = Field(gt=0)
    seed: int
    trained_at_utc: str = Field(min_length=1)
    report: EvaluationReport


def build_baseline_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    """Build a leakage-free balanced logistic regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=BASELINE_MAX_ITER,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_candidate_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    """Build a leakage-free histogram gradient boosting candidate pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=CANDIDATE_MAX_DEPTH,
                    learning_rate=CANDIDATE_LEARNING_RATE,
                    max_iter=CANDIDATE_MAX_ITER,
                    l2_regularization=CANDIDATE_L2_REG,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_cv_pr_auc(
    pipeline: Pipeline,
    x_data: NDArray[np.object_],
    y_data: NDArray[np.int64],
    *,
    n_splits: int = DEFAULT_CV_SPLITS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Compute mean stratified cross-validation PR-AUC (average precision)."""
    if n_splits < MIN_CV_SPLITS:
        msg = f"n_splits must be at least {MIN_CV_SPLITS}, got {n_splits}"
        raise ValueError(msg)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(
        pipeline,
        x_data,
        y_data,
        cv=splitter,
        scoring="average_precision",
    )
    return float(np.mean(scores))


def calibrate_triage_thresholds(
    y_val: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    target_recall: float = TARGET_QA_RECALL,
    min_priority_precision: float = MIN_PRIORITY_PRECISION,
) -> TriageThresholds:
    """Calibrate ordered triage thresholds on validation probabilities."""
    if len(y_val) == 0 or len(y_prob) == 0 or len(y_val) != len(y_prob):
        msg = (
            "y_val and y_prob must be non-empty and have matching lengths: "
            f"got len(y_val)={len(y_val)}, len(y_prob)={len(y_prob)}"
        )
        raise ValueError(msg)

    unique_classes = np.unique(y_val)
    if len(unique_classes) < MIN_CV_SPLITS:
        msg = f"y_val must contain both classes 0 and 1, got {unique_classes.tolist()}"
        raise ValueError(msg)

    _precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    precisions_aligned = _precisions[:-1]
    recalls_aligned = recalls[:-1]

    qa_mask = (
        (thresholds >= MIN_THRESHOLD_BOUND)
        & (thresholds <= MAX_STANDARD_QA_BOUND)
        & (recalls_aligned >= target_recall)
    )
    if bool(np.any(qa_mask)):
        standard_qa = float(np.max(thresholds[qa_mask]))
    else:
        standard_qa = DEFAULT_STANDARD_QA_FALLBACK

    priority_mask = (
        (thresholds > standard_qa)
        & (thresholds <= MAX_PRIORITY_BOUND)
        & (precisions_aligned >= min_priority_precision)
    )
    if bool(np.any(priority_mask)):
        priority_escalation = float(np.min(thresholds[priority_mask]))
    else:
        priority_escalation = min(
            MAX_PRIORITY_BOUND,
            max(standard_qa + PRIORITY_FALLBACK_MARGIN, DEFAULT_PRIORITY_FALLBACK),
        )

    return TriageThresholds(
        standard_qa=standard_qa,
        priority_escalation=priority_escalation,
    )


def train_and_evaluate(
    records: Sequence[ClaimRecord],
    *,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_CV_SPLITS,
    min_pr_auc_lift: float = MIN_PR_AUC_LIFT,
    enforce_gate: bool = True,
) -> tuple[Pipeline, EvaluationReport]:
    """Run stratified CV gate, calibrate thresholds, and evaluate on holdout."""
    x_all, y_all = records_to_feature_matrix(records)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_all,
        y_all,
        test_size=TEST_SPLIT_SIZE,
        stratify=y_all,
        random_state=seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VAL_SPLIT_RATIO,
        stratify=y_train_val,
        random_state=seed,
    )

    baseline_pipeline = build_baseline_pipeline(seed=seed)
    candidate_pipeline = build_candidate_pipeline(seed=seed)

    baseline_cv_pr_auc = evaluate_cv_pr_auc(
        baseline_pipeline,
        x_train,
        y_train,
        n_splits=n_splits,
        seed=seed,
    )
    candidate_cv_pr_auc = evaluate_cv_pr_auc(
        candidate_pipeline,
        x_train,
        y_train,
        n_splits=n_splits,
        seed=seed,
    )

    pr_auc_lift = candidate_cv_pr_auc - baseline_cv_pr_auc
    passed_quality_gate = pr_auc_lift >= min_pr_auc_lift

    if enforce_gate and not passed_quality_gate:
        msg = (
            "Candidate model failed PR-AUC quality gate: "
            f"candidate_cv_pr_auc={candidate_cv_pr_auc:.4f}, "
            f"baseline_cv_pr_auc={baseline_cv_pr_auc:.4f}, "
            f"lift={pr_auc_lift:.4f} < required_lift={min_pr_auc_lift:.4f}"
        )
        raise QualityGateError(msg)

    candidate_pipeline.fit(x_train, y_train)

    val_probs = np.asarray(
        candidate_pipeline.predict_proba(x_val)[:, 1],
        dtype=np.float64,
    )
    thresholds = calibrate_triage_thresholds(y_val, val_probs)

    test_probs = np.asarray(
        candidate_pipeline.predict_proba(x_test)[:, 1],
        dtype=np.float64,
    )
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))

    qa_preds = (test_probs >= thresholds.standard_qa).astype(np.int64)
    test_recall_at_qa = float(
        recall_score(y_test, qa_preds, zero_division=0.0),
    )

    priority_preds = (test_probs >= thresholds.priority_escalation).astype(np.int64)
    test_precision_at_priority = float(
        precision_score(y_test, priority_preds, zero_division=0.0),
    )

    report = EvaluationReport(
        baseline_cv_pr_auc=baseline_cv_pr_auc,
        candidate_cv_pr_auc=candidate_cv_pr_auc,
        pr_auc_lift=pr_auc_lift,
        test_pr_auc=test_pr_auc,
        test_roc_auc=test_roc_auc,
        test_recall_at_qa=test_recall_at_qa,
        test_precision_at_priority=test_precision_at_priority,
        thresholds=thresholds,
        passed_quality_gate=passed_quality_gate,
    )
    return candidate_pipeline, report


def compute_file_sha256(file_path: Path) -> str:
    """Stream a file in binary chunks and return its lowercase SHA-256 hex digest."""
    if not file_path.exists():
        msg = f"File not found for SHA-256 computation: {file_path}"
        raise FileNotFoundError(msg)

    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def save_model_artifacts(
    pipeline: Pipeline,
    report: EvaluationReport,
    output_dir: Path,
    *,
    training_rows: int,
    seed: int,
    model_version: str = __version__,
) -> ModelMetadata:
    """Persist fitted pipeline artifact and signed metadata manifest."""
    if training_rows < MIN_TRAINING_ROWS:
        msg = f"training_rows must be at least {MIN_TRAINING_ROWS}, got {training_rows}"
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)

    model_sha256 = compute_file_sha256(model_path)
    metadata = ModelMetadata(
        model_version=model_version,
        model_filename=MODEL_FILENAME,
        model_sha256=model_sha256,
        training_rows=training_rows,
        seed=seed,
        trained_at_utc=datetime.now(UTC).isoformat(),
        report=report,
    )

    metrics_path = output_dir / METRICS_FILENAME
    metrics_path.write_text(
        metadata.model_dump_json(indent=JSON_INDENT),
        encoding="utf-8",
    )
    return metadata


def load_verified_artifact(artifacts_dir: Path) -> tuple[Pipeline, ModelMetadata]:
    """Validate path containment and SHA-256 digest before loading model artifact."""
    metrics_path = artifacts_dir / METRICS_FILENAME
    if not metrics_path.exists():
        msg = f"Model metadata manifest not found: {metrics_path}"
        raise FileNotFoundError(msg)

    metadata = ModelMetadata.model_validate_json(
        metrics_path.read_text(encoding="utf-8")
    )

    base_dir = artifacts_dir.resolve()
    model_path = (base_dir / metadata.model_filename).resolve()
    if model_path.parent != base_dir:
        msg = "Model path escapes artifacts directory."
        raise ArtifactIntegrityError(msg)

    actual_sha256 = compute_file_sha256(model_path)
    if not hmac.compare_digest(actual_sha256, metadata.model_sha256):
        msg = (
            "Model artifact SHA-256 mismatch: "
            f"expected={metadata.model_sha256}, actual={actual_sha256}"
        )
        raise ArtifactIntegrityError(msg)

    loaded = joblib.load(model_path)
    if not isinstance(loaded, Pipeline):
        loaded_type = type(loaded).__name__
        msg = f"Deserialized artifact is not a sklearn Pipeline: {loaded_type}"
        raise ArtifactIntegrityError(msg)

    return loaded, metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for model training, gating, and artifact export."""
    parser = argparse.ArgumentParser(
        prog="claims_triage.train",
        description="Train and evaluate the claims triage model with quality gates.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input training CSV path (default: {DEFAULT_INPUT_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Output directory for artifacts (default: {DEFAULT_ARTIFACTS_DIR}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_CV_SPLITS,
        help=f"Number of stratified CV folds (default: {DEFAULT_CV_SPLITS}).",
    )
    parser.add_argument(
        "--min-pr-auc-lift",
        type=float,
        default=MIN_PR_AUC_LIFT,
        help=f"Minimum PR-AUC lift over baseline (default: {MIN_PR_AUC_LIFT}).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=__version__,
        help=f"Model version identifier (default: {__version__}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> ModelMetadata:
    """CLI entrypoint that loads data, runs gated training, and saves artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)
    input_path: Path = args.input
    output_dir: Path = args.output_dir
    seed: int = args.seed
    n_splits: int = args.n_splits
    min_pr_auc_lift: float = args.min_pr_auc_lift
    model_version: str = args.model_version

    records = load_claims_csv(input_path)
    pipeline, report = train_and_evaluate(
        records,
        seed=seed,
        n_splits=n_splits,
        min_pr_auc_lift=min_pr_auc_lift,
        enforce_gate=True,
    )
    metadata = save_model_artifacts(
        pipeline,
        report,
        output_dir,
        training_rows=len(records),
        seed=seed,
        model_version=model_version,
    )

    LOGGER.info(
        json.dumps(
            {
                "event": "model_trained_and_saved",
                "model_version": metadata.model_version,
                "training_rows": metadata.training_rows,
                "baseline_cv_pr_auc": report.baseline_cv_pr_auc,
                "candidate_cv_pr_auc": report.candidate_cv_pr_auc,
                "pr_auc_lift": report.pr_auc_lift,
                "test_pr_auc": report.test_pr_auc,
                "model_sha256": metadata.model_sha256,
                "output_dir": str(output_dir),
            }
        )
    )
    return metadata


if __name__ == "__main__":
    main()  # pragma: no cover
