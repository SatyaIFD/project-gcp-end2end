"""FastAPI inference service and application factory for operational claim triage."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from sklearn.pipeline import Pipeline

from claims_triage import __version__
from claims_triage.features import inputs_to_feature_matrix
from claims_triage.schemas import (
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    ClaimInput,
    HealthResponse,
    PredictionResponse,
    TriageBucket,
)
from claims_triage.train import (
    ArtifactIntegrityError,
    ModelMetadata,
    TriageThresholds,
    load_verified_artifact,
)

DEFAULT_ARTIFACTS_DIR: Path = Path("artifacts")
ARTIFACTS_DIR_ENV_VAR: str = "CLAIMS_ARTIFACTS_DIR"
SHA256_PREFIX_LENGTH: int = 12
MS_PER_SECOND: float = 1000.0
LATENCY_DECIMAL_PLACES: int = 3
PROBABILITY_DECIMAL_PLACES: int = 6
LOGGER: logging.Logger = logging.getLogger(__name__)


def assign_triage_bucket(
    probability: float,
    thresholds: TriageThresholds,
) -> TriageBucket:
    """Map a calibrated exception probability to an operational triage bucket."""
    if probability >= thresholds.priority_escalation:
        return "PRIORITY_ESCALATION"
    if probability >= thresholds.standard_qa:
        return "STANDARD_QA"
    return "AUTO_PROCESS"


@dataclass
class ModelState:
    """Typed container holding runtime model state scoped to a single app instance."""

    pipeline: Pipeline | None = None
    metadata: ModelMetadata | None = None
    load_error: str | None = None


def _resolve_artifacts_dir(artifacts_dir: Path | None) -> Path:
    """Resolve the artifact directory from explicit argument, env var, or default."""
    if artifacts_dir is not None:
        return artifacts_dir
    env_dir = os.getenv(ARTIFACTS_DIR_ENV_VAR)
    if env_dir:
        return Path(env_dir)
    return DEFAULT_ARTIFACTS_DIR


def _emit_json_log(level: int, payload: dict[str, object]) -> None:
    """Emit a single-line JSON log record for Cloud Logging `jsonPayload` ingestion."""
    LOGGER.log(level, json.dumps(payload, separators=(",", ":")))


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    """Create a configured FastAPI application instance with isolated runtime state."""
    resolved_dir = _resolve_artifacts_dir(artifacts_dir)
    state = ModelState()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        try:
            pipeline, metadata = load_verified_artifact(resolved_dir)
        except (FileNotFoundError, ArtifactIntegrityError, ValueError) as exc:
            state.pipeline = None
            state.metadata = None
            state.load_error = str(exc)
            _emit_json_log(
                logging.ERROR,
                {
                    "event": "model_artifact_load_failed",
                    "error_type": type(exc).__name__,
                },
            )
        else:
            state.pipeline = pipeline
            state.metadata = metadata
            state.load_error = None
            _emit_json_log(
                logging.INFO,
                {
                    "event": "model_artifact_loaded",
                    "model_version": metadata.model_version,
                    "model_sha256_prefix": metadata.model_sha256[:SHA256_PREFIX_LENGTH],
                },
            )
        yield

    app = FastAPI(
        title="Claims Triage API",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse | JSONResponse:
        if state.pipeline is None or state.metadata is None:
            not_ready = HealthResponse(
                status="not_ready",
                model_loaded=False,
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=not_ready.model_dump(),
            )

        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=state.metadata.model_version,
            model_sha256_prefix=state.metadata.model_sha256[:SHA256_PREFIX_LENGTH],
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(claim: ClaimInput) -> PredictionResponse:
        if state.pipeline is None or state.metadata is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifact is not loaded.",
            )

        start_time = time.perf_counter()
        x_matrix = inputs_to_feature_matrix([claim])
        raw_prob = float(state.pipeline.predict_proba(x_matrix)[0, 1])
        clamped_prob = min(MAX_PROBABILITY, max(MIN_PROBABILITY, raw_prob))
        rounded_prob = round(clamped_prob, PROBABILITY_DECIMAL_PLACES)

        thresholds = state.metadata.report.thresholds
        bucket = assign_triage_bucket(clamped_prob, thresholds)
        latency_ms = round(
            (time.perf_counter() - start_time) * MS_PER_SECOND,
            LATENCY_DECIMAL_PLACES,
        )

        _emit_json_log(
            logging.INFO,
            {
                "event": "prediction_served",
                "model_version": state.metadata.model_version,
                "triage_bucket": bucket,
                "exception_probability": rounded_prob,
                "latency_ms": latency_ms,
            },
        )

        return PredictionResponse(
            exception_probability=clamped_prob,
            triage_bucket=bucket,
            model_version=state.metadata.model_version,
            standard_qa_threshold=thresholds.standard_qa,
            priority_escalation_threshold=thresholds.priority_escalation,
        )

    return app


app: FastAPI = create_app()
