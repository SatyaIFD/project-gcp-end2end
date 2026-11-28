GCP_PROJECT_ID ?= claims-triage-mlops
GCP_REGION ?= us-central1
GCS_BUCKET ?= $(GCP_PROJECT_ID)-mlops-artifacts
BQ_DATASET ?= claims_triage
MODEL_VERSION ?= v0.1.0
BQ_MAX_BYTES_BILLED ?= 104857600

.PHONY: lint typecheck security test all gcp-upload bq-load bqml-baseline

lint:
	uv run --isolated ruff format --check .
	uv run --isolated ruff check .

typecheck:
	uv run --isolated mypy src tests

security:
	uv run --isolated bandit -c pyproject.toml -r src
	uv run --isolated pip-audit
	detect-secrets scan --exclude-files 'uv\.lock|infra/\.terraform/|infra/terraform\.tfstate.*|infra/\.terraform\.lock\.hcl' > .secrets.baseline

test:
	uv run --isolated pytest

all: lint typecheck security test

gcp-upload:
	gcloud storage cp data/claims_train.csv data/claims_drift.csv "gs://$(GCS_BUCKET)/datasets/"
	gcloud storage cp artifacts/model.joblib artifacts/metrics.json "gs://$(GCS_BUCKET)/models/$(MODEL_VERSION)/"

bq-load:
	bq --location=$(GCP_REGION) load --replace --autodetect --source_format=CSV "$(GCP_PROJECT_ID):$(BQ_DATASET).claims_train" "gs://$(GCS_BUCKET)/datasets/claims_train.csv"
	bq --location=$(GCP_REGION) load --replace --autodetect --source_format=CSV "$(GCP_PROJECT_ID):$(BQ_DATASET).claims_drift" "gs://$(GCS_BUCKET)/datasets/claims_drift.csv"

bqml-baseline:
	GCP_PROJECT_ID="$(GCP_PROJECT_ID)" BQ_DATASET="$(BQ_DATASET)" envsubst '$${GCP_PROJECT_ID} $${BQ_DATASET}' < sql/bqml_baseline.sql | bq --location=$(GCP_REGION) query --use_legacy_sql=false --maximum_bytes_billed=$(BQ_MAX_BYTES_BILLED)