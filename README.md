# Google Cloud End-to-End MLOps: Claims & Transaction QA Triage

End-to-end, cost-controlled Cloud ML and MLOps pipeline for **imbalanced claims/transaction exception detection and QA triage**, engineered to run on a **$0/month steady-state Google Cloud Always Free Tier** architecture (`us-central1`).

- **Live Portfolio & Engineering Logbook:** https://vraidaten.com/logbook/
- **Author:** Satyajit Chakraborty (transitioning from 9.5 years in TCS financial services, UK pension, and US healthcare operations into Cloud Engineering, Cloud ML, and MLOps)

## Architecture Highlights
- **Data & Low-Code Baseline:** Deterministic, zero-PII synthetic QA generator (`src/claims_triage/`) → Cloud Storage (`us-central1`) → BigQuery & BigQuery ML baseline model.
- **Custom ML & Quality Gate:** Python 3.13 (`uv`) + scikit-learn / XGBoost with stratified cross-validation, PR-AUC evaluation, threshold calibration, and an automated baseline-beat promotion gate.
- **Containerized Serving & Canary Rollout:** Multi-stage non-root Docker image (FastAPI `/health` + `/predict`) → Artifact Registry → Cloud Run (`min-instances=0`, `max-instances=2`, 90/10 canary traffic splitting).
- **Observability & Drift Detection:** Structured JSON telemetry in Cloud Logging + scheduled Cloud Run Job computing feature drift (PSI / KS-test) into BigQuery.
- **DevSecOps & IaC:** Terraform infrastructure provisioning, keyless GitHub Actions CD via Workload Identity Federation (WIF), Ruff, Mypy, Bandit, `pip-audit`, `detect-secrets`, Pytest (`>=80%` coverage), and Trivy.

## Engineering Documentation (`docs/`)
- `docs/requirements.md` — Functional, statistical, FinOps, and security specifications.
- `docs/architecture.md` — End-to-end topology, service boundaries, and IAM least-privilege map.
- `docs/security-and-threat-model.md` — Threat model, keyless auth, PII ban, and supply-chain controls.
- `docs/data-governance.md` — Synthetic data design, zero-PII guarantees, class imbalance, and drift injection.
- `docs/runbook-and-finops.md` — Budget alerts, operational triage, and `terraform destroy` teardown runbook.
- `docs/known-limitations.md` — Explicit engineering tradeoffs and scope boundaries.
- `docs/decisions/` — Architectural Decision Records (ADRs).
- `docs/hurdles/` — Real-world Operational Hurdles and Root-Cause Logs.

## Local Quickstart & Phase 1 CLI
Run the full local DevSecOps quality and security suite:
- `make all` — Runs `ruff`, `mypy --strict`, `bandit`, `pip-audit`, `detect-secrets`, and `pytest` (`>=80%` coverage gate).

Generate reproducible training and covariate-shifted evaluation datasets locally (output in `data/`, excluded from Git):
- `uv run --isolated python -m claims_triage.generate --n-samples 10000 --seed 42 --output data/claims_train.csv`
- `uv run --isolated python -m claims_triage.generate --n-samples 2000 --seed 99 --drift --output data/claims_drift.csv`

Run gated Phase 2 model training, threshold calibration, and SHA-256 artifact export locally (output in `artifacts/`, excluded from Git):
- `uv run --isolated python -m claims_triage.train --input data/claims_train.csv --output-dir artifacts --seed 42`
