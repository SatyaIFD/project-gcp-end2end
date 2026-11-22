# Security Policy

## Core Security Invariants
- **Zero Static Cloud Keys:** No GCP service-account JSON keys or `.env` files are ever created, downloaded, or committed. Local access uses short-lived Application Default Credentials (ADC), and CI/CD uses keyless GitHub OIDC Workload Identity Federation (WIF).
- **Zero Real PII:** All datasets are generated deterministically from synthetic distributions. No real customer, healthcare (HIPAA), or pension (GDPR) data is ever processed or stored.
- **Automated Supply-Chain & Static Analysis:** Every pull request is gated by `ruff`, `mypy`, `bandit`, `pip-audit`, `detect-secrets`, `pytest`, and Trivy filesystem/container scanning.
- **Hardened Containers:** Production Docker images use multi-stage slim builds running strictly as a non-root user.

## Reporting a Vulnerability
If you discover a security concern in this repository, please open a private security advisory via GitHub Security Advisories rather than posting sensitive exploit details in a public issue.
