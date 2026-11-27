# Hurdle 006: Default Compute Engine `roles/editor` Service Account, WIF Branch Scope & Free-Trial BigQuery Quota Lock

- **Symptoms:**
  1. Running a pre-deployment security audit on `claims-triage-mlops` revealed that `862331503448-compute@developer.gserviceaccount.com` held project-wide `roles/editor` and four unused APIs (`compute.googleapis.com`, `oslogin.googleapis.com`, `containerregistry.googleapis.com`, `backupdr.googleapis.com`) were enabled.
  2. The initial Workload Identity Federation (WIF) provider condition checked only `assertion.repository`, which could allow OIDC token exchange from unmerged pull-request branches rather than protected `main` only.
  3. Attempting to lower the project-level BigQuery `Query usage per day` quota in Google Cloud Console failed because Free Trial projects lock that quota at 200 TiB.
- **Root Cause:**
  - Setting `gcloud config set compute/region us-central1` during initial CLI setup prompted to enable `compute.googleapis.com`, which automatically provisions the legacy default Compute Engine service account with broad `roles/editor` permissions.
  - Scoping WIF strictly to repository identity without a branch `ref` predicate leaves a wider trust boundary than needed for production CD.
  - Google Cloud disables custom quota reductions on un-upgraded Free Trial accounts, relying on client/job-level byte limits instead.
- **Resolution:**
  1. Revoked `roles/editor` from `862331503448-compute@developer.gserviceaccount.com`, disabled that service account, disabled the four unused APIs, and switched `gcloud` defaults from `compute/region` to `run/region us-central1`.
  2. Tightened `google_iam_workload_identity_pool_provider.github_provider` in `infra/iam_wif.tf` to enforce `assertion.repository == 'SatyaIFD/project-gcp-end2end' && assertion.ref == 'refs/heads/main'` and verified zero drift via `terraform apply` and `terraform plan -detailed-exitcode`.
  3. Adopted server-side `--maximum_bytes_billed=104857600` (100 MiB) on every BigQuery CLI and Python job so oversized queries abort pre-execution at zero cost, paired with IAM-private Cloud Run deployment (`--no-allow-unauthenticated`).
