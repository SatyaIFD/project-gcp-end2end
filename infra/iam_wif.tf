# ==============================================================================
# A. Dedicated Service Accounts (Separation of Duties)
# ==============================================================================

resource "google_service_account" "ci_deployer" {
  account_id   = "claims-triage-ci"
  display_name = "Claims Triage CI/CD Deployer"
  description  = "Impersonated via WIF by GitHub Actions to push images, upload artifacts, and deploy Cloud Run."

  depends_on = [google_project_service.enabled_apis]
}

resource "google_service_account" "serving_runtime" {
  account_id   = "claims-triage-serving"
  display_name = "Claims Triage Online Serving Runtime"
  description  = "Attached to Cloud Run FastAPI serving service; read-only GCS access and no BigQuery access."

  depends_on = [google_project_service.enabled_apis]
}

resource "google_service_account" "drift_job" {
  account_id   = "claims-triage-drift"
  display_name = "Claims Triage Drift Detection Job"
  description  = "Attached to scheduled Cloud Run drift job; read-only GCS and dataset-scoped BigQuery access."

  depends_on = [google_project_service.enabled_apis]
}

# ==============================================================================
# B. Least-Privilege IAM Grants (Additive *_iam_member Only)
# ==============================================================================

# --- ci_deployer permissions ---
resource "google_artifact_registry_repository_iam_member" "ci_ar_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.docker_repo.location
  repository = google_artifact_registry_repository.docker_repo.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.ci_deployer.email}"
}

resource "google_storage_bucket_iam_member" "ci_gcs_object_admin" {
  bucket = google_storage_bucket.mlops_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ci_deployer.email}"
}

resource "google_project_iam_member" "ci_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.ci_deployer.email}"
}

resource "google_project_iam_member" "ci_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.ci_deployer.email}"
}

resource "google_bigquery_dataset_iam_member" "ci_bq_data_editor" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.claims_triage.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.ci_deployer.email}"
}

# Scoped actAs so CI can attach runtime SAs to Cloud Run without broad impersonation
resource "google_service_account_iam_member" "ci_act_as_serving" {
  service_account_id = google_service_account.serving_runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.ci_deployer.email}"
}

resource "google_service_account_iam_member" "ci_act_as_drift" {
  service_account_id = google_service_account.drift_job.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.ci_deployer.email}"
}

# --- serving_runtime permissions (read-only GCS + Cloud Logging only) ---
resource "google_storage_bucket_iam_member" "serving_gcs_object_viewer" {
  bucket = google_storage_bucket.mlops_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.serving_runtime.email}"
}

resource "google_project_iam_member" "serving_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.serving_runtime.email}"
}

# --- drift_job permissions (read-only GCS + dataset-scoped BQ write + BQ job + logs) ---
resource "google_storage_bucket_iam_member" "drift_gcs_object_viewer" {
  bucket = google_storage_bucket.mlops_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.drift_job.email}"
}

resource "google_bigquery_dataset_iam_member" "drift_bq_data_editor" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.claims_triage.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.drift_job.email}"
}

resource "google_project_iam_member" "drift_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.drift_job.email}"
}

resource "google_project_iam_member" "drift_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.drift_job.email}"
}

# ==============================================================================
# C. Keyless Workload Identity Federation (GitHub Actions -> ci_deployer)
# ==============================================================================

resource "google_iam_workload_identity_pool" "github_pool" {
  workload_identity_pool_id = "github-actions-pool"
  display_name              = "GitHub Actions Pool"
  description               = "Workload Identity Pool for keyless GitHub Actions authentication."

  depends_on = [google_project_service.enabled_apis]
}

resource "google_iam_workload_identity_pool_provider" "github_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc-provider"
  display_name                       = "GitHub OIDC Provider"

  attribute_mapping = {
    "google.subject"             = "assertion.sub"
    "attribute.actor"            = "assertion.actor"
    "attribute.repository"       = "assertion.repository"
    "attribute.repository_owner" = "assertion.repository_owner"
    "attribute.ref"              = "assertion.ref"
  }

  attribute_condition = "assertion.repository == '${var.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "wif_ci_impersonation" {
  service_account_id = google_service_account.ci_deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/${var.github_repo}"
}