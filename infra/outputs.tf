output "wif_provider_name" {
  description = "Full resource name of the GitHub OIDC Workload Identity Provider."
  value       = google_iam_workload_identity_pool_provider.github_provider.name
}

output "ci_service_account_email" {
  description = "Service account email impersonated by GitHub Actions CI/CD."
  value       = google_service_account.ci_deployer.email
}

output "serving_service_account_email" {
  description = "Service account email attached to the online Cloud Run serving service."
  value       = google_service_account.serving_runtime.email
}

output "drift_service_account_email" {
  description = "Service account email attached to the Cloud Run drift detection job."
  value       = google_service_account.drift_job.email
}

output "artifacts_bucket_name" {
  description = "GCS bucket name storing MLOps artifacts and datasets."
  value       = google_storage_bucket.mlops_artifacts.name
}

output "artifact_registry_url" {
  description = "Artifact Registry Docker repository base URL."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "bigquery_dataset_id" {
  description = "BigQuery dataset ID for claims triage tables and ML/drift workflows."
  value       = google_bigquery_dataset.claims_triage.dataset_id
}