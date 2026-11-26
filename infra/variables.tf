variable "project_id" {
  description = "Target GCP project ID (no default to enforce explicit environment targeting)."
  type        = string
}

variable "region" {
  description = "GCP region for Always Free Tier co-location (us-central1)."
  type        = string
  default     = "us-central1"
}

variable "github_repo" {
  description = "GitHub repository in owner/repo format for Workload Identity Federation."
  type        = string
  default     = "SatyaIFD/project-gcp-end2end"
}

variable "artifact_repo_id" {
  description = "Artifact Registry Docker repository ID."
  type        = string
  default     = "claims-triage"
}

variable "bq_dataset_id" {
  description = "BigQuery dataset ID co-located in the primary region."
  type        = string
  default     = "claims_triage"
}