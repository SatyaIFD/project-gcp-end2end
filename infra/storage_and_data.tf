resource "google_storage_bucket" "mlops_artifacts" {
  name                        = "${var.project_id}-mlops-artifacts"
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 3
      with_state         = "ARCHIVED"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_bigquery_dataset" "claims_triage" {
  dataset_id                 = var.bq_dataset_id
  location                   = var.region
  delete_contents_on_destroy = true

  depends_on = [google_project_service.enabled_apis]
}

resource "google_artifact_registry_repository" "docker_repo" {
  location               = var.region
  repository_id          = var.artifact_repo_id
  format                 = "DOCKER"
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-latest-2-versions"
    action = "KEEP"

    most_recent_versions {
      keep_count = 2
    }
  }

  cleanup_policies {
    id     = "delete-older-than-24h"
    action = "DELETE"

    condition {
      tag_state  = "ANY"
      older_than = "86400s"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}