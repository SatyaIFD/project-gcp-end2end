terraform {
  required_version = ">= 1.10.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0.0"
    }
  }
}

# Local state is intentionally retained (infra/terraform.tfstate, gitignored)
# to avoid circular bootstrap dependencies before the GCS bucket exists.
provider "google" {
  project = var.project_id
  region  = var.region
}