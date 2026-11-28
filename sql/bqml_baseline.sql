CREATE OR REPLACE MODEL `${GCP_PROJECT_ID}.${BQ_DATASET}.bqml_logistic_baseline`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['is_exception'],
  auto_class_weights = TRUE,
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2,
  l2_reg = 1.0
) AS
SELECT
  claim_amount,
  processing_age_hours,
  reconciliation_variance,
  prior_qa_flags,
  document_count,
  submission_channel,
  auth_doc_verified,
  domain_type,
  is_exception
FROM `${GCP_PROJECT_ID}.${BQ_DATASET}.claims_train`;

SELECT
  'holdout_20pct' AS eval_split,
  ROUND(roc_auc, 4) AS roc_auc,
  ROUND(precision, 4) AS precision,
  ROUND(recall, 4) AS recall,
  ROUND(f1_score, 4) AS f1_score,
  ROUND(log_loss, 4) AS log_loss
FROM ML.EVALUATE(
  MODEL `${GCP_PROJECT_ID}.${BQ_DATASET}.bqml_logistic_baseline`
)

UNION ALL

SELECT
  'covariate_drift' AS eval_split,
  ROUND(roc_auc, 4) AS roc_auc,
  ROUND(precision, 4) AS precision,
  ROUND(recall, 4) AS recall,
  ROUND(f1_score, 4) AS f1_score,
  ROUND(log_loss, 4) AS log_loss
FROM ML.EVALUATE(
  MODEL `${GCP_PROJECT_ID}.${BQ_DATASET}.bqml_logistic_baseline`,
  (
    SELECT
      claim_amount,
      processing_age_hours,
      reconciliation_variance,
      prior_qa_flags,
      document_count,
      submission_channel,
      auth_doc_verified,
      domain_type,
      is_exception
    FROM `${GCP_PROJECT_ID}.${BQ_DATASET}.claims_drift`
  )
);