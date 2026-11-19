# MSP runbook

## Daily operation
`python manage.py run_daily` (batch predictions -> redistribution suggestions -> monitoring). Scheduled once a day after data ingestion.

## Alert ownership (also served at GET /api/v1/monitoring/rules/)
| Check | Trigger | Owner | First response |
|---|---|---|---|
| data_freshness | >20% of facilities without a record in 3 days | Data engineer | Check feeds and ingestion batches |
| data_completeness | 28-day completeness < 80% | Data engineer + analyst | Review /quality/report/ |
| prediction_freshness | No batch in 3 days or no active model | Platform engineer | Check scheduler logs, run run_daily |
| prediction_distribution | Alert rate >1.5x or <0.5x test-period rate | Analyst | Check drift and data quality |
| feature_drift | PSI > 0.2 on >= 3 features | Analyst -> sponsor | Investigate, consider retraining |
| model_performance | Recall >10 pts below test recall (>= 30 positives) | Analyst -> sponsor | Roll back or retrain |
| model_age | Model older than 90 days | Analyst | Schedule retraining |

## Procedures
- **Roll back:** `POST /api/v1/lifecycle/rollback/` (admin, rationale required). Restores the most recently approved archived model.
- **Retrain:** `python manage.py retrain_models` shows whether criteria are met; add `--execute` to build a dataset and train CANDIDATE models. An admin reviews `comparison_to_active` and promotes with a rationale.
- **Change risk bands:** `POST /api/v1/decision/risk-policy/` (admin, rationale required).
- **Models and libraries:** a saved model must be loaded with the same scikit-learn version it was trained with (see `library_versions` on the ModelVersion). Pin versions in requirements before deploying.

## Prohibited
No automatic ordering, stock transfer or supply reduction from predictions. All actions need authorised human approval.