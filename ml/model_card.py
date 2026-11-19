from django.conf import settings


def build_model_card(mv_name, algorithm, dataset, metrics, calibration, subgroup, threshold):
    cfg = settings.MSP
    return {
        "model": mv_name, "algorithm": algorithm,
        "intended_use": "Decision support for supply-chain managers: ranks facility-medicine combinations by "
                        f"risk of a stockout within {cfg['HORIZON_DAYS']} days so staff can investigate and act early.",
        "intended_users": ["Supply-chain managers", "Analysts", "Facility users (own facilities only)"],
        "not_intended_for": [
            "Fully automated ordering or stock transfers",
            "Evaluating or penalising individual facilities or staff",
            "Medicines or facilities that were not in the training data",
            "Clinical decisions about individual patients",
        ],
        "prohibited_autonomous_actions": [
            "Placing, cancelling or changing procurement orders without authorised human approval",
            "Moving stock between facilities without authorised human approval",
            "Reducing or withholding supply to a facility solely because of a prediction",
        ],
        "human_oversight": "Every operational action derived from a prediction (replenishment, redistribution) "
                           "requires approval by an authorised user, recorded with actor, time and rationale.",
        "escalation": "Flagged subgroup disparities, data-quality alerts or unexpected prediction drift are "
                      "escalated to the analyst team and the project sponsor before the model is promoted or kept active.",
        "training_data": {"dataset_id": dataset.pk, "rows": dataset.row_count, "positive_rate": dataset.positive_rate,
                          "period": f"{dataset.start_date} to {dataset.end_date}", "label": dataset.label_definition,
                          "exclusions": dataset.exclusions, "split": dataset.split},
        "operating_threshold": {"value": threshold,
                                "basis": f"Minimises expected cost with cost(false negative)={cfg['COST_FALSE_NEGATIVE']} "
                                         f"and cost(false positive)={cfg['COST_FALSE_POSITIVE']} (placeholders "
                                         "to be confirmed with stakeholders)."},
        "performance": {"validation": metrics.get("val_b"), "test": metrics.get("test")},
        "calibration": calibration,
        "subgroup_summary": {"flags": subgroup.get("flags", []), "requires_review": subgroup.get("requires_review")},
        "limitations": [
            "Trained on historical data; behaviour may shift with policy, supplier or disease-pattern changes.",
            "Stockout is defined as end-of-day stock of zero; intra-day stockouts are not observed.",
            "Rows with unverified labels or already-zero stock are excluded from training.",
            "Probabilities are only to be read as calibrated when calibration.probability_calibrated is true.",
            "Subgroups with few events cannot be reliably assessed.",
        ],
    }