from pathlib import Path

import joblib
import pandas as pd
from django.db import transaction
from django.utils import timezone
from rest_framework.exceptions import ValidationError

from accounts.audit import log_action
from ml.evaluation import classification_metrics
from ml.models import ModelVersion
from ml.splits import split_masks

RETRAIN_CRITERIA = [
    "Model performance: recall on realised outcomes is more than 10 points below the test recall (needs >= 30 positive outcomes).",
    "Feature drift: PSI > 0.2 on 3 or more key features against the training data.",
    "Model age: the active model is older than 90 days.",
    "Manual request from the project sponsor (retrain_models --force).",
]


def retrain_recommendation(report):
    reasons = [a["message"] for a in (report.alerts if report else [])
               if (a["check"] in ("model_performance", "feature_drift") and a["level"] == "ALERT")
               or a["check"] == "model_age"]
    return {"recommended": bool(reasons), "reasons": reasons, "criteria": RETRAIN_CRITERIA,
            "based_on_report": report.pk if report else None,
            "report_date": report.as_of_date if report else None,
            "approval": "Retraining only creates CANDIDATE models. An administrator must promote one with a written rationale."}


def rollback(user, rationale, target_id=None):
    if not str(rationale or "").strip():
        raise ValidationError({"rationale": "A written rationale is required to roll back."})
    with transaction.atomic():
        current = ModelVersion.objects.select_for_update().filter(status=ModelVersion.Status.ACTIVE).first()
        qs = ModelVersion.objects.filter(status=ModelVersion.Status.ARCHIVED, promoted_at__isnull=False) \
            .exclude(artifact_path="")
        if target_id:
            qs = qs.filter(pk=target_id)
        target = next((m for m in qs.order_by("-promoted_at") if Path(m.artifact_path).exists()), None)
        if target is None:
            raise ValidationError("No previously approved model with a loadable artifact is available.")
        if current:
            current.status = ModelVersion.Status.ARCHIVED
            current.save(update_fields=["status"])
        target.status, target.promoted_by, target.promoted_at = ModelVersion.Status.ACTIVE, user, timezone.now()
        target.review_note = f"Rollback: {rationale}"
        target.save()
    log_action(user, "model.rolled_back", target, rationale=rationale, from_model=current.pk if current else None)
    return target


def evaluate_on_dataset(mv, dataset):
    """Evaluate an existing model on the test period of another dataset (fair comparison with new candidates)."""
    bundle = joblib.load(mv.artifact_path)
    df = pd.read_parquet(dataset.file_path)
    df["date"] = pd.to_datetime(df["date"])
    test = df[split_masks(df["date"], dataset.split)["test"]]
    return classification_metrics(test["y"], bundle.predict_proba(test), bundle.threshold)