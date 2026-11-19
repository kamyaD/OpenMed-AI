import math
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.db import transaction
from django.db.models import Max
from django.utils import timezone
from rest_framework.exceptions import APIException

from inventory.models import InventorySnapshot

from .features import build_features
from .models import BatchPredictionJob, ModelVersion, PredictionRecord

_cache = {}


class NoActiveModel(APIException):
    status_code = 503
    default_detail = "No active model. An administrator must promote a model version first."
    default_code = "no_active_model"


class PredictionError(APIException):
    status_code = 422
    default_code = "prediction_error"


def get_active():
    mv = ModelVersion.objects.filter(status=ModelVersion.Status.ACTIVE).first()
    if mv is None or not mv.artifact_path or not Path(mv.artifact_path).exists():
        raise NoActiveModel()
    key = (mv.pk, mv.artifact_path)
    if key not in _cache:
        _cache.clear()
        _cache[key] = joblib.load(mv.artifact_path)
    return mv, _cache[key]


def _clean(v):
    if v is None or isinstance(v, str):
        return v
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def predict_single(user, facility, medicine, as_of=None, overrides=None):
    cfg = settings.MSP
    mv, bundle = get_active()
    overrides = overrides or {}
    cols = bundle.numeric + bundle.categorical

    if as_of is None:
        as_of = InventorySnapshot.objects.filter(facility=facility, medicine=medicine).aggregate(d=Max("date"))["d"]
        if as_of is None:
            if "stock_on_hand" not in overrides:
                raise PredictionError("No inventory data exist for this facility and medicine.")
            as_of = timezone.localdate()

    rows = build_features(as_of, as_of, [facility.id], [medicine.id])
    if rows.empty and "stock_on_hand" not in overrides:
        raise PredictionError("No stock observation within 3 days of the as-of date. Ingest fresh data, or "
                              "supply at least 'stock_on_hand' in 'features'.")
    features = {"facility_level": float(facility.level), "is_urban": float(facility.is_urban),
                "medicine_category": medicine.category or "Unknown"}
    if not rows.empty:
        features.update({c: rows.iloc[0][c] for c in cols})
    features.update(overrides)

    missing = [c for c in bundle.numeric if c not in features or _clean(features[c]) is None]
    X = pd.DataFrame([{c: (np.nan if _clean(features.get(c)) is None else features[c]) for c in cols}])
    X[bundle.numeric] = X[bundle.numeric].astype(float)

    warnings = []
    age = (timezone.localdate() - as_of).days
    if age > cfg["FRESHNESS_ALERT_DAYS"]:
        warnings.append(f"Latest data is {age} days old.")
    if overrides:
        warnings.append("Prediction uses user-supplied feature overrides (what-if) and is not based solely on recorded data.")
    if missing:
        warnings.append(f"{len(missing)} feature(s) unavailable and imputed with training medians: {', '.join(missing)}.")
    if (_clean(features.get("days_since_stock_obs")) or 0) > 0:
        warnings.append("Latest stock observation is not from the as-of date; it was carried forward.")
    if (_clean(features.get("obs_days_28")) or 28) < 21:
        warnings.append("Low data completeness over the last 28 days; treat this prediction with caution.")

    stock = _clean(features.get("stock_on_hand"))
    out_of_stock = stock is not None and stock <= 0
    if out_of_stock:
        proba, applied = 1.0, False
        warnings.append("Facility is already out of stock; model not applied.")
    else:
        proba, applied = float(bundle.predict_proba(X)[0]), True

    return PredictionRecord.objects.create(
        model=mv, facility=facility, medicine=medicine, as_of_date=as_of, horizon_days=cfg["HORIZON_DAYS"],
        probability=proba, predicted_stockout=bool(out_of_stock or proba >= mv.threshold), threshold=mv.threshold,
        probability_calibrated=bool(applied and mv.calibration.get("probability_calibrated")),
        model_applied=applied, features={c: _clean(features.get(c)) for c in cols}, warnings=warnings,
        created_by=user if getattr(user, "is_authenticated", False) else None)


def run_batch(user, as_of=None, facility_ids=None, medicine_ids=None):
    cfg = settings.MSP
    mv, bundle = get_active()
    cols = bundle.numeric + bundle.categorical
    if as_of is None:
        as_of = InventorySnapshot.objects.aggregate(d=Max("date"))["d"]
        if as_of is None:
            raise PredictionError("No inventory data available.")

    job = BatchPredictionJob.objects.create(
        model=mv, as_of_date=as_of, created_by=user if getattr(user, "is_authenticated", False) else None)
    qs = InventorySnapshot.objects.filter(date__range=(as_of - timedelta(days=30), as_of),
                                          facility__is_active=True, medicine__is_essential=True)
    if facility_ids is not None:
        qs = qs.filter(facility_id__in=facility_ids)
    if medicine_ids is not None:
        qs = qs.filter(medicine_id__in=medicine_ids)
    expected = set(qs.values_list("facility_id", "medicine_id").distinct())
    fac_ids = sorted({f for f, _ in expected})

    records, got = [], set()
    for i in range(0, len(fac_ids), 50):
        feats = build_features(as_of, as_of, fac_ids[i:i + 50], medicine_ids)
        if feats.empty:
            continue
        out = feats["stock_on_hand"].to_numpy() <= 0
        proba = np.ones(len(feats))
        if (~out).any():
            proba[~out] = bundle.predict_proba(feats.loc[~out])
        for rec, p, o in zip(feats[cols + ["facility_id", "medicine_id"]].to_dict("records"), proba, out):
            key = (int(rec["facility_id"]), int(rec["medicine_id"]))
            if key not in expected:
                continue
            got.add(key)
            records.append(PredictionRecord(
                model=mv, facility_id=key[0], medicine_id=key[1], as_of_date=as_of,
                horizon_days=cfg["HORIZON_DAYS"], probability=float(p), predicted_stockout=bool(o or p >= mv.threshold),
                threshold=mv.threshold, probability_calibrated=bool((not o) and mv.calibration.get("probability_calibrated")),
                model_applied=not o, features={c: _clean(rec[c]) for c in cols},
                warnings=["Already out of stock; model not applied."] if o else [], batch=job, created_by=job.created_by))

    missing = sorted(expected - got)
    with transaction.atomic():
        PredictionRecord.objects.bulk_create(records, batch_size=2000)
        job.total, job.succeeded, job.failed = len(expected), len(records), len(missing)
        job.errors = [{"facility_id": f, "medicine_id": m, "error": "No stock observation within 3 days of as-of date"}
                      for f, m in missing[:200]]
        job.status = (BatchPredictionJob.Status.FAILED if not records else
                      BatchPredictionJob.Status.COMPLETED_WITH_ERRORS if missing else BatchPredictionJob.Status.COMPLETED)
        job.finished_at = timezone.now()
        job.save()
    return job