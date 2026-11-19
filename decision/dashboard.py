import datetime as dt

from django.db.models import Q
from rest_framework.exceptions import ValidationError

from accounts.permissions import scope_queryset
from ml.models import BatchPredictionJob, PredictionRecord

from .risk import CATEGORIES, categorize, category_q, get_policy


def latest_batch():
    return BatchPredictionJob.objects.exclude(status=BatchPredictionJob.Status.FAILED).order_by("-created_at").first()


def _date(value, name):
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValidationError({name: "Use YYYY-MM-DD."})


def base_predictions(request, apply_risk=True):
    """Permission-scoped, filtered predictions. Used by the overview, filters and CSV export."""
    p, policy = request.query_params, get_policy()
    qs = scope_queryset(PredictionRecord.objects.select_related("facility", "medicine", "model"), request.user)
    lo, hi = p.get("as_of_from"), p.get("as_of_to")
    if p.get("batch"):
        try:
            qs = qs.filter(batch_id=int(p["batch"]))
        except ValueError:
            raise ValidationError({"batch": "Must be an integer."})
    elif lo or hi:
        if lo:
            qs = qs.filter(as_of_date__gte=_date(lo, "as_of_from"))
        if hi:
            qs = qs.filter(as_of_date__lte=_date(hi, "as_of_to"))
    else:
        b = latest_batch()
        qs = qs.filter(batch=b) if b else qs.none()
    if p.get("region"):
        qs = qs.filter(facility__region=p["region"])
    if p.get("facility_code"):
        qs = qs.filter(facility__code=p["facility_code"])
    if p.get("medicine_code"):
        qs = qs.filter(medicine__code=p["medicine_code"])
    if p.get("medicine_category"):
        qs = qs.filter(medicine__category=p["medicine_category"])
    if apply_risk:
        if p.get("risk"):
            cats = [c.strip().upper() for c in p["risk"].split(",") if c.strip()]
            bad = [c for c in cats if c not in CATEGORIES]
            if bad:
                raise ValidationError({"risk": f"Unknown categories {bad}. Allowed: {CATEGORIES}"})
            q = Q()
            for c in cats:
                q |= category_q(c, policy)
            qs = qs.filter(q)
        kind = p.get("kind", "all")
        if kind == "predicted":
            qs = qs.filter(model_applied=True, predicted_stockout=True)
        elif kind == "stocked_out":
            qs = qs.filter(model_applied=False)
        elif kind != "all":
            raise ValidationError({"kind": "Must be all, predicted or stocked_out."})
    return qs


def prediction_row(p, policy, today):
    f = p.features or {}
    return {
        "request_id": str(p.request_id), "facility_code": p.facility.code, "facility_name": p.facility.name,
        "region": p.facility.region, "medicine_code": p.medicine.code, "medicine_name": p.medicine.name,
        "medicine_category": p.medicine.category, "as_of_date": p.as_of_date,
        "prediction_age_days": (today - p.as_of_date).days, "horizon_days": p.horizon_days,
        "probability": round(p.probability, 4), "risk_category": categorize(p.probability, p.model_applied, policy),
        "predicted_stockout": p.predicted_stockout, "model_applied": p.model_applied,
        "probability_calibrated": p.probability_calibrated, "stock_on_hand": f.get("stock_on_hand"),
        "days_of_stock": f.get("days_of_stock"), "has_warnings": bool(p.warnings), "model_version": p.model.label,
    }


def csv_safe(v):
    """Neutralise spreadsheet formula injection."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v