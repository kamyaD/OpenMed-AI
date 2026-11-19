import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.db.models import Max
from django.utils import timezone

from inventory.models import Facility, InventorySnapshot, Medicine
from inventory.quality import quality_report
from ml.evaluation import classification_metrics
from ml.models import BatchPredictionJob, ModelVersion, PredictionRecord

from .config import cfg
from .models import MonitoringReport
from .risk import categorize, get_policy

DRIFT_FEATURES = ["stock_on_hand", "days_of_stock", "coverage_ratio", "cons_avg_14", "cons_trend_7_28",
                  "lead_time_mean", "stock_change_7d", "obs_days_28"]

# Alert rules, thresholds and response ownership. Served at GET /monitoring/rules/ and in docs/RUNBOOK.md.
RULES = {
    "data_freshness": {
        "rule": "WARNING if >10% of active facilities have no stock record within FRESHNESS_ALERT_DAYS; ALERT if >20%.",
        "owner": "Data engineer (ingestion owner)",
        "response": "Check upstream feeds and recent ingestion batches; contact stale facilities; re-ingest."},
    "data_completeness": {
        "rule": "ALERT if overall 28-day stock-record completeness is below QUALITY_COMPLETENESS_THRESHOLD.",
        "owner": "Data engineer with analyst",
        "response": "Review /quality/report/ by facility and source; agree remediation with regional supply managers."},
    "prediction_freshness": {
        "rule": "ALERT if there is no batch for the active model, or the latest batch is older than FRESHNESS_ALERT_DAYS.",
        "owner": "Platform engineer",
        "response": "Check the daily scheduler (run_daily) logs, run it manually, and confirm an active model exists."},
    "prediction_distribution": {
        "rule": "WARNING if the batch alert rate is >1.5x or <0.5x the alert rate seen on the model's test period.",
        "owner": "Analyst",
        "response": "Inspect feature drift and recent data-quality issues before trusting the alerts."},
    "feature_drift": {
        "rule": "WARNING if any key feature has PSI > 0.2 against training data; ALERT if 3 or more do.",
        "owner": "Analyst, escalated to the project sponsor",
        "response": "Investigate policy, supplier or reporting changes; consider retraining (lifecycle/status)."},
    "model_performance": {
        "rule": "ALERT if recall on realised outcomes is >10 points below test recall (needs >=30 positive outcomes).",
        "owner": "Analyst, escalated to the project sponsor",
        "response": "Review realised misses; roll back (lifecycle/rollback) or retrain; document the decision."},
    "model_age": {
        "rule": "WARNING if the active model is older than MODEL_MAX_AGE_DAYS.",
        "owner": "Analyst",
        "response": "Schedule retraining and evaluation; promotion still needs administrator approval."},
}


def psi(expected, actual, bins=10):
    e, a = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) < 50 or len(a) < 50:
        return None
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, edges)[0] / len(e), 1e-4, None)
    pa = np.clip(np.histogram(a, edges)[0] / len(a), 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(mv, feature_rows):
    if mv is None or mv.dataset is None or not Path(mv.dataset.file_path).exists() or len(feature_rows) < 50:
        return None
    ref = pd.read_parquet(mv.dataset.file_path, columns=["date"] + DRIFT_FEATURES)
    ref = ref[pd.to_datetime(ref["date"]) <= pd.Timestamp(mv.dataset.split["train_end"])]
    if len(ref) > 100_000:
        ref = ref.sample(100_000, random_state=0)
    cur = pd.DataFrame(feature_rows).reindex(columns=DRIFT_FEATURES).astype(float)
    out = {}
    for k in DRIFT_FEATURES:
        v = psi(ref[k].to_numpy(dtype=float), cur[k].to_numpy(dtype=float))
        out[k] = None if v is None else round(v, 4)
    return out


def realized_performance(mv, today, facility_ids=None, window_days=60):
    """Compare past predictions with what actually happened once the horizon has elapsed."""
    H, min_obs = settings.MSP["HORIZON_DAYS"], settings.MSP["MIN_FUTURE_OBS"]
    latest = InventorySnapshot.objects.aggregate(d=Max("date"))["d"]
    if mv is None or latest is None:
        return {"status": "no_data", "n": 0}
    limit = min(today, latest) - timedelta(days=H)
    qs = PredictionRecord.objects.filter(model=mv, model_applied=True, as_of_date__lte=limit,
                                         as_of_date__gte=limit - timedelta(days=window_days))
    if facility_ids is not None:
        qs = qs.filter(facility_id__in=list(facility_ids))
    preds = {}
    for f, m, d, p in qs.order_by("created_at").values_list("facility_id", "medicine_id", "as_of_date", "probability"):
        preds[(f, m, d)] = p
    if not preds:
        return {"status": "no_labelled_predictions", "n": 0}
    lo = min(k[2] for k in preds) + timedelta(days=1)
    hi = max(k[2] for k in preds) + timedelta(days=H)
    pairs = {(k[0], k[1]) for k in preds}
    sq = InventorySnapshot.objects.filter(date__range=(lo, hi))
    if facility_ids is not None:
        sq = sq.filter(facility_id__in=list(facility_ids))
    stock = {(f, m, d): s for f, m, d, s in sq.values_list("facility_id", "medicine_id", "date", "stock_on_hand")
             if (f, m) in pairs}
    y, p, unverified = [], [], 0
    for (f, m, d), prob in preds.items():
        seen = [s for s in (stock.get((f, m, d + timedelta(days=k))) for k in range(1, H + 1)) if s is not None]
        if any(s <= 0 for s in seen):
            y.append(1)
        elif len(seen) >= min_obs:
            y.append(0)
        else:
            unverified += 1
            continue
        p.append(prob)
    if not y:
        return {"status": "no_verified_labels", "n": 0, "unverified_excluded": unverified}
    out = classification_metrics(y, p, mv.threshold)
    out.update(status="ok", unverified_excluded=unverified)
    return out


def run_monitoring(user=None, as_of=None):
    cfgm = settings.MSP
    today, fresh = as_of or timezone.localdate(), cfgm["FRESHNESS_ALERT_DAYS"]
    alerts, metrics = [], {}

    def add(level, check, message, **extra):
        alerts.append({"level": level, "check": check, "message": message, "owner": RULES[check]["owner"],
                       "response": RULES[check]["response"], **extra})

    mv = ModelVersion.objects.filter(status=ModelVersion.Status.ACTIVE).select_related("dataset").first()

    # 1. data freshness
    last = {r["facility_id"]: r["d"] for r in InventorySnapshot.objects.values("facility_id").annotate(d=Max("date"))}
    active = list(Facility.objects.filter(is_active=True).values_list("id", "code"))
    stale = [(code, (today - last[fid]).days if fid in last else None) for fid, code in active
             if fid not in last or (today - last[fid]).days > fresh]
    stale.sort(key=lambda t: -(t[1] if t[1] is not None else 10 ** 6))
    share = len(stale) / len(active) if active else 0.0
    metrics["data_freshness"] = {"active_facilities": len(active), "stale_facilities": len(stale),
                                 "stale_share": round(share, 4),
                                 "worst": [{"facility_code": c, "days_since_last_record": d} for c, d in stale[:10]]}
    limit = cfg("STALE_FACILITY_SHARE_ALERT")
    if share > limit:
        add("ALERT", "data_freshness", f"{share:.0%} of active facilities have no stock record in {fresh} days.")
    elif share > limit / 2:
        add("WARNING", "data_freshness", f"{share:.0%} of active facilities have no stock record in {fresh} days.")

    # 2. missingness / completeness
    n_med = Medicine.objects.filter(is_essential=True).count()
    got = InventorySnapshot.objects.filter(date__range=(today - timedelta(days=27), today), facility__is_active=True,
                                           medicine__is_essential=True).count()
    expected = len(active) * n_med * 28
    completeness = round(got / expected, 4) if expected else None
    metrics["data_completeness"] = {
        "completeness_28d": completeness, "threshold": cfgm["QUALITY_COMPLETENESS_THRESHOLD"],
        "facilities_below_threshold": quality_report(28, "facility")["summary"]["below_threshold"]}
    if completeness is not None and completeness < cfgm["QUALITY_COMPLETENESS_THRESHOLD"]:
        add("ALERT", "data_completeness", f"28-day completeness is {completeness:.0%}.")

    # 3. predictions: freshness, distribution, feature drift
    preds = []
    if mv is None:
        add("ALERT", "prediction_freshness", "No active model.")
    else:
        batch = BatchPredictionJob.objects.filter(model=mv).exclude(status="FAILED").order_by("-created_at").first()
        if batch is None or (today - batch.as_of_date).days > fresh:
            add("ALERT", "prediction_freshness", "No recent batch predictions for the active model.")
        if batch is not None:
            preds = list(PredictionRecord.objects.filter(batch=batch).values(
                "probability", "model_applied", "predicted_stockout", "features"))
            applied = [x for x in preds if x["model_applied"]]
            policy, ref = get_policy(), (mv.metrics.get("test") or {}).get("alert_rate")
            rate = float(np.mean([x["predicted_stockout"] for x in applied])) if applied else None
            shares = {}
            for x in preds:
                c = categorize(x["probability"], x["model_applied"], policy)
                shares[c] = shares.get(c, 0) + 1
            metrics["prediction_distribution"] = {
                "batch": batch.pk, "as_of_date": batch.as_of_date, "predictions": len(preds),
                "alert_rate": None if rate is None else round(rate, 4), "reference_alert_rate": ref,
                "mean_probability": round(float(np.mean([x["probability"] for x in applied])), 4) if applied else None,
                "risk_category_counts": shares}
            if rate is not None and ref:
                ratio = rate / ref
                if ratio > cfg("ALERT_RATE_RATIO") or ratio < 1 / cfg("ALERT_RATE_RATIO"):
                    add("WARNING", "prediction_distribution",
                        f"Alert rate {rate:.0%} vs {ref:.0%} on the test period (ratio {ratio:.2f}).")
            drift = feature_drift(mv, [x["features"] or {} for x in applied])
            metrics["feature_drift"] = {"psi": drift, "psi_alert": cfgm["PSI_ALERT"]}
            if drift:
                hi = [k for k, v in drift.items() if v is not None and v > cfgm["PSI_ALERT"]]
                if len(hi) >= cfg("DRIFT_FEATURES_ALERT_COUNT"):
                    add("ALERT", "feature_drift", f"{len(hi)} features drifted (PSI > {cfgm['PSI_ALERT']}): {', '.join(hi)}.")
                elif hi:
                    add("WARNING", "feature_drift", f"Drift on: {', '.join(hi)}.")

    # 4. model performance on realised outcomes
    perf = realized_performance(mv, today)
    metrics["model_performance"] = perf
    if mv is not None and perf.get("status") == "ok":
        ref_recall = (mv.metrics.get("test") or {}).get("recall")
        perf["reference_test_recall"] = ref_recall
        if perf.get("positives", 0) >= cfg("MIN_LABELLED_POSITIVES") and perf.get("recall") is not None \
                and ref_recall is not None and ref_recall - perf["recall"] > cfg("RECALL_DROP_ALERT"):
            add("ALERT", "model_performance",
                f"Recall on realised outcomes is {perf['recall']:.2f} vs {ref_recall:.2f} on the test period.")
        elif perf.get("positives", 0) < cfg("MIN_LABELLED_POSITIVES"):
            perf["note"] = "Too few realised stockouts to judge recall reliably."

    # 5. model age
    if mv is not None and mv.trained_at:
        age = (today - mv.trained_at.date()).days
        metrics["model_age_days"] = age
        if age > cfg("MODEL_MAX_AGE_DAYS"):
            add("WARNING", "model_age", f"Active model is {age} days old.")

    levels = {a["level"] for a in alerts}
    status = "ALERT" if "ALERT" in levels else "WARNING" if "WARNING" in levels else "OK"
    return MonitoringReport.objects.create(
        as_of_date=today, model=mv, status=status,
        alerts=json.loads(json.dumps(alerts, default=str)), metrics=json.loads(json.dumps(metrics, default=str)),
        created_by=user if getattr(user, "is_authenticated", False) else None)