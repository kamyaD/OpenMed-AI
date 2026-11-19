import math
from collections import defaultdict
from datetime import timedelta

from django.conf import settings
from django.db import transaction

from inventory.models import InventorySnapshot
from ml.models import BatchPredictionJob, PredictionRecord

from .config import cfg
from .coverage import coverage_from_features, num, pending_quantity_map
from .models import RedistributionSuggestion
from .risk import categorize, get_policy


def haversine_km(a, b):
    if None in (a.latitude, a.longitude, b.latitude, b.longitude):
        return None
    p1, p2 = math.radians(a.latitude), math.radians(b.latitude)
    dl = math.radians(b.longitude - a.longitude)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * 6371.0 * math.asin(math.sqrt(h)), 1)


def generate_suggestions(batch=None):
    """
    Candidate transfers from LOW-risk donors with surplus to STOCKED_OUT / CRITICAL / HIGH recipients.
    Nothing is executed: every suggestion starts as PENDING_REVIEW and needs authorised human approval.
    """
    if batch is None:
        batch = BatchPredictionJob.objects.exclude(status=BatchPredictionJob.Status.FAILED) \
            .order_by("-created_at").first()
    if batch is None:
        raise ValueError("No batch predictions exist. Run a batch prediction first.")

    cfgm = settings.MSP
    as_of, policy = batch.as_of_date, get_policy()
    min_shelf, transit, fresh = cfgm["MIN_REMAINING_SHELF_DAYS"], cfgm["TRANSIT_DAYS"], cfgm["FRESHNESS_ALERT_DAYS"]
    min_obs, max_km, max_donors = cfg("REDISTRIBUTION_MIN_DATA_DAYS"), cfg("REDISTRIBUTION_MAX_DISTANCE_KM"), \
        cfg("REDISTRIBUTION_MAX_DONORS")

    preds = list(PredictionRecord.objects.filter(batch=batch).select_related("facility", "medicine"))
    pending = pending_quantity_map(as_of)
    expiry = {}
    for fid, mid, exp in InventorySnapshot.objects.filter(date__range=(as_of - timedelta(days=fresh), as_of)) \
            .order_by("date").values_list("facility_id", "medicine_id", "earliest_expiry_date"):
        expiry[(fid, mid)] = exp

    groups = defaultdict(lambda: {"recipients": [], "donors": []})
    excluded = defaultdict(int)
    for p in preds:
        cov = coverage_from_features(p.features, pending.get((p.facility_id, p.medicine_id), 0.0))
        cat = categorize(p.probability, p.model_applied, policy)
        daily, stock, gap, target = (cov["avg_daily_consumption"], cov["stock_on_hand"],
                                     cov["replenishment_gap"], cov["target_stock_level"])
        if cat in ("STOCKED_OUT", "CRITICAL", "HIGH"):
            if daily is not None and gap is not None and gap >= 1:
                groups[p.medicine_id]["recipients"].append({"p": p, "cov": cov, "cat": cat})
        elif cat == "LOW":
            f, exp, remaining, reason = p.features or {}, expiry.get((p.facility_id, p.medicine_id)), None, None
            if daily is None or target is None or stock is None or stock <= 0:
                reason = "no usable consumption or stock data"
            elif (num(f.get("obs_days_28")) or 0) < min_obs:
                reason = "insufficient data completeness"
            elif (num(f.get("days_since_stock_obs")) or 0) > fresh:
                reason = "stale stock record"
            elif exp is None and cfg("REDISTRIBUTION_REQUIRE_EXPIRY"):
                reason = "expiry date unknown"
            else:
                remaining = (exp - as_of).days - transit if exp else None
                if remaining is not None and remaining < min_shelf:
                    reason = "insufficient remaining shelf life"
                elif stock - target < 1:
                    reason = "no surplus above minimum stock"
            if reason:
                excluded[reason] += 1
            else:
                groups[p.medicine_id]["donors"].append(
                    {"p": p, "cov": cov, "surplus": stock - target, "remaining": remaining, "expiry": exp})

    new, n_recipients, n_donors, no_donor = [], 0, 0, 0
    for grp in groups.values():
        n_donors += len(grp["donors"])
        recips = sorted(grp["recipients"], key=lambda r: (r["cat"] != "STOCKED_OUT", -r["p"].probability))
        for r in recips:
            n_recipients += 1
            rp, rc = r["p"], r["cov"]
            need, used, made = math.ceil(rc["replenishment_gap"]), 0, 0
            cands = []
            for d in grp["donors"]:
                if d["p"].facility_id == rp.facility_id or d["surplus"] < 1:
                    continue
                dist = haversine_km(d["p"].facility, rp.facility)
                same_region = d["p"].facility.region == rp.facility.region
                if dist is None and not same_region:
                    continue
                if dist is not None and dist > max_km:
                    continue
                cands.append((not same_region, dist if dist is not None else 1e9, d, dist, same_region))
            cands.sort(key=lambda t: (t[0], t[1]))
            for _, _, d, dist, same_region in cands:
                if need <= 0 or used >= max_donors:
                    break
                shelf_cap = math.floor(rc["avg_daily_consumption"] * d["remaining"]) \
                    if d["remaining"] is not None else float("inf")
                q = math.floor(min(need, d["surplus"], shelf_cap))
                if q < 1:
                    continue
                dc = d["cov"]
                new.append(RedistributionSuggestion(
                    medicine=rp.medicine, from_facility=d["p"].facility, to_facility=rp.facility, quantity=float(q),
                    priority_probability=rp.probability, batch=batch, recipient_prediction=rp,
                    model_label=batch.model.label,
                    details={
                        "recipient": {"risk_category": r["cat"], "probability": rp.probability,
                                      "probability_calibrated": rp.probability_calibrated,
                                      "stock_on_hand": rc["stock_on_hand"], "days_of_stock": rc["days_of_stock"],
                                      "lead_time_days": rc["lead_time_days"], "replenishment_gap": rc["replenishment_gap"],
                                      "pending_order_quantity": rc["pending_order_quantity"]},
                        "donor": {"stock_on_hand": dc["stock_on_hand"], "minimum_stock_retained": dc["target_stock_level"],
                                  "surplus_before_transfer": round(d["surplus"], 2),
                                  "stock_after_transfer": round(dc["stock_on_hand"] - q, 2),
                                  "days_of_stock_after_transfer": round((dc["stock_on_hand"] - q) / dc["avg_daily_consumption"], 1),
                                  "earliest_expiry_date": d["expiry"].isoformat() if d["expiry"] else None,
                                  "remaining_shelf_days_after_transit": d["remaining"]},
                        "distance_km": dist, "same_region": same_region,
                        "constraints": [
                            "Donor keeps at least its minimum stock (lead-time plus safety-stock cover).",
                            f"Donor stock has at least {min_shelf} days of shelf life left after {transit} days of transit.",
                            "Quantity does not exceed what the recipient can use before the stock expires.",
                            "Same medicine code and unit. The reviewer must verify batch, formulation and storage conditions.",
                        ],
                        "assumptions": rc["assumptions"],
                        "data_warnings": rc["flags"] + dc["flags"],
                        "human_review_required": True,
                    }))
                d["surplus"] -= q
                need -= q
                used += 1
                made += 1
            if made == 0:
                no_donor += 1

    with transaction.atomic():
        superseded = RedistributionSuggestion.objects.filter(
            status=RedistributionSuggestion.Status.PENDING_REVIEW
        ).update(status=RedistributionSuggestion.Status.SUPERSEDED)
        RedistributionSuggestion.objects.bulk_create(new, batch_size=1000)
    return {"batch": batch.pk, "as_of_date": as_of, "recipients": n_recipients, "donor_candidates": n_donors,
            "suggestions_created": len(new), "recipients_without_donor": no_donor,
            "pending_superseded": superseded, "donor_exclusions": dict(excluded)}