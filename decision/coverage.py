import math
from datetime import timedelta

from django.conf import settings
from django.db.models import Sum

from inventory.models import Order

from .config import cfg


def num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def pending_quantity_map(as_of, facility_ids=None, medicine_ids=None):
    """Quantity on open (undelivered) orders per (facility, medicine); orders older than the cap are ignored."""
    qs = Order.objects.filter(delivered_date__isnull=True, ordered_date__lte=as_of,
                              ordered_date__gte=as_of - timedelta(days=cfg("PENDING_ORDER_MAX_AGE_DAYS")))
    if facility_ids is not None:
        qs = qs.filter(facility_id__in=list(facility_ids))
    if medicine_ids is not None:
        qs = qs.filter(medicine_id__in=list(medicine_ids))
    return {(r["facility_id"], r["medicine_id"]): r["q"] or 0.0
            for r in qs.values("facility_id", "medicine_id").annotate(q=Sum("quantity_ordered"))}


def coverage_from_features(f, pending_qty=0.0):
    """
    Days of stock vs lead time and safety stock. Safe with zero/low consumption and missing data:
    in those cases nothing is estimated and the reason is returned in `flags`.
    """
    f = f or {}
    safety = settings.MSP["SAFETY_STOCK_DAYS"]
    stock = num(f.get("stock_on_hand"))
    daily = next((v for v in (num(f.get("cons_avg_14")), num(f.get("cons_avg_28")), num(f.get("cons_avg_7")))
                  if v is not None), None)
    lead = num(f.get("lead_time_mean")) or settings.MSP["DEFAULT_LEAD_TIME_DAYS"]
    observed = (num(f.get("n_past_deliveries")) or 0) > 0
    flags = []
    out = {
        "stock_on_hand": stock, "avg_daily_consumption": daily, "days_of_stock": None,
        "lead_time_days": lead, "lead_time_source": "observed deliveries" if observed else "default assumption",
        "safety_stock_days": safety, "target_stock_level": None, "pending_order_quantity": pending_qty,
        "projected_position": None, "replenishment_gap": None, "covers_lead_time": None, "flags": flags,
        "assumptions": [
            "Daily consumption = 14-day average (falls back to 28-day, then 7-day).",
            f"Target stock level = daily consumption x (lead time + {safety} safety-stock days).",
            "Open orders are assumed to arrive in full; orders older than "
            f"{cfg('PENDING_ORDER_MAX_AGE_DAYS')} days are ignored.",
            "Replenishment gap = max(0, target stock level - (stock on hand + open orders)).",
        ],
    }
    if stock is None:
        flags.append("Stock level unavailable; coverage not estimated.")
        return out
    out["projected_position"] = round(stock + pending_qty, 2)
    if daily is None or daily < cfg("MIN_DAILY_CONSUMPTION"):
        flags.append("Little or no recent consumption; days of stock and replenishment gap are not estimated.")
        out["replenishment_gap"] = 0.0
        return out
    target = daily * (lead + safety)
    out.update(days_of_stock=round(stock / daily, 2), target_stock_level=round(target, 2),
               replenishment_gap=round(max(0.0, target - (stock + pending_qty)), 2),
               covers_lead_time=bool(stock / daily >= lead))
    if not observed:
        flags.append("No delivery history; default lead time assumed.")
    return out


def compute_coverage(facility, medicine, as_of):
    from ml.features import build_features
    rows = build_features(as_of, as_of, [facility.id], [medicine.id])
    if rows.empty:
        return {"available": False, "flags": ["No stock observation within 3 days of the as-of date."]}
    pend = pending_quantity_map(as_of, [facility.id], [medicine.id]).get((facility.id, medicine.id), 0.0)
    return {"available": True, **coverage_from_features(rows.iloc[0].to_dict(), pend)}