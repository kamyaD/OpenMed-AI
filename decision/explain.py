import numpy as np
import pandas as pd
from django.conf import settings

from .risk import categorize

DISCLAIMER = ("These factors describe how this prediction relates to recorded stock, consumption and delivery data. "
              "They are associations, not proven causes, and a human must review any action taken.")


def deterministic_factors(f, horizon):
    out = []

    def add(direction, factor, detail):
        out.append({"factor": factor, "direction": direction, "detail": detail, "source": "deterministic_rule"})

    g = lambda k: (None if f.get(k) is None else float(f[k]))
    dos, cov, trend, lead = g("days_of_stock"), g("coverage_ratio"), g("cons_trend_7_28"), g("lead_time_mean")
    if dos is not None:
        if dos < horizon:
            add("increases_risk", "Low stock cover",
                f"Stock covers about {dos:.1f} days of recent consumption, less than the {horizon}-day horizon.")
        elif dos >= 3 * horizon:
            add("reduces_risk", "Ample stock cover", f"Stock covers about {dos:.0f} days of recent consumption.")
    if cov is not None and cov < 1:
        add("increases_risk", "Stock below lead-time demand",
            f"Stock is below the demand expected during a typical delivery lead time (coverage ratio {cov:.2f}).")
    if trend is not None and trend >= 1.25:
        add("increases_risk", "Consumption rising",
            f"The last 7 days' average consumption is {trend:.2f}x the 28-day average.")
    elif trend is not None and trend <= 0.75:
        add("reduces_risk", "Consumption falling",
            f"The last 7 days' average consumption is {trend:.2f}x the 28-day average.")
    if lead is not None:
        n = g("n_past_deliveries") or 0
        add("increases_risk" if lead >= 14 else "context", "Delivery lead time",
            f"Deliveries have taken about {lead:.0f} days" + ("." if n > 0 else " (default assumed; no delivery history)."))
    if (g("has_pending_order") or 0) > 0:
        add("reduces_risk", "Open order on record",
            f"An open order is on record (about {g('pending_cover_days') or 0:.0f} days of consumption).")
    elif dos is not None and lead is not None and dos < lead:
        add("increases_risk", "No open order", "No open order is on record while stock covers less than the lead time.")
    if (g("stockout_days_28") or 0) > 0:
        add("increases_risk", "Recent stockouts",
            f"Stock was zero on {g('stockout_days_28'):.0f} of the last 28 observed days.")
    if (g("obs_days_28") or 28) < 21:
        add("context", "Limited data", f"Only {g('obs_days_28'):.0f} of the last 28 days have stock records.")
    return out


def model_contributions(record):
    """Per-feature log-odds contributions. Available only for the active logistic-regression model."""
    from ml.services import get_active
    try:
        mv, bundle = get_active()
    except Exception:
        return None
    if mv.pk != record.model_id or mv.algorithm != "LOGISTIC_REGRESSION":
        return None
    cols = bundle.numeric + bundle.categorical
    feats = record.features or {}
    X = pd.DataFrame([{c: (np.nan if feats.get(c) is None else feats[c]) for c in cols}])
    X[bundle.numeric] = X[bundle.numeric].astype(float)
    prep = bundle.pipeline.named_steps["prep"]
    z = np.asarray(prep.transform(X[cols]))[0]
    contrib = z * bundle.pipeline.named_steps["clf"].coef_[0]
    names = [n.replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]
    top = sorted(zip(names, contrib), key=lambda t: abs(t[1]), reverse=True)[:6]
    return {"method": "logistic regression coefficients x standardised feature values (log-odds units)",
            "top_factors": [{"feature": n, "contribution": round(float(c), 4),
                             "direction": "raises predicted risk" if c > 0 else "lowers predicted risk"}
                            for n, c in top],
            "note": "Shows what the model weighs for this prediction, relative to an average training row."}


def explain_prediction(record):
    cat = categorize(record.probability, record.model_applied)
    base = {"request_id": str(record.request_id), "risk_category": cat, "probability": record.probability,
            "probability_calibrated": record.probability_calibrated, "disclaimer": DISCLAIMER}
    if not record.model_applied:
        return {**base, "factors": [{"factor": "Already out of stock", "direction": "increases_risk",
                                     "detail": "Recorded stock is zero; the model was not applied.",
                                     "source": "deterministic_rule"}],
                "model_attribution": None}
    return {**base, "factors": deterministic_factors(record.features or {}, settings.MSP["HORIZON_DAYS"]),
            "model_attribution": model_contributions(record)}