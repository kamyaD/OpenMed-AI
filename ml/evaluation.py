import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _f(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) or np.isinf(x) else round(x, 6)


def expected_calibration_error(y, p, bins=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(y, p, threshold, pred=None, probabilistic=True):
    cfg = settings.MSP
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= threshold) if pred is None else np.asarray(pred, dtype=bool)
    n = len(y)
    out = {"n": n, "positives": int(y.sum()), "threshold": _f(threshold)}
    if n == 0:
        return out
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = n - tp - fp - fn
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    out.update({
        "prevalence": _f(y.mean()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": _f(recall), "precision": _f(precision), "alert_rate": _f(pred.mean()),
        "f1": _f(2 * precision * recall / (precision + recall)) if precision and recall else None,
        "cost_per_1000": _f((fn * cfg["COST_FALSE_NEGATIVE"] + fp * cfg["COST_FALSE_POSITIVE"]) / n * 1000),
    })
    if 0 < y.sum() < n:
        out["pr_auc"] = _f(average_precision_score(y, p))
        out["roc_auc"] = _f(roc_auc_score(y, p))
        if probabilistic:
            out["brier"] = _f(brier_score_loss(y, np.clip(p, 0, 1)))
            out["ece"] = _f(expected_calibration_error(y, p))
    return out


def select_threshold(y, p, c_fn, c_fp):
    """Threshold minimising expected cost = FN*c_fn + FP*c_fp. Ties go to the lower threshold (more recall)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    best, curve = None, []
    for thr in np.round(np.arange(0.02, 0.99, 0.01), 2):
        pred = p >= thr
        fn = int((~pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        cost = fn * c_fn + fp * c_fp
        curve.append({"threshold": float(thr), "fn": fn, "fp": fp, "cost": float(cost)})
        if best is None or cost < best[1]:
            best = (float(thr), cost)
    return best[0], curve[::7]


def subgroup_report(df, proba, threshold):
    """Performance by region, urban/rural, facility size, medicine category and data completeness."""
    cfg = settings.MSP
    d = df.reset_index(drop=True).copy()
    d["_p"] = np.asarray(proba, dtype=float)
    d["urban_rural"] = np.where(d["is_urban"] >= 0.5, "urban", "rural")
    d["facility_size"] = "KEPH level " + d["facility_level"].round().astype(int).astype(str)
    d["data_completeness"] = pd.cut(d["obs_days_28"], bins=[-1, 20, 25, 28.5],
                                    labels=["low (<=20 of 28 days)", "medium (21-25)", "high (26-28)"]).astype(str)
    overall = classification_metrics(d["y"], d["_p"], threshold)
    groups, flags = {}, []
    for g in ["region", "urban_rural", "facility_size", "medicine_category", "data_completeness"]:
        rows = []
        for val, part in d.groupby(g):
            m = classification_metrics(part["y"], part["_p"], threshold)
            m["group"] = str(val)
            m["insufficient_support"] = m["positives"] < cfg["SUBGROUP_MIN_SUPPORT"]
            gap = None
            if not m["insufficient_support"] and m.get("recall") is not None and overall.get("recall") is not None:
                gap = overall["recall"] - m["recall"]
            m["recall_gap_vs_overall"] = _f(gap)
            m["flagged"] = bool(gap is not None and gap > cfg["SUBGROUP_RECALL_GAP"])
            if m["flagged"]:
                flags.append(f"{g} = {val}: recall {m['recall']:.2f} vs overall {overall['recall']:.2f}")
            rows.append(m)
        groups[g] = sorted(rows, key=lambda r: r["group"])
    return {"overall": overall, "groups": groups, "flags": flags, "requires_review": bool(flags),
            "criteria": {"recall_gap": cfg["SUBGROUP_RECALL_GAP"], "min_positives": cfg["SUBGROUP_MIN_SUPPORT"]},
            "notes": "Groups with fewer positive events than min_positives are reported but not assessed for "
                     "disparity. Disparities trigger human review before promotion; they do not prove bias."}