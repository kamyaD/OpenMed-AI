from datetime import timedelta

import numpy as np

from .features import NUMERIC_FEATURES, build_features


def leakage_check(dates, facility_ids=None, medicine_ids=None):
    """Returns a list of problems; an empty list means no future information influenced the features."""
    problems = []
    for t in dates:
        a = build_features(t, t, facility_ids, medicine_ids)                          # data <= t only
        b = build_features(t, t + timedelta(days=10), facility_ids, medicine_ids)     # later data exists
        b = b[b["date"] == a["date"].iloc[0]] if len(a) and len(b) else b
        if a.empty:
            problems.append({"date": str(t), "feature": "*", "detail": "no rows produced"})
            continue
        m = a.merge(b, on=["facility_id", "medicine_id"], suffixes=("_a", "_b"))
        if len(m) != len(a):
            problems.append({"date": str(t), "feature": "*", "detail": f"row count {len(a)} vs {len(m)}"})
        for f in NUMERIC_FEATURES:
            x, y = m[f + "_a"].to_numpy(dtype=float), m[f + "_b"].to_numpy(dtype=float)
            bad = int((~np.isclose(x, y, rtol=1e-4, atol=1e-4, equal_nan=True)).sum())
            if bad:
                problems.append({"date": str(t), "feature": f, "detail": f"{bad} rows differ"})
    return problems