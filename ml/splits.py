import numpy as np
import pandas as pd


def compute_boundaries(dates, horizon, train_frac=0.60, val_frac=0.20):
    """
    Chronological split with purge gaps of `horizon` days, so that labels (which look `horizon` days
    ahead) never reach into the next period. Validation is split again into val_a (calibration fit)
    and val_b (calibration check, model selection, threshold).
    """
    u = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values().reset_index(drop=True)
    n, gap = len(u), pd.Timedelta(days=horizon + 1)
    if n < 100:
        raise ValueError(f"Only {n} distinct dates; at least 100 are needed for a train/val/test split. "
                         "Re-seed with --days 365 or ingest more history.")
    train_end = u[int(n * train_frac) - 1]
    val_start = train_end + gap
    val_end = u[int(n * (train_frac + val_frac)) - 1]
    test_start, test_end = val_end + gap, u.iloc[-1]
    val_a_end = val_start + pd.Timedelta(days=(val_end - val_start).days // 2)
    val_b_start = val_a_end + gap
    if not (val_start < val_a_end < val_b_start <= val_end and test_start < test_end):
        raise ValueError("History too short for the requested purge gaps.")
    iso = lambda t: t.date().isoformat()
    return {"train_end": iso(train_end), "val_start": iso(val_start), "val_a_end": iso(val_a_end),
            "val_b_start": iso(val_b_start), "val_end": iso(val_end), "test_start": iso(test_start),
            "test_end": iso(test_end), "purge_days": horizon + 1, "train_frac": train_frac, "val_frac": val_frac,
            "rationale": "Chronological split; purge gap of horizon+1 days between periods prevents label overlap."}


def split_masks(dates, b):
    d = pd.Series(pd.to_datetime(dates)).to_numpy()
    T = lambda k: np.datetime64(pd.Timestamp(b[k]))
    return {
        "train": d <= T("train_end"),
        "val_a": (d >= T("val_start")) & (d <= T("val_a_end")),
        "val_b": (d >= T("val_b_start")) & (d <= T("val_end")),
        "test": (d >= T("test_start")) & (d <= T("test_end")),
    }