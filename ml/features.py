import numpy as np
import pandas as pd
from django.conf import settings

from inventory.models import ConsumptionRecord, Facility, InventorySnapshot, Medicine, Order

WARMUP_DAYS = 120        # history loaded before `start`; identical for training and serving (avoids skew)
MAX_STALE_DAYS = 3       # stock is carried forward at most this many days
PAIR_MOD = 1_000_000     # pair id = facility_id * PAIR_MOD + medicine_id

NUMERIC_FEATURES = [
    "stock_on_hand", "cons_avg_7", "cons_avg_14", "cons_avg_28", "cons_std_28", "cons_trend_7_28",
    "days_of_stock", "lead_time_mean", "n_past_deliveries", "lead_time_demand", "coverage_ratio",
    "has_pending_order", "pending_cover_days", "days_since_last_order", "stockout_days_28",
    "stock_change_7d", "days_since_stock_obs", "obs_days_28", "facility_level", "is_urban",
]
CATEGORICAL_FEATURES = ["medicine_category"]
ALL_COLUMNS = ["facility_id", "medicine_id", "date", "region", "county"] + NUMERIC_FEATURES + \
              CATEGORICAL_FEATURES + ["y", "future_obs"]


def _add_pair(df):
    if not df.empty:
        df["pair"] = df["facility_id"].astype("int64") * PAIR_MOD + df["medicine_id"].astype("int64")
    return df


def _load(load_start, load_end, facility_ids, medicine_ids):
    def scope(qs):
        if facility_ids is not None:
            qs = qs.filter(facility_id__in=list(facility_ids))
        if medicine_ids is not None:
            qs = qs.filter(medicine_id__in=list(medicine_ids))
        return qs

    snaps = pd.DataFrame(list(scope(InventorySnapshot.objects.filter(date__range=(load_start, load_end)))
                              .values("facility_id", "medicine_id", "date", "stock_on_hand")))
    cons = pd.DataFrame(list(scope(ConsumptionRecord.objects.filter(date__range=(load_start, load_end)))
                             .values("facility_id", "medicine_id", "date", "quantity")))
    orders = pd.DataFrame(list(scope(Order.objects.filter(ordered_date__lte=load_end))
                               .values("facility_id", "medicine_id", "ordered_date", "delivered_date",
                                       "quantity_ordered")))
    return _add_pair(snaps), _add_pair(cons), _add_pair(orders)


def _wide(df, value, cal, pairs):
    if df.empty:
        return pd.DataFrame(np.nan, index=cal, columns=pairs)
    d = df[["date", "pair", value]].copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.pivot(index="date", columns="pair", values=value).reindex(index=cal, columns=pairs)


def _events(df, datecol, valcol, cal, pairs):
    """Sum of `valcol` per day/pair. Dates before the calendar start collapse onto day 0."""
    if df.empty:
        return pd.DataFrame(0.0, index=cal, columns=pairs)
    d = df[[datecol, "pair", valcol]].copy()
    d[datecol] = pd.to_datetime(d[datecol]).clip(lower=cal[0])
    d = d[d[datecol] <= cal[-1]]
    if d.empty:
        return pd.DataFrame(0.0, index=cal, columns=pairs)
    w = d.groupby([datecol, "pair"])[valcol].sum().unstack("pair")
    return w.reindex(index=cal, columns=pairs).fillna(0.0)


def _lead_times(orders, cal, pairs):
    """Expanding mean lead time using only deliveries that have happened by each date."""
    nan = pd.DataFrame(np.nan, index=cal, columns=pairs)
    if orders.empty:
        return nan, pd.DataFrame(0.0, index=cal, columns=pairs)
    d = orders[orders["delivered_date"].notna()].copy()
    if d.empty:
        return nan, pd.DataFrame(0.0, index=cal, columns=pairs)
    d["lead"] = (pd.to_datetime(d["delivered_date"]) - pd.to_datetime(d["ordered_date"])).dt.days
    d = d[d["lead"] >= 0]
    d["dd"] = pd.to_datetime(d["delivered_date"]).clip(lower=cal[0])
    d = d[d["dd"] <= cal[-1]]
    if d.empty:
        return nan, pd.DataFrame(0.0, index=cal, columns=pairs)
    g = d.groupby(["dd", "pair"])["lead"].agg(["sum", "count"])
    s = g["sum"].unstack("pair").reindex(index=cal, columns=pairs).fillna(0.0).cumsum()
    c = g["count"].unstack("pair").reindex(index=cal, columns=pairs).fillna(0.0).cumsum()
    return s / c.where(c > 0), c


def build_features(start, end, facility_ids=None, medicine_ids=None, with_labels=False):
    """
    One row per (facility, medicine, date) for dates in [start, end] that have a stock observation
    within the last MAX_STALE_DAYS. Every feature at date t uses data dated <= t only.
    With with_labels=True the caller must ensure data exist up to end + horizon.
    """
    cfg = settings.MSP
    H, default_lt = cfg["HORIZON_DAYS"], cfg["DEFAULT_LEAD_TIME_DAYS"]
    s0, e0 = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if e0 < s0:
        raise ValueError("end must not precede start")
    load_start = s0 - pd.Timedelta(days=WARMUP_DAYS)
    load_end = e0 + pd.Timedelta(days=H) if with_labels else e0
    cal = pd.date_range(load_start, load_end, freq="D")

    snaps, cons, orders = _load(load_start.date(), load_end.date(), facility_ids, medicine_ids)
    if snaps.empty:
        return pd.DataFrame(columns=ALL_COLUMNS)
    pairs = np.sort(snaps["pair"].unique())

    S = _wide(snaps, "stock_on_hand", cal, pairs)
    C = _wide(cons, "quantity", cal, pairs)
    Sf = S.ffill(limit=MAX_STALE_DAYS)                      # causal: carries past values forward only
    pos = np.arange(len(cal))[:, None]

    seen = pd.DataFrame(np.where(S.notna().to_numpy(), pos, np.nan), index=cal, columns=pairs).ffill()
    since_obs = pd.DataFrame(pos - seen.to_numpy(), index=cal, columns=pairs)

    avg7 = C.rolling(7, min_periods=3).mean()
    avg14 = C.rolling(14, min_periods=5).mean()
    avg28 = C.rolling(28, min_periods=10).mean()
    std28 = C.rolling(28, min_periods=10).std()
    pos28 = avg28.where(avg28 > 0)
    trend = (avg7 / pos28).clip(0, 10)
    dos = (Sf / pos28).where(avg28 != 0, 365.0).clip(upper=365.0)      # no dispensing -> plenty of cover

    lt, lt_n = _lead_times(orders, cal, pairs)
    lt = lt.fillna(default_lt)
    ltd = avg14 * lt                                                    # lead-time demand
    cov = (Sf / ltd.where(ltd > 0)).where(ltd != 0, 50.0).clip(upper=50.0)

    if orders.empty:
        placed = placed_q = deliv = deliv_q = pd.DataFrame(0.0, index=cal, columns=pairs)
    else:
        o = orders.copy()
        o["one"] = 1.0
        placed = _events(o, "ordered_date", "one", cal, pairs)
        placed_q = _events(o, "ordered_date", "quantity_ordered", cal, pairs)
        dv = o[o["delivered_date"].notna()]
        deliv = _events(dv, "delivered_date", "one", cal, pairs)
        deliv_q = _events(dv, "delivered_date", "quantity_ordered", cal, pairs)
    pending_n = (placed - deliv).cumsum()
    pending_q = (placed_q - deliv_q).cumsum().clip(lower=0)
    has_pending = (pending_n > 0).astype(float)
    pending_cover = (pending_q / pos28).clip(upper=365.0).where(pending_q > 0, 0.0)
    last_order = pd.DataFrame(np.where(placed.to_numpy() > 0, pos, np.nan), index=cal, columns=pairs).ffill()
    days_since_order = pd.DataFrame(pos - last_order.to_numpy(), index=cal, columns=pairs).clip(upper=90).fillna(90)

    zero28 = (S == 0).astype(float).where(S.notna()).rolling(28, min_periods=1).sum()
    obs28 = S.notna().astype(float).rolling(28, min_periods=1).sum()
    change7 = ((Sf - Sf.shift(7)) / pos28).clip(-90, 90)               # change in days-of-consumption

    i0 = WARMUP_DAYS
    i1 = i0 + (e0 - s0).days + 1
    n_days, n_pairs = i1 - i0, len(pairs)

    def cut(w):
        return w.iloc[i0:i1].to_numpy(dtype="float64").ravel()

    data = {
        "stock_on_hand": cut(Sf), "cons_avg_7": cut(avg7), "cons_avg_14": cut(avg14), "cons_avg_28": cut(avg28),
        "cons_std_28": cut(std28), "cons_trend_7_28": cut(trend), "days_of_stock": cut(dos),
        "lead_time_mean": cut(lt), "n_past_deliveries": cut(lt_n.clip(upper=50)), "lead_time_demand": cut(ltd),
        "coverage_ratio": cut(cov), "has_pending_order": cut(has_pending), "pending_cover_days": cut(pending_cover),
        "days_since_last_order": cut(days_since_order), "stockout_days_28": cut(zero28),
        "stock_change_7d": cut(change7), "days_since_stock_obs": cut(since_obs), "obs_days_28": cut(obs28),
    }
    out = pd.DataFrame({k: v.astype("float32") for k, v in data.items()})

    if with_labels:
        rev = S.shift(-1).iloc[::-1]                                    # value at t+1, reversed in time
        fmin = rev.rolling(H, min_periods=1).min().iloc[::-1]           # min stock over t+1..t+H
        fobs = rev.notna().astype(float).rolling(H, min_periods=1).sum().iloc[::-1]
        fm, fo = cut(fmin), cut(fobs)
        out["y"] = np.where(fm <= 0, 1.0, np.where(fo >= cfg["MIN_FUTURE_OBS"], 0.0, np.nan)).astype("float32")
        out["future_obs"] = fo.astype("float32")

    fids = (pairs // PAIR_MOD).astype(int)
    mids = (pairs % PAIR_MOD).astype(int)
    fac = {f.id: f for f in Facility.objects.filter(id__in=np.unique(fids).tolist())}
    med = {m.id: m for m in Medicine.objects.filter(id__in=np.unique(mids).tolist())}
    out["facility_id"] = np.tile(fids, n_days)
    out["medicine_id"] = np.tile(mids, n_days)
    out["date"] = np.repeat(cal[i0:i1].to_numpy(), n_pairs)
    out["region"] = np.tile(np.array([fac[i].region for i in fids], dtype=object), n_days)
    out["county"] = np.tile(np.array([fac[i].county for i in fids], dtype=object), n_days)
    out["facility_level"] = np.tile(np.array([fac[i].level for i in fids], dtype="float32"), n_days)
    out["is_urban"] = np.tile(np.array([float(fac[i].is_urban) for i in fids], dtype="float32"), n_days)
    out["medicine_category"] = np.tile(np.array([med[i].category or "Unknown" for i in mids], dtype=object), n_days)

    return out[out["stock_on_hand"].notna()].reset_index(drop=True)