import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Max, Min
from django.utils import timezone

from inventory.models import Facility, InventorySnapshot
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features
from ml.models import DatasetVersion
from ml.splits import compute_boundaries
from ml.utils import code_version

LABEL_DEF = ("y=1 if end-of-day stock_on_hand == 0 on any observed day in t+1..t+{H}. y=0 only if no zero is observed "
             "and at least {N} of the {H} days were observed; otherwise the row is excluded (unverified label). "
             "Rows with zero stock at t are excluded (stockout already occurred). All features use data dated <= t.")


class Command(BaseCommand):
    help = "Build a versioned, leakage-safe analytical dataset (parquet) with stockout labels"

    def add_arguments(self, p):
        p.add_argument("--name", default="stockout_h7")
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--chunk", type=int, default=50, help="facilities per chunk (controls memory)")

    def handle(self, *args, **o):
        cfg = settings.MSP
        H = cfg["HORIZON_DAYS"]
        agg = InventorySnapshot.objects.aggregate(lo=Min("date"), hi=Max("date"))
        if agg["lo"] is None:
            raise CommandError("No inventory data. Run seed_demo_data or ingest data first.")
        parse = lambda s: datetime.strptime(s, "%Y-%m-%d").date()
        start = parse(o["start"]) if o["start"] else agg["lo"] + timedelta(days=28)
        end = parse(o["end"]) if o["end"] else agg["hi"] - timedelta(days=H)
        if start >= end:
            raise CommandError(f"Not enough history: start={start}, end={end}.")

        ids = list(Facility.objects.filter(is_active=True).order_by("id").values_list("id", flat=True))
        parts, excl = [], {"currently_out_of_stock": 0, "unverified_label": 0}
        for i in range(0, len(ids), o["chunk"]):
            df = build_features(start, end, ids[i:i + o["chunk"]], None, with_labels=True)
            if df.empty:
                continue
            out, unver = df["stock_on_hand"] <= 0, df["y"].isna()
            excl["currently_out_of_stock"] += int(out.sum())
            excl["unverified_label"] += int((~out & unver).sum())
            parts.append(df[~out & ~unver])
            self.stdout.write(f"facilities {i + 1}-{min(i + o['chunk'], len(ids))}/{len(ids)}: kept {len(parts[-1])} rows")
        if not parts:
            raise CommandError("No rows produced.")
        data = pd.concat(parts, ignore_index=True)
        data["y"] = data["y"].astype("int8")
        boundaries = compute_boundaries(data["date"], H)

        out_dir = Path(settings.DATASET_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"dataset_{timezone.now():%Y%m%d%H%M%S}.parquet"
        data.to_parquet(path, index=False)
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                sha.update(block)

        ds = DatasetVersion.objects.create(
            name=o["name"], horizon_days=H, start_date=start, end_date=end, row_count=len(data),
            positive_count=int(data["y"].sum()), positive_rate=float(data["y"].mean()), file_path=str(path),
            checksum=sha.hexdigest(), label_definition=LABEL_DEF.format(H=H, N=cfg["MIN_FUTURE_OBS"]),
            exclusions=excl, split=boundaries, code_version=code_version(),
            params={"numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES})
        self.stdout.write(self.style.SUCCESS(
            f"Dataset {ds.pk}: {ds.row_count} rows, positive rate {ds.positive_rate:.3f}, excluded {excl}"))
        self.stdout.write(f"Split: {boundaries}")