import csv
import io
import json
import math
from datetime import date, datetime, timedelta

import pandas as pd
from django.db import transaction
from django.utils import timezone

from .models import (ConsumptionRecord, DataQualityIssue, Facility, IngestionBatch,
                     InventorySnapshot, Medicine, Order)

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d")
CHUNK = 2000
REQUIRED = {
    "INVENTORY": ["facility_code", "medicine_code", "date", "stock_on_hand"],
    "CONSUMPTION": ["facility_code", "medicine_code", "date", "quantity_dispensed"],
    "ORDERS": ["facility_code", "medicine_code", "order_number", "ordered_date", "quantity_ordered"],
}


class RowError(Exception):
    def __init__(self, issue_type, message):
        super().__init__(message)
        self.issue_type, self.message = issue_type, message


# ---------------------------------------------------------------- parsing helpers
def read_rows(uploaded_file=None, records=None):
    if records is not None:
        return list(records)
    text = uploaded_file.read().decode("utf-8-sig")
    if getattr(uploaded_file, "name", "").lower().endswith(".json"):
        data = json.loads(text)
        return data if isinstance(data, list) else data["records"]
    return list(csv.DictReader(io.StringIO(text)))


def parse_date(value, name="date", allow_future=False):
    if isinstance(value, datetime):
        d = value.date()
    elif isinstance(value, date):
        d = value
    else:
        text, d = str(value).strip(), None
        for fmt in DATE_FORMATS:
            try:
                d = datetime.strptime(text, fmt).date()
                break
            except ValueError:
                pass
        if d is None:
            raise RowError("INVALID_DATE", f"{name} is not a valid date: {text!r}")
    if d.year < 2000:
        raise RowError("INVALID_DATE", f"{name} is implausibly old: {d}")
    if not allow_future and d > timezone.localdate():
        raise RowError("INVALID_DATE", f"{name} is in the future: {d}")
    return d


def parse_number(value, name, allow_negative=False):
    try:
        n = float(str(value).replace(",", "").strip())
    except ValueError:
        raise RowError("MALFORMED", f"{name} is not numeric: {value!r}")
    if math.isnan(n) or math.isinf(n):
        raise RowError("MALFORMED", f"{name} is not a finite number")
    if n < 0 and not allow_negative:
        raise RowError("NEGATIVE_VALUE", f"{name} is negative: {n}")
    return n


def _blank(v):
    return v is None or str(v).strip() == ""


class Ctx:
    def __init__(self, batch, source):
        self.batch, self.source = batch, source
        self.facilities = {f.code: f for f in Facility.objects.all()}
        self.medicines = {m.code: m for m in Medicine.objects.all()}
        self.f = self.m = None
        self.pending = []   # warnings for the current row

    def lookup(self, row):
        missing = [c for c in REQUIRED[self.batch.dataset_type] if _blank(row.get(c))]
        if missing:
            raise RowError("MISSING_FIELD", "missing required field(s): " + ", ".join(missing))
        self.f = self.facilities.get(str(row["facility_code"]).strip())
        if self.f is None:
            raise RowError("UNKNOWN_FACILITY", f"unknown facility_code {row['facility_code']!r}")
        self.m = self.medicines.get(str(row["medicine_code"]).strip())
        if self.m is None:
            raise RowError("UNKNOWN_MEDICINE", f"unknown medicine_code {row['medicine_code']!r}")
        return self.f, self.m

    def warn(self, issue_type, message):
        self.pending.append((issue_type, message))


# ---------------------------------------------------------------- per-dataset handlers
# Each returns a list of (dedupe_key, model_instance)
def _inventory(row, ctx):
    f, m = ctx.lookup(row)
    d = parse_date(row["date"])
    stock = parse_number(row["stock_on_hand"], "stock_on_hand")
    exp = None if _blank(row.get("expiry_date")) else parse_date(row["expiry_date"], "expiry_date", allow_future=True)
    obj = InventorySnapshot(facility=f, medicine=m, date=d, stock_on_hand=stock, earliest_expiry_date=exp,
                            source=ctx.source, record_id=str(row.get("record_id") or ""), ingestion_batch=ctx.batch)
    return [((f.id, m.id, d), obj)]


def _consumption(row, ctx):
    f, m = ctx.lookup(row)
    end = parse_date(row["date"])
    qty = parse_number(row["quantity_dispensed"], "quantity_dispensed")
    try:
        period = 1 if _blank(row.get("period_days")) else int(float(row["period_days"]))
    except ValueError:
        raise RowError("MALFORMED", f"period_days is not numeric: {row['period_days']!r}")
    if not 1 <= period <= 31:
        raise RowError("MALFORMED", "period_days must be between 1 and 31")
    items = []
    for i in range(period):   # weekly data is spread evenly over its days -> daily cadence
        d = end - timedelta(days=i)
        items.append(((f.id, m.id, d), ConsumptionRecord(
            facility=f, medicine=m, date=d, quantity=qty / period, original_quantity=qty,
            original_period_days=period, source=ctx.source, ingestion_batch=ctx.batch)))
    return items


def _orders(row, ctx):
    f, m = ctx.lookup(row)
    ordered = parse_date(row["ordered_date"], "ordered_date")
    qty = parse_number(row["quantity_ordered"], "quantity_ordered")
    expected = None if _blank(row.get("expected_delivery_date")) else parse_date(
        row["expected_delivery_date"], "expected_delivery_date", allow_future=True)
    delivered = None if _blank(row.get("delivered_date")) else parse_date(row["delivered_date"], "delivered_date")
    qdel = None if _blank(row.get("quantity_delivered")) else parse_number(row["quantity_delivered"], "quantity_delivered")
    if delivered and delivered < ordered:
        ctx.warn("INCONSISTENT_LEAD_TIME", f"delivered_date {delivered} precedes ordered_date {ordered}")
    if not delivered and not expected:
        ctx.warn("MISSING_LEAD_TIME", "no delivered_date or expected_delivery_date")
    num = str(row["order_number"]).strip()
    obj = Order(facility=f, medicine=m, order_number=num, ordered_date=ordered, quantity_ordered=qty,
                expected_delivery_date=expected, delivered_date=delivered, quantity_delivered=qdel,
                source=ctx.source, ingestion_batch=ctx.batch)
    return [((f.id, m.id, num), obj)]


HANDLERS = {"INVENTORY": _inventory, "CONSUMPTION": _consumption, "ORDERS": _orders}
MODELS = {"INVENTORY": InventorySnapshot, "CONSUMPTION": ConsumptionRecord, "ORDERS": Order}


def _existing_keys(dataset_type, rows):
    if dataset_type == "ORDERS":
        return set(Order.objects.values_list("facility_id", "medicine_id", "order_number"))
    dates = []
    for r in rows:
        try:
            dates.append(parse_date(r.get("date"), allow_future=True))
        except Exception:
            pass
    if not dates:
        return set()
    pad = 31 if dataset_type == "CONSUMPTION" else 0
    return set(MODELS[dataset_type].objects.filter(
        date__range=(min(dates) - timedelta(days=pad), max(dates))
    ).values_list("facility_id", "medicine_id", "date"))


def _jsonable(row):
    return json.loads(json.dumps(row, default=str)) if isinstance(row, dict) else {"raw": str(row)}


def _issue(ctx, issue_type, severity, ref, row, message):
    return DataQualityIssue(batch=ctx.batch, dataset_type=ctx.batch.dataset_type, issue_type=issue_type,
                            severity=severity, record_ref=ref[:200], facility=ctx.f, medicine=ctx.m,
                            raw_payload=_jsonable(row), message=message)


# ---------------------------------------------------------------- main entry point
def ingest(dataset_type, rows, source, user=None, filename=""):
    batch = IngestionBatch.objects.create(
        dataset_type=dataset_type, source=source, filename=filename, total_rows=len(rows),
        created_by=user if getattr(user, "is_authenticated", False) else None)
    ctx, handler = Ctx(batch, source), HANDLERS[dataset_type]
    existing = _existing_keys(dataset_type, [r for r in rows if isinstance(r, dict)])
    seen, objs, issues = set(), [], []
    accepted = rejected = 0

    for i, row in enumerate(rows, start=1):
        ctx.f = ctx.m = None
        ctx.pending = []
        ref = str(row.get("record_id") or row.get("order_number") or f"row {i}") if isinstance(row, dict) else f"row {i}"
        try:
            if not isinstance(row, dict):
                raise RowError("MALFORMED", "row is not an object")
            fresh = []
            for key, obj in handler(row, ctx):
                if key in seen or key in existing:
                    issues.append(_issue(ctx, "DUPLICATE", "ERROR", ref, row, f"duplicate record for key {key}"))
                    continue
                seen.add(key)
                fresh.append(obj)
            if fresh:
                objs.extend(fresh)
                accepted += 1
                issues.extend(_issue(ctx, t, "WARNING", ref, row, msg) for t, msg in ctx.pending)
            else:
                rejected += 1
        except RowError as e:
            rejected += 1
            issues.append(_issue(ctx, e.issue_type, "ERROR", ref, row, e.message))

    with transaction.atomic():
        model = MODELS[dataset_type]
        for s in range(0, len(objs), CHUNK):
            model.objects.bulk_create(objs[s:s + CHUNK], ignore_conflicts=True)
        DataQualityIssue.objects.bulk_create(issues, batch_size=CHUNK)
        extra = 0
        if dataset_type in ("INVENTORY", "CONSUMPTION") and objs:
            extra = cross_check_zero_stock(batch, min(o.date for o in objs), max(o.date for o in objs))
        batch.accepted_rows, batch.rejected_rows = accepted, rejected
        batch.warning_count = sum(1 for x in issues if x.severity == "WARNING") + extra
        batch.finished_at = timezone.now()
        if accepted == 0 and rows:
            batch.status = IngestionBatch.Status.FAILED
        elif issues or extra:
            batch.status = IngestionBatch.Status.COMPLETED_WITH_ISSUES
        batch.save()
    return batch


def cross_check_zero_stock(batch, lo, hi):
    """Flag: stock is 0 at end of day d and d+1, yet dispensing was recorded on d+1."""
    keys = ["facility_id", "medicine_id", "date"]
    window = (lo, hi + timedelta(days=1))
    zero = pd.DataFrame(list(InventorySnapshot.objects.filter(stock_on_hand=0, date__range=window).values(*keys)))
    cons = pd.DataFrame(list(ConsumptionRecord.objects.filter(quantity__gt=0, date__range=window).values(*keys)))
    if zero.empty or cons.empty:
        return 0
    for df in (zero, cons):
        df["date"] = pd.to_datetime(df["date"])
    zero_next = zero.assign(date=zero["date"] - pd.Timedelta(days=1))    # zero at d+1, aligned to d
    cons_next = cons.assign(date=cons["date"] - pd.Timedelta(days=1))    # dispensing at d+1, aligned to d
    hits = zero.merge(zero_next, on=keys).merge(cons_next, on=keys).head(5000)
    if hits.empty:
        return 0
    already = set(DataQualityIssue.objects.filter(
        issue_type="ZERO_STOCK_DISPENSING", status="OPEN").values_list("facility_id", "medicine_id", "record_ref"))
    new = []
    for r in hits.itertuples():
        ref = (r.date + pd.Timedelta(days=1)).date().isoformat()
        if (r.facility_id, r.medicine_id, ref) in already:
            continue
        new.append(DataQualityIssue(
            batch=batch, dataset_type="CROSS_CHECK", issue_type="ZERO_STOCK_DISPENSING", severity="WARNING",
            record_ref=ref, facility_id=r.facility_id, medicine_id=r.medicine_id,
            message=f"Stock was 0 on {r.date.date()} and {ref} but dispensing was recorded on {ref}"))
    DataQualityIssue.objects.bulk_create(new, batch_size=CHUNK)
    return len(new)