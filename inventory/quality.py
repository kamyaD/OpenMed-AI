from datetime import timedelta

from django.conf import settings
from django.db.models import Count, Max
from django.utils import timezone

from .models import DataQualityIssue, Facility, InventorySnapshot, Medicine


def quality_report(days=30, group_by="facility", threshold=None):
    cfg = settings.MSP
    threshold = cfg["QUALITY_COMPLETENESS_THRESHOLD"] if threshold is None else threshold
    today = timezone.localdate()
    start = today - timedelta(days=days - 1)
    n_fac = Facility.objects.filter(is_active=True).count()
    n_med = Medicine.objects.filter(is_essential=True).count()

    field = {"facility": "facility", "medicine": "medicine", "source": "source"}[group_by]
    issue_field = {"facility": "facility", "medicine": "medicine", "source": "batch__source"}[group_by]

    agg = {r[field]: r for r in InventorySnapshot.objects.filter(date__gte=start).values(field)
           .annotate(records=Count("id"), last_date=Max("date"), last_ingested=Max("ingested_at"))}
    open_qs = DataQualityIssue.objects.filter(status="OPEN", created_at__date__gte=start)
    open_issues = {r[issue_field]: r["c"] for r in open_qs.values(issue_field).annotate(c=Count("id"))}

    if group_by == "facility":
        names = {f.id: f.name for f in Facility.objects.filter(is_active=True)}
        expected = n_med * days
    elif group_by == "medicine":
        names = {m.id: m.name for m in Medicine.objects.filter(is_essential=True)}
        expected = n_fac * days
    else:
        names = {s: s for s in set(agg) | set(open_issues)}
        expected = None

    groups = []
    for key, name in names.items():
        a = agg.get(key)
        records = a["records"] if a else 0
        last = a["last_date"] if a else None
        completeness = round(records / expected, 4) if expected else None
        groups.append({
            "id": key, "name": name, "records": records, "expected_records": expected,
            "completeness": completeness, "last_record_date": last,
            "days_since_last_record": (today - last).days if last else None,
            "last_ingested_at": a["last_ingested"] if a else None,
            "open_issues": open_issues.get(key, 0),
            "below_threshold": completeness is not None and completeness < threshold,
        })
    groups.sort(key=lambda g: g["completeness"] if g["completeness"] is not None else 1)

    by_type = {r["issue_type"]: r["c"] for r in open_qs.values("issue_type").annotate(c=Count("id"))}
    return {
        "window": {"start": start, "end": today, "days": days},
        "group_by": group_by,
        "completeness_threshold": threshold,
        "summary": {
            "groups": len(groups),
            "below_threshold": sum(g["below_threshold"] for g in groups),
            "open_issues_by_type": by_type,
            "open_issues_total": sum(by_type.values()),
        },
        "groups": groups,
    }