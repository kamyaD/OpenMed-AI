import json
import statistics
from datetime import datetime, timedelta

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Q
from django.utils import timezone

from decision.models import RedistributionSuggestion
from decision.monitoring import realized_performance
from inventory.models import Facility, InventorySnapshot
from ml.models import ModelVersion


def stockout_rate(ids, lo, hi):
    a = InventorySnapshot.objects.filter(facility_id__in=ids, date__range=(lo, hi)).aggregate(
        n=Count("id"), z=Count("id", filter=Q(stock_on_hand__lte=0)))
    return round(a["z"] / a["n"], 4) if a["n"] else None


class Command(BaseCommand):
    help = "Pilot outcome report: stockout rate (pilot vs comparison), adoption, predictive performance, data limits"

    def add_arguments(self, p):
        p.add_argument("--facilities", required=True, help="comma-separated pilot facility codes")
        p.add_argument("--start", required=True)
        p.add_argument("--end", required=True)
        p.add_argument("--baseline-days", type=int, default=60)

    def handle(self, *args, **o):
        parse = lambda s: datetime.strptime(s, "%Y-%m-%d").date()
        start, end = parse(o["start"]), parse(o["end"])
        codes = [c.strip() for c in o["facilities"].split(",") if c.strip()]
        pilot = list(Facility.objects.filter(code__in=codes).values_list("id", flat=True))
        if len(pilot) != len(set(codes)):
            raise CommandError("One or more pilot facility codes are unknown.")
        comp = list(Facility.objects.filter(is_active=True).exclude(id__in=pilot).values_list("id", flat=True))
        b_lo, b_hi = start - timedelta(days=o["baseline_days"]), start - timedelta(days=1)

        rates = {g: {"baseline": stockout_rate(ids, b_lo, b_hi), "pilot_period": stockout_rate(ids, start, end)}
                 for g, ids in (("pilot", pilot), ("comparison", comp))}
        change = {g: (None if None in r.values() else round(r["pilot_period"] - r["baseline"], 4)) for g, r in rates.items()}
        did = None if None in change.values() else round(change["pilot"] - change["comparison"], 4)

        sugg = list(RedistributionSuggestion.objects.filter(
            Q(from_facility_id__in=pilot) | Q(to_facility_id__in=pilot), generated_at__date__range=(start, end)))
        by_status = {}
        for s in sugg:
            by_status[s.status] = by_status.get(s.status, 0) + 1
        hours = [(s.decided_at - s.generated_at).total_seconds() / 3600 for s in sugg if s.decided_at]
        moved = sum(s.details.get("completion", {}).get("quantity_moved", 0) for s in sugg if s.status == "COMPLETED")
        got = InventorySnapshot.objects.filter(facility_id__in=pilot, date__range=(start, end)).count()
        expected = len(pilot) * (end - start).days

        report = {
            "pilot_facilities": codes, "window": [str(start), str(end)], "baseline_window": [str(b_lo), str(b_hi)],
            "stockout_rate": rates, "change": change, "difference_in_differences": did,
            "adoption": {"suggestions": len(sugg), "by_status": by_status,
                         "median_hours_to_decision": round(statistics.median(hours), 1) if hours else None,
                         "quantity_moved_completed": moved},
            "predictive": realized_performance(ModelVersion.objects.filter(status="ACTIVE").first(), end, pilot,
                                               window_days=(end - start).days + 1),
            "data_limitations": {"stock_record_completeness_pilot_period": round(got / (expected * 1), 4) if expected else None,
                                 "note": "Completeness is records / (facilities x days) for all medicines combined in the denominator "
                                         "only if each facility reports one record per day per medicine; divide by medicines for a per-medicine figure."},
            "caveats": ["Comparison facilities are not randomised; differences may reflect other factors.",
                        "Stockout rate counts end-of-day zero-stock records only.",
                        "Wastage and expiry outcomes are not computed here; add them from the agreed source."],
        }
        path = settings.BASE_DIR / "artifacts" / f"pilot_report_{timezone.now():%Y%m%d%H%M%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
        self.stdout.write(json.dumps(report, indent=2, default=str))
        self.stdout.write(self.style.SUCCESS(f"Saved {path}"))