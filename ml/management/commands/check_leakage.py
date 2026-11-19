import random
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Max, Min

from inventory.models import Facility, InventorySnapshot
from ml.leakage import leakage_check


class Command(BaseCommand):
    help = "Automated temporal-leakage check on real data"

    def add_arguments(self, p):
        p.add_argument("--samples", type=int, default=3)
        p.add_argument("--facilities", type=int, default=15)

    def handle(self, *args, **o):
        agg = InventorySnapshot.objects.aggregate(lo=Min("date"), hi=Max("date"))
        if agg["lo"] is None:
            raise CommandError("No data.")
        span = (agg["hi"] - agg["lo"]).days
        dates = [agg["lo"] + timedelta(days=random.randint(40, max(41, span - 15))) for _ in range(o["samples"])]
        ids = list(Facility.objects.values_list("id", flat=True))
        ids = random.sample(ids, min(o["facilities"], len(ids)))
        problems = leakage_check(dates, ids)
        if problems:
            for p in problems:
                self.stderr.write(str(p))
            raise CommandError("LEAKAGE DETECTED")
        self.stdout.write(self.style.SUCCESS(f"No leakage detected on {len(dates)} dates x {len(ids)} facilities."))