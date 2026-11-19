from django.core.management.base import BaseCommand, CommandError

from ml.models import DatasetVersion
from ml.training import train_all


class Command(BaseCommand):
    help = "Train baseline, logistic regression and random forest on a dataset version"

    def add_arguments(self, p):
        p.add_argument("--dataset", type=int, help="DatasetVersion id (default: latest)")
        p.add_argument("--facility-holdout", type=float, default=0.0,
                       help="fraction of facilities withheld from training to test generalisation to unseen facilities")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--max-rows", type=int, default=400_000)

    def handle(self, *args, **o):
        ds = DatasetVersion.objects.filter(pk=o["dataset"]).first() if o["dataset"] else DatasetVersion.objects.first()
        if ds is None:
            raise CommandError("No dataset. Run build_dataset first.")
        try:
            versions = train_all(ds, None, o["facility_holdout"], o["seed"], o["max_rows"], log=self.stdout.write)
        except ValueError as e:
            raise CommandError(str(e))
        for v in versions:
            rec = v.metrics.get("selection", {}).get("recommended")
            self.stdout.write(f"  ModelVersion {v.pk}: {v.name} [{v.status}]" + ("  <- recommended" if rec else ""))
        self.stdout.write(self.style.SUCCESS("Done. Review /api/v1/ml/models/ and promote one (admin) when approved."))