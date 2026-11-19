from django.core.management.base import BaseCommand, CommandError

from ml.models import ModelVersion
from ml.training import evaluate_model_subgroups


class Command(BaseCommand):
    help = "Re-run subgroup performance evaluation for a model version"

    def add_arguments(self, p):
        p.add_argument("--model", type=int, required=True)

    def handle(self, *args, **o):
        mv = ModelVersion.objects.filter(pk=o["model"]).exclude(artifact_path="").first()
        if mv is None or mv.dataset is None:
            raise CommandError("Model not found, or it has no artifact/dataset.")
        rep = evaluate_model_subgroups(mv)
        self.stdout.write(f"requires_review={rep['requires_review']}")
        for f in rep["flags"]:
            self.stdout.write(f"  FLAG {f}")