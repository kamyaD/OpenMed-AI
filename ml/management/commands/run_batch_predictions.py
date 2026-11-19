from datetime import datetime

from django.core.management.base import BaseCommand

from ml.services import run_batch


class Command(BaseCommand):
    help = "Run batch predictions with the active model (schedule daily after data ingestion)"

    def add_arguments(self, p):
        p.add_argument("--as-of")

    def handle(self, *args, **o):
        as_of = datetime.strptime(o["as_of"], "%Y-%m-%d").date() if o["as_of"] else None
        job = run_batch(None, as_of)
        self.stdout.write(self.style.SUCCESS(
            f"Job {job.pk}: {job.status}, {job.succeeded}/{job.total} predictions, {job.failed} failed"))