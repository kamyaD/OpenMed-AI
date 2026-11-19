from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Daily pipeline: batch predictions -> redistribution suggestions -> monitoring"

    def handle(self, *args, **o):
        failed = []
        for name in ("run_batch_predictions", "generate_redistribution", "run_monitoring"):
            try:
                call_command(name)
            except Exception as e:
                failed.append(name)
                self.stderr.write(f"{name} failed: {e}")
        if failed:
            raise CommandError(f"Failed steps: {', '.join(failed)}")