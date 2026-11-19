from django.core.management.base import BaseCommand, CommandError

from accounts.audit import log_action
from decision.redistribution import generate_suggestions


class Command(BaseCommand):
    help = "Generate redistribution suggestions from the latest batch (all start as PENDING_REVIEW)"

    def handle(self, *args, **o):
        try:
            summary = generate_suggestions()
        except ValueError as e:
            raise CommandError(str(e))
        log_action(None, "redistribution.generated", None, **summary)
        self.stdout.write(self.style.SUCCESS(str(summary)))