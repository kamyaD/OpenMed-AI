from django.core.management.base import BaseCommand

from accounts.audit import log_action
from decision.monitoring import run_monitoring


class Command(BaseCommand):
    help = "Run data, prediction, drift and model-performance monitoring and store a report"

    def handle(self, *args, **o):
        r = run_monitoring()
        log_action(None, "monitoring.run", r, status=r.status, alerts=len(r.alerts))
        self.stdout.write(f"Report {r.pk}: {r.status}")
        for a in r.alerts:
            self.stdout.write(f"  [{a['level']}] {a['check']}: {a['message']}  (owner: {a['owner']})")