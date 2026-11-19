from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from accounts.audit import log_action
from decision.lifecycle import evaluate_on_dataset, retrain_recommendation
from decision.monitoring import run_monitoring
from ml.models import DatasetVersion, ModelVersion
from ml.training import train_all


class Command(BaseCommand):
    help = "Check the documented retraining criteria; with --execute build a dataset and train CANDIDATE models"

    def add_arguments(self, p):
        p.add_argument("--force", action="store_true", help="retrain even if criteria are not met (sponsor request)")
        p.add_argument("--execute", action="store_true", help="actually build the dataset and train")

    def handle(self, *args, **o):
        rec = retrain_recommendation(run_monitoring())
        for r in rec["reasons"]:
            self.stdout.write(f"  criterion met: {r}")
        if not (rec["recommended"] or o["force"]):
            return self.stdout.write("Retraining criteria not met. Nothing to do.")
        if not o["execute"]:
            return self.stdout.write("Retraining is recommended. Re-run with --execute to build and train candidates.")

        call_command("build_dataset")
        ds = DatasetVersion.objects.first()
        try:
            versions = train_all(ds, None, log=self.stdout.write)
        except ValueError as e:
            raise CommandError(str(e))
        active = ModelVersion.objects.filter(status=ModelVersion.Status.ACTIVE).first()
        active_metrics = None
        if active is not None:
            try:
                active_metrics = evaluate_on_dataset(active, ds)
            except Exception as e:
                self.stderr.write(f"Could not evaluate the active model on the new test period: {e}")
        for v in versions:
            if not v.artifact_path or active_metrics is None:
                continue
            cand, cur = v.metrics["test"]["cost_per_1000"], active_metrics["cost_per_1000"]
            v.metrics["comparison_to_active"] = {
                "active_model": active.label, "active_on_new_test": active_metrics, "candidate_test": v.metrics["test"],
                "candidate_better": bool(cand < cur),
                "note": "Both models evaluated on the same new test period with their own thresholds."}
            v.save(update_fields=["metrics"])
            self.stdout.write(f"  {v.label}: cost {cand} vs active {cur} -> {'better' if cand < cur else 'not better'}")
        log_action(None, "retraining.executed", ds, reasons=rec["reasons"], forced=o["force"],
                   candidates=[v.pk for v in versions])
        self.stdout.write(self.style.SUCCESS(
            "Candidates created. An administrator must review them and promote one with a written rationale."))