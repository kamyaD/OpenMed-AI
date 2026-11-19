from django.conf import settings
from django.db import models
from django.db.models import Q

USER = settings.AUTH_USER_MODEL


class RiskPolicy(models.Model):
    """Versioned risk bands. Exactly one row is active. With none, provisional defaults apply."""
    bands = models.JSONField()                    # {"CRITICAL": 0.6, "HIGH": 0.3, "MEDIUM": 0.1}
    rationale = models.TextField()
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(USER, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["is_active"], condition=Q(is_active=True),
                                               name="one_active_risk_policy")]


class RedistributionSuggestion(models.Model):
    class Status(models.TextChoices):
        PENDING_REVIEW = "PENDING_REVIEW", "Pending review"
        APPROVED = "APPROVED", "Approved"
        REJECTED = "REJECTED", "Rejected"
        COMPLETED = "COMPLETED", "Completed"
        SUPERSEDED = "SUPERSEDED", "Superseded by newer suggestions"

    medicine = models.ForeignKey("inventory.Medicine", on_delete=models.CASCADE, related_name="redistributions")
    from_facility = models.ForeignKey("inventory.Facility", on_delete=models.CASCADE, related_name="suggested_outgoing")
    to_facility = models.ForeignKey("inventory.Facility", on_delete=models.CASCADE, related_name="suggested_incoming")
    quantity = models.FloatField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING_REVIEW, db_index=True)
    priority_probability = models.FloatField(default=0)
    batch = models.ForeignKey("ml.BatchPredictionJob", null=True, blank=True, on_delete=models.SET_NULL)
    recipient_prediction = models.ForeignKey("ml.PredictionRecord", null=True, blank=True,
                                             on_delete=models.SET_NULL, related_name="+")
    model_label = models.CharField(max_length=120, blank=True)
    details = models.JSONField(default=dict, blank=True)     # quantities, constraints, assumptions, warnings
    generated_at = models.DateTimeField(auto_now_add=True, db_index=True)
    decided_by = models.ForeignKey(USER, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    decided_at = models.DateTimeField(null=True, blank=True)
    decision_note = models.TextField(blank=True)
    completed_by = models.ForeignKey(USER, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    completed_at = models.DateTimeField(null=True, blank=True)
    completion_note = models.TextField(blank=True)

    class Meta:
        ordering = ["-generated_at", "-priority_probability"]


class MonitoringReport(models.Model):
    class Status(models.TextChoices):
        OK = "OK", "OK"
        WARNING = "WARNING", "Warning"
        ALERT = "ALERT", "Alert"

    as_of_date = models.DateField()
    model = models.ForeignKey("ml.ModelVersion", null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    status = models.CharField(max_length=10, choices=Status.choices)
    alerts = models.JSONField(default=list, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    created_by = models.ForeignKey(USER, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]