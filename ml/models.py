import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q


class DatasetVersion(models.Model):
    name = models.CharField(max_length=100)
    horizon_days = models.PositiveSmallIntegerField()
    start_date = models.DateField()
    end_date = models.DateField()
    row_count = models.PositiveIntegerField(default=0)
    positive_count = models.PositiveIntegerField(default=0)
    positive_rate = models.FloatField(default=0)
    file_path = models.CharField(max_length=500)
    checksum = models.CharField(max_length=64)
    label_definition = models.TextField()
    exclusions = models.JSONField(default=dict, blank=True)   # rows dropped and why
    split = models.JSONField(default=dict, blank=True)        # chronological boundaries
    params = models.JSONField(default=dict, blank=True)
    code_version = models.CharField(max_length=40, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class ModelVersion(models.Model):
    class Status(models.TextChoices):
        CANDIDATE = "CANDIDATE", "Candidate"
        ACTIVE = "ACTIVE", "Active"
        ARCHIVED = "ARCHIVED", "Archived"

    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=30)     # RULE_BASELINE | LOGISTIC_REGRESSION | RANDOM_FOREST
    dataset = models.ForeignKey(DatasetVersion, null=True, blank=True, on_delete=models.SET_NULL)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.CANDIDATE, db_index=True)
    artifact_path = models.CharField(max_length=500, blank=True)
    feature_schema = models.JSONField(default=dict, blank=True)
    threshold = models.FloatField(default=0.5)
    calibration = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    subgroup_report = models.JSONField(default=dict, blank=True)
    requires_review = models.BooleanField(default=False)
    review_note = models.TextField(blank=True)
    model_card = models.JSONField(default=dict, blank=True)
    code_version = models.CharField(max_length=40, blank=True)
    library_versions = models.JSONField(default=dict, blank=True)
    trained_at = models.DateTimeField(null=True, blank=True)
    promoted_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                    on_delete=models.SET_NULL, related_name="+")
    promoted_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["status"], condition=Q(status="ACTIVE"),
                                               name="one_active_model")]

    @property
    def label(self):
        return f"{self.name}-{self.pk}"

    def __str__(self):
        return self.label


class BatchPredictionJob(models.Model):
    class Status(models.TextChoices):
        COMPLETED = "COMPLETED", "Completed"
        COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS", "Completed with errors"
        FAILED = "FAILED", "Failed"

    model = models.ForeignKey(ModelVersion, on_delete=models.PROTECT)
    as_of_date = models.DateField()
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.COMPLETED)
    total = models.PositiveIntegerField(default=0)
    succeeded = models.PositiveIntegerField(default=0)
    failed = models.PositiveIntegerField(default=0)
    errors = models.JSONField(default=list, blank=True)       # capped at 200 entries
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]


class PredictionRecord(models.Model):
    request_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    model = models.ForeignKey(ModelVersion, on_delete=models.PROTECT, related_name="predictions")
    facility = models.ForeignKey("inventory.Facility", on_delete=models.CASCADE, related_name="predictions")
    medicine = models.ForeignKey("inventory.Medicine", on_delete=models.CASCADE, related_name="predictions")
    as_of_date = models.DateField()
    horizon_days = models.PositiveSmallIntegerField()
    probability = models.FloatField()
    predicted_stockout = models.BooleanField()
    threshold = models.FloatField()
    probability_calibrated = models.BooleanField(default=False)
    model_applied = models.BooleanField(default=True)        # False when already out of stock
    features = models.JSONField(default=dict, blank=True)
    warnings = models.JSONField(default=list, blank=True)
    batch = models.ForeignKey(BatchPredictionJob, null=True, blank=True, on_delete=models.CASCADE,
                              related_name="predictions")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["facility", "medicine", "as_of_date"])]