from django.conf import settings
from django.db import models


class Facility(models.Model):
    class Type(models.TextChoices):
        DISPENSARY = "DISPENSARY", "Dispensary"
        HEALTH_CENTRE = "HEALTH_CENTRE", "Health centre"
        SUB_COUNTY_HOSPITAL = "SUB_COUNTY_HOSPITAL", "Sub-county hospital"
        COUNTY_HOSPITAL = "COUNTY_HOSPITAL", "County hospital"
        REFERRAL_HOSPITAL = "REFERRAL_HOSPITAL", "Referral hospital"

    code = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=200)
    region = models.CharField(max_length=100, db_index=True)
    county = models.CharField(max_length=100, blank=True)
    facility_type = models.CharField(max_length=30, choices=Type.choices, default=Type.HEALTH_CENTRE)
    level = models.PositiveSmallIntegerField(default=3, help_text="KEPH level 2-6, used as a proxy for facility size")
    is_urban = models.BooleanField(default=False)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["code"]

    def __str__(self):
        return f"{self.code} {self.name}"


class Medicine(models.Model):
    code = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=200)
    category = models.CharField(max_length=100, blank=True, db_index=True)
    unit = models.CharField(max_length=30, default="unit")
    is_essential = models.BooleanField(default=True)

    class Meta:
        ordering = ["code"]

    def __str__(self):
        return f"{self.code} {self.name}"


class IngestionBatch(models.Model):
    class Status(models.TextChoices):
        COMPLETED = "COMPLETED", "Completed"
        COMPLETED_WITH_ISSUES = "COMPLETED_WITH_ISSUES", "Completed with issues"
        FAILED = "FAILED", "Failed"

    dataset_type = models.CharField(max_length=20)
    source = models.CharField(max_length=100)
    filename = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.COMPLETED)
    total_rows = models.PositiveIntegerField(default=0)
    accepted_rows = models.PositiveIntegerField(default=0)
    rejected_rows = models.PositiveIntegerField(default=0)
    warning_count = models.PositiveIntegerField(default=0)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]


class InventorySnapshot(models.Model):
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="snapshots")
    medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE, related_name="snapshots")
    date = models.DateField()
    stock_on_hand = models.FloatField()
    earliest_expiry_date = models.DateField(null=True, blank=True)
    source = models.CharField(max_length=100)
    record_id = models.CharField(max_length=100, blank=True)
    ingestion_batch = models.ForeignKey(IngestionBatch, null=True, blank=True, on_delete=models.SET_NULL)
    ingested_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date"]
        constraints = [models.UniqueConstraint(fields=["facility", "medicine", "date"], name="uniq_snapshot")]
        indexes = [models.Index(fields=["date"])]


class ConsumptionRecord(models.Model):
    """Normalised to DAILY quantity; original value and period are kept for traceability."""
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="consumption")
    medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE, related_name="consumption")
    date = models.DateField()
    quantity = models.FloatField()
    original_quantity = models.FloatField()
    original_period_days = models.PositiveSmallIntegerField(default=1)
    source = models.CharField(max_length=100)
    ingestion_batch = models.ForeignKey(IngestionBatch, null=True, blank=True, on_delete=models.SET_NULL)
    ingested_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date"]
        constraints = [models.UniqueConstraint(fields=["facility", "medicine", "date"], name="uniq_consumption")]
        indexes = [models.Index(fields=["date"])]


class Order(models.Model):
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="orders")
    medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE, related_name="orders")
    order_number = models.CharField(max_length=100)
    ordered_date = models.DateField()
    quantity_ordered = models.FloatField()
    expected_delivery_date = models.DateField(null=True, blank=True)
    delivered_date = models.DateField(null=True, blank=True)
    quantity_delivered = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=100)
    ingestion_batch = models.ForeignKey(IngestionBatch, null=True, blank=True, on_delete=models.SET_NULL)
    ingested_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-ordered_date"]
        constraints = [models.UniqueConstraint(fields=["facility", "medicine", "order_number"], name="uniq_order")]

    @property
    def lead_time_days(self):
        return (self.delivered_date - self.ordered_date).days if self.delivered_date else None


class DataQualityIssue(models.Model):
    class Type(models.TextChoices):
        MISSING_FIELD = "MISSING_FIELD", "Missing field"
        INVALID_DATE = "INVALID_DATE", "Invalid date"
        NEGATIVE_VALUE = "NEGATIVE_VALUE", "Negative value"
        DUPLICATE = "DUPLICATE", "Duplicate record"
        UNKNOWN_FACILITY = "UNKNOWN_FACILITY", "Unknown facility"
        UNKNOWN_MEDICINE = "UNKNOWN_MEDICINE", "Unknown medicine"
        MALFORMED = "MALFORMED", "Malformed value"
        ZERO_STOCK_DISPENSING = "ZERO_STOCK_DISPENSING", "Zero stock but dispensing"
        INCONSISTENT_LEAD_TIME = "INCONSISTENT_LEAD_TIME", "Inconsistent lead time"
        MISSING_LEAD_TIME = "MISSING_LEAD_TIME", "Missing lead time"

    class Severity(models.TextChoices):
        WARNING = "WARNING", "Warning (stored, flagged)"
        ERROR = "ERROR", "Error (rejected)"

    class Status(models.TextChoices):
        OPEN = "OPEN", "Open"
        RESOLVED = "RESOLVED", "Resolved"
        IGNORED = "IGNORED", "Ignored"

    batch = models.ForeignKey(IngestionBatch, null=True, blank=True, on_delete=models.CASCADE, related_name="issues")
    dataset_type = models.CharField(max_length=20)
    issue_type = models.CharField(max_length=30, choices=Type.choices, db_index=True)
    severity = models.CharField(max_length=10, choices=Severity.choices)
    record_ref = models.CharField(max_length=200, blank=True)
    facility = models.ForeignKey(Facility, null=True, blank=True, on_delete=models.SET_NULL)
    medicine = models.ForeignKey(Medicine, null=True, blank=True, on_delete=models.SET_NULL)
    raw_payload = models.JSONField(default=dict, blank=True)   # raw source row is always preserved
    message = models.TextField()
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.OPEN, db_index=True)
    resolution_note = models.TextField(blank=True)
    resolved_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                                    related_name="+")
    resolved_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]