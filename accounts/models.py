from django.conf import settings
from django.db import models


class Role(models.TextChoices):
    ADMIN = "ADMIN", "Administrator"
    ANALYST = "ANALYST", "Analyst"
    SUPPLY_MANAGER = "SUPPLY_MANAGER", "Supply-chain manager"
    FACILITY_USER = "FACILITY_USER", "Facility user"


class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.FACILITY_USER)
    # Only used for FACILITY_USER: the facilities they may see
    facilities = models.ManyToManyField("inventory.Facility", blank=True, related_name="users")

    def __str__(self):
        return f"{self.user} ({self.role})"


class AuditLog(models.Model):
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    action = models.CharField(max_length=100, db_index=True)
    object_type = models.CharField(max_length=100, blank=True)
    object_id = models.CharField(max_length=64, blank=True)
    rationale = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]