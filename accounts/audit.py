import json

from .models import AuditLog


def log_action(actor, action, obj=None, rationale="", **metadata):
    return AuditLog.objects.create(
        actor=actor if getattr(actor, "is_authenticated", False) else None,
        action=action,
        object_type=obj.__class__.__name__ if obj is not None else "",
        object_id=str(obj.pk) if obj is not None else "",
        rationale=rationale or "",
        metadata=json.loads(json.dumps(metadata, default=str)),
    )