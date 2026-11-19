from rest_framework.permissions import BasePermission

from .models import Role

ALL_ROLES = tuple(Role)
FULL_ACCESS = (Role.ADMIN, Role.ANALYST, Role.SUPPLY_MANAGER)


def get_role(user):
    if not user or not user.is_authenticated:
        return None
    if user.is_superuser:
        return Role.ADMIN
    profile = getattr(user, "profile", None)
    return profile.role if profile else Role.FACILITY_USER   # deny-by-default: no facilities assigned


def facility_ids_for(user):
    """None = unrestricted. Otherwise the list of facility ids the user may access."""
    if get_role(user) in FULL_ACCESS:
        return None
    profile = getattr(user, "profile", None)
    return list(profile.facilities.values_list("id", flat=True)) if profile else []


def scope_queryset(qs, user, field="facility_id"):
    ids = facility_ids_for(user)
    return qs if ids is None else qs.filter(**{f"{field}__in": ids})


def can_access_facility(user, facility_id):
    ids = facility_ids_for(user)
    return ids is None or facility_id in ids


class RolePermission(BasePermission):
    """
    Views declare `allowed_roles` (tuple) and/or `allowed_roles_by_action`
    ({"list": (...), "*": (...)}). Unauthenticated users are always rejected.
    """
    def has_permission(self, request, view):
        role = get_role(request.user)
        if role is None:
            return False
        by_action = getattr(view, "allowed_roles_by_action", None)
        allowed = None
        if by_action:
            allowed = by_action.get(getattr(view, "action", None), by_action.get("*"))
        if allowed is None:
            allowed = getattr(view, "allowed_roles", None)
        return allowed is None or role in allowed