from django.db import connection
from rest_framework import generics
from rest_framework.decorators import api_view, authentication_classes, permission_classes, throttle_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import AuditLog, Role
from .permissions import RolePermission, facility_ids_for, get_role
from .serializers import AuditLogSerializer


class MeView(APIView):
    def get(self, request):
        return Response({
            "username": request.user.username,
            "role": str(get_role(request.user)),
            "facility_ids": facility_ids_for(request.user),   # null = all facilities
        })


class AuditLogListView(generics.ListAPIView):
    queryset = AuditLog.objects.select_related("actor")
    serializer_class = AuditLogSerializer
    permission_classes = [RolePermission]
    allowed_roles = (Role.ADMIN,)
    filterset_fields = ["action", "object_type", "actor"]


@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
@throttle_classes([])
def health(request):
    try:
        connection.ensure_connection()
        ok = True
    except Exception:
        ok = False
    return Response({"status": "ok" if ok else "degraded", "database": "ok" if ok else "error"},
                    status=200 if ok else 503)