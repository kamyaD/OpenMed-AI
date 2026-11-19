import csv

from django.utils import timezone
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import NotFound, ValidationError
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.audit import log_action
from accounts.permissions import ALL_ROLES, FULL_ACCESS, Role, RolePermission, scope_queryset

from .ingestion import HANDLERS, ingest, read_rows
from .models import (ConsumptionRecord, DataQualityIssue, Facility, IngestionBatch,
                     InventorySnapshot, Medicine, Order)
from .quality import quality_report
from .serializers import (ConsumptionSerializer, DataQualityIssueSerializer, FacilitySerializer,
                          IngestionBatchSerializer, IngestionRequestSerializer, InventorySnapshotSerializer,
                          MedicineSerializer, OrderSerializer)

READ_ALL_WRITE_ADMIN = {"list": ALL_ROLES, "retrieve": ALL_ROLES, "*": (Role.ADMIN,)}


class FacilityViewSet(viewsets.ModelViewSet):
    queryset = Facility.objects.all()
    serializer_class = FacilitySerializer
    permission_classes = [RolePermission]
    allowed_roles_by_action = READ_ALL_WRITE_ADMIN
    filterset_fields = ["region", "county", "level", "is_urban", "facility_type", "is_active"]
    search_fields = ["code", "name"]

    def get_queryset(self):
        return scope_queryset(super().get_queryset(), self.request.user, "id")


class MedicineViewSet(viewsets.ModelViewSet):
    queryset = Medicine.objects.all()
    serializer_class = MedicineSerializer
    permission_classes = [RolePermission]
    allowed_roles_by_action = READ_ALL_WRITE_ADMIN
    filterset_fields = ["category", "is_essential"]
    search_fields = ["code", "name"]


class _ScopedReadOnly(viewsets.ReadOnlyModelViewSet):
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES
    filterset_fields = {"facility": ["exact"], "medicine": ["exact"]}

    def get_queryset(self):
        return scope_queryset(super().get_queryset().select_related("facility", "medicine"), self.request.user)


class InventorySnapshotViewSet(_ScopedReadOnly):
    queryset = InventorySnapshot.objects.all()
    serializer_class = InventorySnapshotSerializer
    filterset_fields = {"facility": ["exact"], "medicine": ["exact"], "date": ["gte", "lte"]}


class ConsumptionViewSet(_ScopedReadOnly):
    queryset = ConsumptionRecord.objects.all()
    serializer_class = ConsumptionSerializer
    filterset_fields = {"facility": ["exact"], "medicine": ["exact"], "date": ["gte", "lte"]}


class OrderViewSet(_ScopedReadOnly):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer
    filterset_fields = {"facility": ["exact"], "medicine": ["exact"], "ordered_date": ["gte", "lte"]}


class IngestionBatchViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = IngestionBatch.objects.all()
    serializer_class = IngestionBatchSerializer
    permission_classes = [RolePermission]
    allowed_roles = FULL_ACCESS
    filterset_fields = ["dataset_type", "source", "status"]


class IngestionView(APIView):
    parser_classes = [MultiPartParser, JSONParser]
    permission_classes = [RolePermission]
    allowed_roles = (Role.ADMIN, Role.ANALYST)
    throttle_scope = "ingestion"

    @extend_schema(request=IngestionRequestSerializer, responses={201: IngestionBatchSerializer})
    def post(self, request, dataset):
        dataset_type = dataset.upper()
        if dataset_type not in HANDLERS:
            raise NotFound(f"Unknown dataset '{dataset}'. Use inventory, consumption or orders.")
        ser = IngestionRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        f, records = ser.validated_data.get("file"), ser.validated_data.get("records")
        if f is None and records is None:
            raise ValidationError("Provide a CSV/JSON 'file' or a 'records' list.")
        try:
            rows = read_rows(f, records)
        except (UnicodeDecodeError, ValueError, KeyError, csv.Error) as e:
            raise ValidationError(f"Could not parse upload: {e}")
        batch = ingest(dataset_type, rows, ser.validated_data["source"], request.user,
                       filename=getattr(f, "name", ""))
        log_action(request.user, "ingestion.completed", batch, dataset=dataset_type,
                   accepted=batch.accepted_rows, rejected=batch.rejected_rows)
        return Response(IngestionBatchSerializer(batch).data, status=201)


class DataQualityIssueViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = DataQualityIssue.objects.select_related("facility", "medicine")
    serializer_class = DataQualityIssueSerializer
    permission_classes = [RolePermission]
    allowed_roles_by_action = {"list": ALL_ROLES, "retrieve": ALL_ROLES, "*": (Role.ADMIN, Role.ANALYST)}
    filterset_fields = ["issue_type", "severity", "status", "facility", "medicine", "dataset_type", "batch"]

    def get_queryset(self):
        return scope_queryset(super().get_queryset(), self.request.user)

    @action(detail=True, methods=["post"])
    def resolve(self, request, pk=None):
        issue = self.get_object()
        new_status = request.data.get("status", "RESOLVED")
        if new_status not in ("RESOLVED", "IGNORED"):
            raise ValidationError({"status": "Must be RESOLVED or IGNORED."})
        note = str(request.data.get("note", "")).strip()
        if not note:
            raise ValidationError({"note": "A resolution note is required."})
        issue.status, issue.resolution_note = new_status, note
        issue.resolved_by, issue.resolved_at = request.user, timezone.now()
        issue.save()
        log_action(request.user, f"data_quality.{new_status.lower()}", issue, rationale=note)
        return Response(DataQualityIssueSerializer(issue).data)


class QualityReportView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = FULL_ACCESS

    @extend_schema(parameters=[
        OpenApiParameter("days", int, description="Window length, 1-365 (default 30)"),
        OpenApiParameter("group_by", str, enum=["facility", "medicine", "source"]),
    ])
    def get(self, request):
        try:
            days = max(1, min(int(request.query_params.get("days", 30)), 365))
        except ValueError:
            raise ValidationError({"days": "Must be an integer."})
        group_by = request.query_params.get("group_by", "facility")
        if group_by not in ("facility", "medicine", "source"):
            raise ValidationError({"group_by": "Must be facility, medicine or source."})
        return Response(quality_report(days, group_by))