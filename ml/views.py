from pathlib import Path

from django.db import transaction
from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import NotFound, PermissionDenied, ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.audit import log_action
from accounts.permissions import ALL_ROLES, FULL_ACCESS, Role, RolePermission, can_access_facility, scope_queryset
from inventory.models import Facility, Medicine

from .models import BatchPredictionJob, DatasetVersion, ModelVersion, PredictionRecord
from .serializers import (BatchJobSerializer, BatchRequestSerializer, DatasetVersionSerializer,
                          ModelVersionDetailSerializer, ModelVersionSerializer, PredictionRecordSerializer,
                          PredictRequestSerializer)
from .services import predict_single, run_batch


class DatasetViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = DatasetVersion.objects.all()
    serializer_class = DatasetVersionSerializer
    permission_classes = [RolePermission]
    allowed_roles = (Role.ADMIN, Role.ANALYST)


class ModelVersionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelVersion.objects.select_related("dataset")
    permission_classes = [RolePermission]
    allowed_roles_by_action = {"active": ALL_ROLES, "promote": (Role.ADMIN,), "*": FULL_ACCESS}
    filterset_fields = ["status", "algorithm"]

    def get_serializer_class(self):
        return ModelVersionDetailSerializer if self.action in ("retrieve", "active") else ModelVersionSerializer

    @action(detail=False, methods=["get"])
    def active(self, request):
        mv = ModelVersion.objects.filter(status=ModelVersion.Status.ACTIVE).first()
        if mv is None:
            raise NotFound("No active model.")
        data = ModelVersionDetailSerializer(mv).data
        if request.user and not request.user.is_superuser:
            data.pop("subgroup_report", None)
        return Response(data)

    @action(detail=True, methods=["get"])
    def card(self, request, pk=None):
        return Response(self.get_object().model_card)

    @action(detail=True, methods=["get"])
    def subgroups(self, request, pk=None):
        mv = self.get_object()
        return Response({"requires_review": mv.requires_review, **mv.subgroup_report})

    @action(detail=True, methods=["post"])
    def promote(self, request, pk=None):
        mv = self.get_object()
        rationale = str(request.data.get("rationale", "")).strip()
        if not rationale:
            raise ValidationError({"rationale": "A written rationale is required to promote a model."})
        if not mv.artifact_path or not Path(mv.artifact_path).exists():
            raise ValidationError("This model version has no loadable artifact and cannot be promoted.")
        if mv.requires_review and request.data.get("acknowledge_review") not in (True, "true", "True"):
            raise ValidationError({"acknowledge_review": "Subgroup disparities were flagged. Review "
                                   f"/ml/models/{mv.pk}/subgroups/ and resend with acknowledge_review=true."})
        with transaction.atomic():
            ModelVersion.objects.filter(status=ModelVersion.Status.ACTIVE).exclude(pk=mv.pk) \
                .update(status=ModelVersion.Status.ARCHIVED)
            mv.status, mv.promoted_by, mv.promoted_at = ModelVersion.Status.ACTIVE, request.user, timezone.now()
            if mv.requires_review:
                mv.review_note = rationale
            mv.save()
        log_action(request.user, "model.promoted", mv, rationale=rationale,
                   requires_review=mv.requires_review, threshold=mv.threshold)
        return Response(ModelVersionSerializer(mv).data)


class PredictView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES
    throttle_scope = "predictions"

    @extend_schema(request=PredictRequestSerializer, responses={200: PredictionRecordSerializer})
    def post(self, request):
        ser = PredictRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        v = ser.validated_data
        facility = Facility.objects.filter(code=v["facility_code"], is_active=True).first()
        medicine = Medicine.objects.filter(code=v["medicine_code"]).first()
        if facility is None or medicine is None:
            raise NotFound("Unknown facility_code or medicine_code.")
        if not can_access_facility(request.user, facility.id):
            raise PermissionDenied("You do not have access to this facility.")
        record = predict_single(request.user, facility, medicine, v.get("as_of_date"), v.get("features"))
        return Response(PredictionRecordSerializer(record).data)


class BatchPredictView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = FULL_ACCESS
    throttle_scope = "batch"

    @extend_schema(request=BatchRequestSerializer, responses={201: BatchJobSerializer})
    def post(self, request):
        ser = BatchRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        v = ser.validated_data
        fac_ids = med_ids = None
        if v.get("facility_codes"):
            fac_ids = list(Facility.objects.filter(code__in=v["facility_codes"]).values_list("id", flat=True))
            if len(fac_ids) != len(set(v["facility_codes"])):
                raise ValidationError({"facility_codes": "One or more facility codes are unknown."})
        if v.get("medicine_codes"):
            med_ids = list(Medicine.objects.filter(code__in=v["medicine_codes"]).values_list("id", flat=True))
            if len(med_ids) != len(set(v["medicine_codes"])):
                raise ValidationError({"medicine_codes": "One or more medicine codes are unknown."})
        job = run_batch(request.user, v.get("as_of_date"), fac_ids, med_ids)
        log_action(request.user, "prediction.batch", job, succeeded=job.succeeded, failed=job.failed)
        return Response(BatchJobSerializer(job).data, status=201)


class BatchJobViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = BatchPredictionJob.objects.select_related("model")
    serializer_class = BatchJobSerializer
    permission_classes = [RolePermission]
    allowed_roles = FULL_ACCESS


class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = PredictionRecord.objects.select_related("facility", "medicine", "model")
    serializer_class = PredictionRecordSerializer
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES
    filterset_fields = {"facility": ["exact"], "medicine": ["exact"], "batch": ["exact"],
                    "as_of_date": ["exact", "gte", "lte"], "predicted_stockout": ["exact"],
                    "model_applied": ["exact"]}
    ordering_fields = ["probability", "as_of_date", "created_at"]

    def get_queryset(self):
        return scope_queryset(super().get_queryset(), self.request.user)