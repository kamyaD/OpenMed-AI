import csv
import math
from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpResponse
from django.utils import timezone
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import APIException, NotFound, PermissionDenied, ValidationError
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.audit import log_action
from accounts.models import AuditLog
from accounts.permissions import (ALL_ROLES, FULL_ACCESS, Role, RolePermission, can_access_facility,
                                  facility_ids_for, get_role, scope_queryset)
from inventory.models import ConsumptionRecord, DataQualityIssue, Facility, InventorySnapshot, Medicine, Order
from ml.models import BatchPredictionJob, ModelVersion, PredictionRecord

from .config import cfg
from .coverage import compute_coverage
from .dashboard import base_predictions, csv_safe, prediction_row
from .explain import explain_prediction
from .lifecycle import RETRAIN_CRITERIA, retrain_recommendation, rollback
from .models import MonitoringReport, RedistributionSuggestion, RiskPolicy
from .monitoring import RULES, run_monitoring
from .redistribution import generate_suggestions
from .risk import CATEGORIES, categorize, category_q, clear_policy_cache, get_policy, validate_bands
from .serializers import MonitoringReportSerializer, RedistributionSerializer

ST = RedistributionSuggestion.Status


class Conflict(APIException):
    status_code = 409
    default_detail = "Action not allowed in the current status."
    default_code = "conflict"


# ------------------------------------------------------------------ risk policy
class RiskPolicyView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES

    def get(self, request):
        data = {**get_policy(), "categories": CATEGORIES,
                "description": "Probability lower bounds per band; below MEDIUM is LOW. STOCKED_OUT is "
                               "assigned when the facility is already out of stock (model not applied)."}
        if facility_ids_for(request.user) is None:
            data["history"] = list(RiskPolicy.objects.values("id", "bands", "rationale", "is_active", "created_at",
                                                             "created_by__username")[:10])
        return Response(data)

    def post(self, request):
        if get_role(request.user) != Role.ADMIN:
            raise PermissionDenied("Only administrators can change risk bands.")
        rationale = str(request.data.get("rationale", "")).strip()
        if not rationale:
            raise ValidationError({"rationale": "A written rationale is required."})
        try:
            bands = validate_bands(request.data.get("bands"))
        except ValueError as e:
            raise ValidationError({"bands": str(e)})
        with transaction.atomic():
            RiskPolicy.objects.filter(is_active=True).update(is_active=False)
            pol = RiskPolicy.objects.create(bands=bands, rationale=rationale, created_by=request.user)
        clear_policy_cache()
        log_action(request.user, "risk_policy.changed", pol, rationale=rationale, bands=bands)
        return Response({"id": pol.pk, "bands": bands, "provisional": False}, status=201)


# ------------------------------------------------------------------ redistribution + approvals
def scoped_suggestions(user, qs=None):
    qs = RedistributionSuggestion.objects.all() if qs is None else qs
    ids = facility_ids_for(user)
    return qs if ids is None else qs.filter(Q(from_facility_id__in=ids) | Q(to_facility_id__in=ids))


class RedistributionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RedistributionSuggestion.objects.select_related(
        "medicine", "from_facility", "to_facility", "decided_by", "completed_by")
    serializer_class = RedistributionSerializer
    permission_classes = [RolePermission]
    allowed_roles_by_action = {"list": ALL_ROLES, "retrieve": ALL_ROLES, "history": ALL_ROLES,
                               "generate": FULL_ACCESS, "*": (Role.ADMIN, Role.SUPPLY_MANAGER)}
    filterset_fields = ["status", "medicine", "from_facility", "to_facility", "batch"]
    ordering_fields = ["generated_at", "priority_probability", "quantity"]

    def get_queryset(self):
        return scoped_suggestions(self.request.user, super().get_queryset())

    @action(detail=False, methods=["post"])
    def generate(self, request):
        batch = None
        if request.data.get("batch"):
            try:
                batch = BatchPredictionJob.objects.filter(pk=int(request.data["batch"])).first()
            except (TypeError, ValueError):
                raise ValidationError({"batch": "Must be an integer."})
            if batch is None:
                raise NotFound("Unknown batch.")
        try:
            summary = generate_suggestions(batch)
        except ValueError as e:
            raise ValidationError(str(e))
        log_action(request.user, "redistribution.generated", None, **summary)
        return Response(summary, status=201)

    @action(detail=True, methods=["get"])
    def history(self, request, pk=None):
        obj = self.get_object()
        logs = AuditLog.objects.filter(object_type="RedistributionSuggestion", object_id=str(obj.pk)).order_by("created_at")
        return Response([{"at": a.created_at, "actor": str(a.actor) if a.actor else None, "action": a.action,
                          "rationale": a.rationale} for a in logs])

    @action(detail=True, methods=["post"])
    def approve(self, request, pk=None):
        return self._transition(request, (ST.PENDING_REVIEW,), ST.APPROVED, "redistribution.approved", True, True)

    @action(detail=True, methods=["post"])
    def reject(self, request, pk=None):
        return self._transition(request, (ST.PENDING_REVIEW,), ST.REJECTED, "redistribution.rejected", True)

    @action(detail=True, methods=["post"])
    def complete(self, request, pk=None):
        extra = None
        if request.data.get("quantity_moved") is not None:
            try:
                moved = float(request.data["quantity_moved"])
            except (TypeError, ValueError):
                raise ValidationError({"quantity_moved": "Must be a number."})
            if moved < 0 or not math.isfinite(moved):
                raise ValidationError({"quantity_moved": "Must be zero or positive."})
            extra = {"completion": {"quantity_moved": moved}}
        return self._transition(request, (ST.APPROVED,), ST.COMPLETED, "redistribution.completed", False, False, extra)

    def _transition(self, request, allowed_from, new_status, action_name, rationale_required, check_stale=False,
                    extra=None):
        obj = self.get_object()
        note = str(request.data.get("rationale") or request.data.get("note") or "").strip()
        if rationale_required and not note:
            raise ValidationError({"rationale": "A written rationale is required."})
        with transaction.atomic():
            s = RedistributionSuggestion.objects.select_for_update().get(pk=obj.pk)
            if s.status not in allowed_from:
                raise Conflict(f"Suggestion is {s.status}; expected {' or '.join(allowed_from)}.")
            if check_stale and (timezone.now() - s.generated_at).days > cfg("SUGGESTION_MAX_AGE_DAYS"):
                raise ValidationError("This suggestion is stale. Generate fresh suggestions before approving.")
            now = timezone.now()
            if new_status in (ST.APPROVED, ST.REJECTED):
                s.decided_by, s.decided_at, s.decision_note = request.user, now, note
            else:
                s.completed_by, s.completed_at, s.completion_note = request.user, now, note
            if extra:
                s.details = {**s.details, **extra}
            s.status = new_status
            s.save()
            log_action(request.user, action_name, s, rationale=note, medicine=s.medicine.code,
                       from_facility=s.from_facility.code, to_facility=s.to_facility.code, quantity=s.quantity)
        return Response(RedistributionSerializer(s).data)


# ------------------------------------------------------------------ dashboard
class OverviewView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES

    def get(self, request):
        policy, today = get_policy(), timezone.localdate()
        base = base_predictions(request, apply_risk=False)
        agg = base.aggregate(total=Count("id"),
                             **{c.lower(): Count("id", filter=category_q(c, policy)) for c in CATEGORIES})
        newest = base.order_by("-as_of_date", "-created_at").first()
        summary = {
            "counts": agg,
            "facilities_with_critical_or_high": base.filter(
                category_q("CRITICAL", policy) | category_q("HIGH", policy)).values("facility_id").distinct().count(),
            "as_of_date": newest.as_of_date if newest else None,
            "prediction_age_days": (today - newest.as_of_date).days if newest else None,
            "model_version": newest.model.label if newest else None,
            "probability_calibrated": base.filter(model_applied=True).values_list(
                "probability_calibrated", flat=True).first(),
            "risk_policy": {**policy["bands"], "provisional": policy["provisional"]},
            "pending_redistribution_suggestions": scoped_suggestions(
                request.user, RedistributionSuggestion.objects.filter(status=ST.PENDING_REVIEW)).count(),
            "note": "STOCKED_OUT rows are already out of stock; they are not model predictions.",
        }
        qs = base_predictions(request)
        order = request.query_params.get("ordering", "-probability")
        if order not in ("-probability", "probability", "as_of_date", "-as_of_date"):
            raise ValidationError({"ordering": "Use probability, -probability, as_of_date or -as_of_date."})
        qs = qs.order_by(order, "facility__code", "medicine__code")
        paginator = PageNumberPagination()
        paginator.page_size, paginator.page_size_query_param, paginator.max_page_size = 50, "page_size", 200
        page = paginator.paginate_queryset(qs, request, view=self)
        data = paginator.get_paginated_response([prediction_row(p, policy, today) for p in page]).data
        return Response({"summary": summary, **data})


class FiltersView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES

    def get(self, request):
        facs = list(scope_queryset(Facility.objects.filter(is_active=True), request.user, "id")
                    .values("code", "name", "region"))
        return Response({
            "regions": sorted({f["region"] for f in facs}), "facilities": facs,
            "medicines": list(Medicine.objects.filter(is_essential=True).values("code", "name", "category")),
            "risk_categories": CATEGORIES, "kinds": ["all", "predicted", "stocked_out"],
            "batches": [{"id": b.pk, "as_of_date": b.as_of_date, "model": b.model.label}
                        for b in BatchPredictionJob.objects.select_related("model").order_by("-created_at")[:10]]})


class DetailView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES

    def get(self, request):
        fc, mc = request.query_params.get("facility_code"), request.query_params.get("medicine_code")
        if not fc or not mc:
            raise ValidationError("facility_code and medicine_code are required.")
        try:
            days = max(7, min(int(request.query_params.get("days", 60)), 180))
        except ValueError:
            raise ValidationError({"days": "Must be an integer."})
        facility, medicine = Facility.objects.filter(code=fc).first(), Medicine.objects.filter(code=mc).first()
        if facility is None or medicine is None:
            raise NotFound("Unknown facility_code or medicine_code.")
        if not can_access_facility(request.user, facility.id):
            raise PermissionDenied("You do not have access to this facility.")

        today = timezone.localdate()
        pair = {"facility": facility, "medicine": medicine}
        last = InventorySnapshot.objects.filter(**pair).order_by("-date").first()
        as_of = last.date if last else today
        start = as_of - timedelta(days=days)

        stock = [{"date": s.date, "stock_on_hand": s.stock_on_hand}
                 for s in InventorySnapshot.objects.filter(**pair, date__gte=start).order_by("date")]
        cons, window = [], []
        for c in ConsumptionRecord.objects.filter(**pair, date__gte=start).order_by("date"):
            window = (window + [c.quantity])[-7:]
            cons.append({"date": c.date, "quantity": c.quantity, "avg_7_observations": round(sum(window) / len(window), 2)})
        lts = [o.lead_time_days for o in Order.objects.filter(**pair, delivered_date__isnull=False)
               .order_by("-delivered_date")[:50] if o.lead_time_days is not None and o.lead_time_days >= 0]
        lead = {"n": len(lts), "mean": round(sum(lts) / len(lts), 1) if lts else None,
                "min": min(lts) if lts else None, "max": max(lts) if lts else None, "recent": lts[:5]}
        orders = [{"order_number": o.order_number, "ordered_date": o.ordered_date, "quantity_ordered": o.quantity_ordered,
                   "expected_delivery_date": o.expected_delivery_date, "delivered_date": o.delivered_date,
                   "lead_time_days": o.lead_time_days} for o in Order.objects.filter(**pair).order_by("-ordered_date")[:10]]

        policy = get_policy()
        preds = list(PredictionRecord.objects.filter(**pair).select_related("model").order_by("-created_at")[:10])
        history = [{"request_id": str(p.request_id), "as_of_date": p.as_of_date, "created_at": p.created_at,
                    "probability": round(p.probability, 4), "model_applied": p.model_applied,
                    "model_version": p.model.label, "risk_category": categorize(p.probability, p.model_applied, policy)}
                   for p in preds]
        
        for p in preds:
            history.append({"request_id": str(p.request_id), "as_of_date": p.as_of_date, "created_at": p.created_at,
                            "probability": round(p.probability, 4), "model_applied": p.model_applied,
                            "model_version": p.model.label,
                            "risk_category": categorize(p.probability, p.model_applied, policy)})

        recent = InventorySnapshot.objects.filter(**pair, date__gt=as_of - timedelta(days=28)).count()
        issues = {r["issue_type"]: r["c"] for r in DataQualityIssue.objects.filter(
            facility=facility, medicine=medicine, status="OPEN").values("issue_type").annotate(c=Count("id"))}
        flags = []
        if last is None:
            flags.append("No inventory records exist for this combination.")
        elif (today - last.date).days > settings.MSP["FRESHNESS_ALERT_DAYS"]:
            flags.append(f"Latest stock record is {(today - last.date).days} days old.")
        if recent < 21:
            flags.append(f"Only {recent} of the last 28 days have stock records.")
        if issues:
            flags.append(f"{sum(issues.values())} open data-quality issue(s).")
        if issues.get("ZERO_STOCK_DISPENSING"):
            flags.append("Stock recorded as zero on days with dispensing; stock figures may be unreliable.")

        return Response({
            "facility": {"code": facility.code, "name": facility.name, "region": facility.region, "level": facility.level},
            "medicine": {"code": medicine.code, "name": medicine.name, "category": medicine.category, "unit": medicine.unit},
            "as_of_date": as_of, "stock_series": stock, "consumption_series": cons, "lead_time": lead,
            "recent_orders": orders, "coverage": compute_coverage(facility, medicine, as_of),
            "latest_prediction": history[0] if history else None, "prediction_history": history,
            "explanation": explain_prediction(preds[0]) if preds else None,
            "data_quality": {"reliable": not flags, "flags": flags, "open_issues_by_type": issues,
                             "last_record_date": last.date if last else None, "source": last.source if last else None,
                             "last_ingested_at": last.ingested_at if last else None,
                             "ingestion_batch": last.ingestion_batch_id if last else None}})


class ExplanationView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES

    def get(self, request, request_id):
        rec = scope_queryset(PredictionRecord.objects.select_related("model"), request.user) \
            .filter(request_id=request_id).first()
        if rec is None:
            raise NotFound("Prediction not found.")
        return Response(explain_prediction(rec))


class ExportView(APIView):
    """CSV of the filtered dashboard view. Rows are scoped to the user's facilities."""
    permission_classes = [RolePermission]
    allowed_roles = ALL_ROLES
    throttle_scope = "batch"

    def get(self, request):
        policy, today = get_policy(), timezone.localdate()
        qs = base_predictions(request).order_by("-probability")[:cfg("MAX_EXPORT_ROWS")]
        cols = ["facility_code", "facility_name", "region", "medicine_code", "medicine_name", "medicine_category",
                "as_of_date", "prediction_age_days", "probability", "risk_category", "predicted_stockout",
                "model_applied", "probability_calibrated", "stock_on_hand", "days_of_stock", "model_version"]
        resp = HttpResponse(content_type="text/csv")
        resp["Content-Disposition"] = f'attachment; filename="msp_risk_{today.isoformat()}.csv"'
        w, n = csv.writer(resp), 0
        w.writerow(cols)
        for p in qs:
            row = prediction_row(p, policy, today)
            w.writerow([csv_safe(row[c]) for c in cols])
            n += 1
        log_action(request.user, "dashboard.export", None, rows=n, filters=dict(request.query_params))
        return resp


# ------------------------------------------------------------------ monitoring and lifecycle
class MonitoringReportViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = MonitoringReport.objects.select_related("model")
    serializer_class = MonitoringReportSerializer
    permission_classes = [RolePermission]
    allowed_roles = FULL_ACCESS

    @action(detail=False, methods=["get"])
    def latest(self, request):
        r = self.get_queryset().first()
        if r is None:
            raise NotFound("No monitoring report yet. POST /monitoring/run/ first.")
        return Response(MonitoringReportSerializer(r).data)


class RunMonitoringView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = (Role.ADMIN, Role.ANALYST)
    throttle_scope = "batch"

    def post(self, request):
        report = run_monitoring(request.user)
        log_action(request.user, "monitoring.run", report, status=report.status, alerts=len(report.alerts))
        return Response(MonitoringReportSerializer(report).data, status=201)


class RulesView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = FULL_ACCESS

    def get(self, request):
        return Response({"alert_rules": RULES, "retraining_criteria": RETRAIN_CRITERIA})


class LifecycleStatusView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = FULL_ACCESS

    def get(self, request):
        active = ModelVersion.objects.filter(status=ModelVersion.Status.ACTIVE).first()
        cands = ModelVersion.objects.filter(status=ModelVersion.Status.ARCHIVED, promoted_at__isnull=False) \
            .exclude(artifact_path="").order_by("-promoted_at")[:5]
        return Response({
            "active_model": {"id": active.pk, "label": active.label, "promoted_at": active.promoted_at,
                             "trained_at": active.trained_at, "threshold": active.threshold} if active else None,
            "retraining": retrain_recommendation(MonitoringReport.objects.first()),
            "rollback_candidates": [{"id": m.pk, "label": m.label, "promoted_at": m.promoted_at} for m in cands]})


class RollbackView(APIView):
    permission_classes = [RolePermission]
    allowed_roles = (Role.ADMIN,)

    def post(self, request):
        target = rollback(request.user, request.data.get("rationale"), request.data.get("model_id"))
        return Response({"active_model": target.label, "status": target.status})