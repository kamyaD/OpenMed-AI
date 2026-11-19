import math

from rest_framework import serializers

from .features import NUMERIC_FEATURES
from .models import BatchPredictionJob, DatasetVersion, ModelVersion, PredictionRecord
from decision.risk import get_policy
from decision.risk import categorize


class DatasetVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetVersion
        exclude = ["file_path"]


class ModelVersionSerializer(serializers.ModelSerializer):
    label = serializers.CharField(read_only=True)

    class Meta:
        model = ModelVersion
        exclude = ["artifact_path", "subgroup_report", "model_card"]


class ModelVersionDetailSerializer(serializers.ModelSerializer):
    label = serializers.CharField(read_only=True)

    class Meta:
        model = ModelVersion
        exclude = ["artifact_path"]


class PredictRequestSerializer(serializers.Serializer):
    facility_code = serializers.CharField()
    medicine_code = serializers.CharField()
    as_of_date = serializers.DateField(required=False)
    features = serializers.DictField(child=serializers.FloatField(), required=False,
                                     help_text="Optional what-if overrides of computed features")

    def validate_features(self, value):
        unknown = sorted(set(value) - set(NUMERIC_FEATURES))
        if unknown:
            raise serializers.ValidationError(f"Unknown feature(s): {unknown}. Allowed: {NUMERIC_FEATURES}")
        bad = [k for k, v in value.items() if not math.isfinite(v)]
        if bad:
            raise serializers.ValidationError(f"Non-finite value for: {bad}")
        return value

    def validate_as_of_date(self, value):
        from django.utils import timezone
        if value > timezone.localdate():
            raise serializers.ValidationError("as_of_date cannot be in the future.")
        return value


class BatchRequestSerializer(serializers.Serializer):
    facility_codes = serializers.ListField(child=serializers.CharField(), required=False)
    medicine_codes = serializers.ListField(child=serializers.CharField(), required=False)
    as_of_date = serializers.DateField(required=False)


class PredictionRecordSerializer(serializers.ModelSerializer):
    facility_code = serializers.CharField(source="facility.code", read_only=True)
    medicine_code = serializers.CharField(source="medicine.code", read_only=True)
    model_version = serializers.CharField(source="model.label", read_only=True)
    prediction_timestamp = serializers.DateTimeField(source="created_at", read_only=True)
    data_freshness_days = serializers.SerializerMethodField()
    risk_category = serializers.SerializerMethodField()
    risk_thresholds = serializers.SerializerMethodField()

    class Meta:
        model = PredictionRecord
        fields = ["request_id", "facility", "facility_code", "medicine", "medicine_code", "as_of_date",
                  "prediction_timestamp", "horizon_days", "probability", "predicted_stockout", "threshold",
                  "probability_calibrated", "model_applied", "model_version", "data_freshness_days",
                  "warnings", "features", "batch", "risk_category", "risk_thresholds"]

    def get_data_freshness_days(self, obj):
        return (obj.created_at.date() - obj.as_of_date).days

    def get_risk_category(self, obj):
        return categorize(obj.probability, obj.model_applied)

    def get_risk_thresholds(self, obj):
        p = get_policy()
        return {**p["bands"], "provisional": p["provisional"]}

class BatchJobSerializer(serializers.ModelSerializer):
    model_version = serializers.CharField(source="model.label", read_only=True)

    class Meta:
        model = BatchPredictionJob
        fields = "__all__"