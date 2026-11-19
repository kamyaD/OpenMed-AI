from django.db.models import Count
from rest_framework import serializers

from .models import (ConsumptionRecord, DataQualityIssue, Facility, IngestionBatch,
                     InventorySnapshot, Medicine, Order)


class FacilitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Facility
        fields = "__all__"


class MedicineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Medicine
        fields = "__all__"


class _Coded(serializers.ModelSerializer):
    facility_code = serializers.CharField(source="facility.code", read_only=True)
    medicine_code = serializers.CharField(source="medicine.code", read_only=True)


class InventorySnapshotSerializer(_Coded):
    class Meta:
        model = InventorySnapshot
        fields = "__all__"


class ConsumptionSerializer(_Coded):
    class Meta:
        model = ConsumptionRecord
        fields = "__all__"


class OrderSerializer(_Coded):
    lead_time_days = serializers.IntegerField(read_only=True)

    class Meta:
        model = Order
        fields = "__all__"


class DataQualityIssueSerializer(serializers.ModelSerializer):
    facility_code = serializers.CharField(source="facility.code", read_only=True, default=None)
    medicine_code = serializers.CharField(source="medicine.code", read_only=True, default=None)

    class Meta:
        model = DataQualityIssue
        fields = "__all__"


class IngestionRequestSerializer(serializers.Serializer):
    source = serializers.CharField(max_length=100)
    file = serializers.FileField(required=False)
    records = serializers.ListField(child=serializers.DictField(), required=False)

    def validate_file(self, f):
        if f.size > 25 * 1024 * 1024:
            raise serializers.ValidationError("File is larger than 25 MB.")
        return f


class IngestionBatchSerializer(serializers.ModelSerializer):
    issue_counts = serializers.SerializerMethodField()
    sample_issues = serializers.SerializerMethodField()

    class Meta:
        model = IngestionBatch
        fields = "__all__"

    def get_issue_counts(self, obj):
        return {r["issue_type"]: r["c"] for r in obj.issues.values("issue_type").annotate(c=Count("id"))}

    def get_sample_issues(self, obj):
        return DataQualityIssueSerializer(obj.issues.all()[:20], many=True).data