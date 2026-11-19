from rest_framework import serializers

from .models import MonitoringReport, RedistributionSuggestion


class RedistributionSerializer(serializers.ModelSerializer):
    medicine_code = serializers.CharField(source="medicine.code", read_only=True)
    medicine_name = serializers.CharField(source="medicine.name", read_only=True)
    from_facility_code = serializers.CharField(source="from_facility.code", read_only=True)
    from_facility_name = serializers.CharField(source="from_facility.name", read_only=True)
    to_facility_code = serializers.CharField(source="to_facility.code", read_only=True)
    to_facility_name = serializers.CharField(source="to_facility.name", read_only=True)
    decided_by_name = serializers.CharField(source="decided_by.username", read_only=True, default=None)
    completed_by_name = serializers.CharField(source="completed_by.username", read_only=True, default=None)
    review_notice = serializers.SerializerMethodField()

    class Meta:
        model = RedistributionSuggestion
        fields = "__all__"

    def get_review_notice(self, obj):
        return "Suggestion only. No stock moves until an authorised user approves and the transfer is carried out."


class MonitoringReportSerializer(serializers.ModelSerializer):
    model_version = serializers.CharField(source="model.label", read_only=True, default=None)

    class Meta:
        model = MonitoringReport
        fields = "__all__"