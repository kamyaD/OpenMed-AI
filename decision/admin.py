from django.contrib import admin

from .models import MonitoringReport, RedistributionSuggestion, RiskPolicy

admin.site.register([RiskPolicy, RedistributionSuggestion, MonitoringReport])