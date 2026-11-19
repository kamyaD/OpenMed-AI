from django.contrib import admin

from .models import (ConsumptionRecord, DataQualityIssue, Facility, IngestionBatch,
                     InventorySnapshot, Medicine, Order)

admin.site.register([Facility, Medicine, IngestionBatch, DataQualityIssue])
admin.site.register([InventorySnapshot, ConsumptionRecord, Order])