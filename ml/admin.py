from django.contrib import admin

from .models import BatchPredictionJob, DatasetVersion, ModelVersion, PredictionRecord

admin.site.register([DatasetVersion, ModelVersion, BatchPredictionJob, PredictionRecord])