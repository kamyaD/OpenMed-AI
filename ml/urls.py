from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("ml/datasets", views.DatasetViewSet, basename="dataset")
router.register("ml/models", views.ModelVersionViewSet, basename="modelversion")
router.register("ml/predictions", views.PredictionViewSet, basename="prediction")
router.register("ml/batch-jobs", views.BatchJobViewSet, basename="batchjob")

urlpatterns = [
    path("", include(router.urls)),
    path("ml/predict/", views.PredictView.as_view()),
    path("ml/predict/batch/", views.BatchPredictView.as_view()),
]