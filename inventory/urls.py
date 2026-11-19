from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("facilities", views.FacilityViewSet)
router.register("medicines", views.MedicineViewSet)
router.register("inventory/snapshots", views.InventorySnapshotViewSet, basename="snapshot")
router.register("inventory/consumption", views.ConsumptionViewSet, basename="consumption")
router.register("inventory/orders", views.OrderViewSet, basename="order")
router.register("ingestion/batches", views.IngestionBatchViewSet, basename="ingestionbatch")
router.register("quality/issues", views.DataQualityIssueViewSet, basename="dqissue")

urlpatterns = [
    path("", include(router.urls)),                       # router first so ingestion/batches wins
    path("ingestion/<str:dataset>/", views.IngestionView.as_view()),
    path("quality/report/", views.QualityReportView.as_view()),
]