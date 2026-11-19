from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("decision/redistribution", views.RedistributionViewSet, basename="redistribution")
router.register("monitoring/reports", views.MonitoringReportViewSet, basename="monitoringreport")

urlpatterns = [
    path("", include(router.urls)),
    path("decision/risk-policy/", views.RiskPolicyView.as_view()),
    path("dashboard/overview/", views.OverviewView.as_view()),
    path("dashboard/filters/", views.FiltersView.as_view()),
    path("dashboard/detail/", views.DetailView.as_view()),
    path("dashboard/export/", views.ExportView.as_view()),
    path("dashboard/predictions/<uuid:request_id>/explanation/", views.ExplanationView.as_view()),
    path("monitoring/run/", views.RunMonitoringView.as_view()),
    path("monitoring/rules/", views.RulesView.as_view()),
    path("lifecycle/status/", views.LifecycleStatusView.as_view()),
    path("lifecycle/rollback/", views.RollbackView.as_view()),
]