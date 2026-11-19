import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.utils import timezone
from rest_framework.exceptions import ValidationError
from rest_framework.test import APITestCase

from accounts.models import AuditLog, Role, UserProfile
from decision.coverage import coverage_from_features
from decision.dashboard import csv_safe
from decision.lifecycle import rollback
from decision.models import RedistributionSuggestion
from decision.monitoring import psi, realized_performance
from decision.redistribution import generate_suggestions
from decision.risk import categorize, get_policy
from inventory.models import Facility, InventorySnapshot, Medicine
from ml.models import BatchPredictionJob, ModelVersion, PredictionRecord


def feats(stock, daily=10.0, lead=10.0):
    return {"stock_on_hand": stock, "cons_avg_7": daily, "cons_avg_14": daily, "cons_avg_28": daily,
            "lead_time_mean": lead, "n_past_deliveries": 5.0, "obs_days_28": 28.0, "days_since_stock_obs": 0.0,
            "days_of_stock": (stock / daily) if daily else 365.0, "coverage_ratio": 1.0, "has_pending_order": 0.0}

class RiskAndCoverageTests(APITestCase):
    def setUp(self):
        cache.clear()

    def test_default_bands(self):
        self.assertTrue(get_policy()["provisional"])
        self.assertEqual(categorize(0.7), "CRITICAL")
        self.assertEqual(categorize(0.35), "HIGH")
        self.assertEqual(categorize(0.15), "MEDIUM")
        self.assertEqual(categorize(0.01), "LOW")
        self.assertEqual(categorize(1.0, model_applied=False), "STOCKED_OUT")

    def test_replenishment_gap(self):
        target = 10.0 * (10.0 + settings.MSP["SAFETY_STOCK_DAYS"])
        c = coverage_from_features(feats(100), 0.0)
        self.assertAlmostEqual(c["replenishment_gap"], target - 100)
        self.assertAlmostEqual(coverage_from_features(feats(100), 50.0)["replenishment_gap"], target - 150)
        self.assertEqual(c["days_of_stock"], 10.0)

    def test_zero_consumption_and_missing_data_are_safe(self):
        c = coverage_from_features(feats(100, daily=0.0))
        self.assertIsNone(c["days_of_stock"])
        self.assertEqual(c["replenishment_gap"], 0.0)
        self.assertTrue(c["flags"])
        self.assertIsNone(coverage_from_features({})["replenishment_gap"])


class MonitoringMathTests(APITestCase):
    def test_psi_detects_shift(self):
        rng = np.random.default_rng(1)
        a, same, shifted = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)
        self.assertLess(psi(a, same), 0.05)
        self.assertGreater(psi(a, shifted), 0.2)
        self.assertIsNone(psi(a[:10], same))

    def test_csv_injection_is_neutralised(self):
        self.assertEqual(csv_safe("=1+1"), "'=1+1")
        self.assertEqual(csv_safe("normal"), "normal")


class Base(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cache.clear()
        U = get_user_model()

        def mk(name, role, facilities=()):
            u = U.objects.create_user(name, password="x")
            UserProfile.objects.create(user=u, role=role).facilities.set(facilities)
            return u

        cls.today = timezone.localdate()
        cls.donor = Facility.objects.create(code="D1", name="Donor", region="R", level=4, latitude=-1.29, longitude=36.82)
        cls.recip = Facility.objects.create(code="R1", name="Recipient", region="R", level=3, latitude=-1.0, longitude=36.9)
        cls.med = Medicine.objects.create(code="M1", name="Med", category="Cat")
        cls.model = ModelVersion.objects.create(name="lr", algorithm="LOGISTIC_REGRESSION", status="ACTIVE")
        cls.batch = BatchPredictionJob.objects.create(model=cls.model, as_of_date=cls.today)
        common = dict(model=cls.model, medicine=cls.med, as_of_date=cls.today, horizon_days=7, threshold=0.05,
                      batch=cls.batch)
        PredictionRecord.objects.create(facility=cls.recip, probability=0.9, predicted_stockout=True,
                                        features=feats(10), **common)
        PredictionRecord.objects.create(facility=cls.donor, probability=0.01, predicted_stockout=False,
                                        features=feats(1000), **common)
        InventorySnapshot.objects.create(facility=cls.donor, medicine=cls.med, date=cls.today, stock_on_hand=1000,
                                         earliest_expiry_date=cls.today + timedelta(days=200), source="t")
        InventorySnapshot.objects.create(facility=cls.recip, medicine=cls.med, date=cls.today, stock_on_hand=10, source="t")
        cls.admin = mk("admin", Role.ADMIN)
        cls.manager = mk("manager", Role.SUPPLY_MANAGER)
        cls.analyst = mk("analyst", Role.ANALYST)
        cls.fac_donor = mk("facd", Role.FACILITY_USER, [cls.donor])
        cls.fac_recip = mk("facr", Role.FACILITY_USER, [cls.recip])

    def as_user(self, user):
        self.client.force_authenticate(user=user)
        return self.client


class RedistributionTests(Base):
    def test_donor_keeps_minimum_stock(self):
        res = generate_suggestions(self.batch)
        self.assertEqual(res["suggestions_created"], 1)
        s = RedistributionSuggestion.objects.get()
        target = 10.0 * (10.0 + settings.MSP["SAFETY_STOCK_DAYS"])
        self.assertEqual((s.from_facility, s.to_facility, s.status), (self.donor, self.recip, "PENDING_REVIEW"))
        self.assertEqual(s.quantity, target - 10)
        self.assertGreaterEqual(1000 - s.quantity, target)
        self.assertTrue(s.details["human_review_required"])

    def test_short_shelf_life_donor_is_excluded(self):
        InventorySnapshot.objects.filter(facility=self.donor).update(earliest_expiry_date=self.today + timedelta(days=20))
        res = generate_suggestions(self.batch)
        self.assertEqual(res["suggestions_created"], 0)
        self.assertIn("insufficient remaining shelf life", res["donor_exclusions"])

    def test_unknown_expiry_donor_is_excluded(self):
        InventorySnapshot.objects.filter(facility=self.donor).update(earliest_expiry_date=None)
        self.assertEqual(generate_suggestions(self.batch)["suggestions_created"], 0)


class ApprovalWorkflowTests(Base):
    def setUp(self):
        generate_suggestions(self.batch)
        self.s = RedistributionSuggestion.objects.get()
        self.url = f"/api/v1/decision/redistribution/{self.s.pk}"

    def test_manager_approves_with_audit_trail(self):
        r = self.as_user(self.manager).post(f"{self.url}/approve/", {"rationale": "Stock verified by phone"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.s.refresh_from_db()
        self.assertEqual((self.s.status, self.s.decided_by), ("APPROVED", self.manager))
        log = AuditLog.objects.get(action="redistribution.approved", object_id=str(self.s.pk))
        self.assertEqual((log.actor, log.rationale), (self.manager, "Stock verified by phone"))

    def test_rationale_is_required(self):
        r = self.as_user(self.manager).post(f"{self.url}/approve/", {}, format="json")
        self.assertEqual(r.status_code, 400)

    def test_unauthorised_roles_cannot_approve(self):
        for u in (self.analyst, self.fac_recip, self.fac_donor):
            r = self.as_user(u).post(f"{self.url}/approve/", {"rationale": "x"}, format="json")
            self.assertEqual(r.status_code, 403, u.username)
        self.s.refresh_from_db()
        self.assertEqual(self.s.status, "PENDING_REVIEW")

    def test_cannot_approve_twice_or_complete_unapproved(self):
        c = self.as_user(self.manager)
        self.assertEqual(c.post(f"{self.url}/complete/", {}, format="json").status_code, 409)
        c.post(f"{self.url}/approve/", {"rationale": "ok"}, format="json")
        self.assertEqual(c.post(f"{self.url}/approve/", {"rationale": "again"}, format="json").status_code, 409)
        self.assertEqual(c.post(f"{self.url}/complete/", {"quantity_moved": 150}, format="json").status_code, 200)

    def test_stale_suggestion_cannot_be_approved(self):
        RedistributionSuggestion.objects.filter(pk=self.s.pk).update(generated_at=timezone.now() - timedelta(days=10))
        r = self.as_user(self.manager).post(f"{self.url}/approve/", {"rationale": "ok"}, format="json")
        self.assertEqual(r.status_code, 400)


class DashboardTests(Base):
    def test_overview_summary_and_scoping(self):
        r = self.as_user(self.manager).get("/api/v1/dashboard/overview/").json()
        self.assertEqual(len(r["results"]), 2)
        self.assertEqual((r["summary"]["counts"]["total"], r["summary"]["counts"]["critical"],
                          r["summary"]["counts"]["low"]), (2, 1, 1))
        own = self.as_user(self.fac_donor).get("/api/v1/dashboard/overview/").json()
        self.assertEqual([x["facility_code"] for x in own["results"]], ["D1"])

    def test_risk_filter(self):
        r = self.as_user(self.manager).get("/api/v1/dashboard/overview/?risk=CRITICAL").json()
        self.assertEqual([x["facility_code"] for x in r["results"]], ["R1"])
        self.assertEqual(self.as_user(self.manager).get("/api/v1/dashboard/overview/?risk=BOGUS").status_code, 400)

    def test_export_respects_permissions(self):
        body = self.as_user(self.fac_donor).get("/api/v1/dashboard/export/").content.decode()
        self.assertIn("D1", body)
        self.assertNotIn("R1", body)
        self.assertTrue(AuditLog.objects.filter(action="dashboard.export").exists())

    def test_detail_denied_for_other_facility(self):
        r = self.as_user(self.fac_donor).get("/api/v1/dashboard/detail/?facility_code=R1&medicine_code=M1")
        self.assertEqual(r.status_code, 403)

    def test_explanation_lists_deterministic_factors(self):
        rec = PredictionRecord.objects.get(facility=self.recip)
        r = self.as_user(self.manager).get(f"/api/v1/dashboard/predictions/{rec.request_id}/explanation/").json()
        self.assertEqual(r["risk_category"], "CRITICAL")
        self.assertTrue(any(f["factor"] == "Low stock cover" for f in r["factors"]))
        self.assertIn("not proven causes", r["disclaimer"])

    def test_only_admin_changes_risk_policy(self):
        body = {"bands": {"CRITICAL": 0.5, "HIGH": 0.25, "MEDIUM": 0.1}, "rationale": "Agreed with sponsor"}
        self.assertEqual(self.as_user(self.manager).post("/api/v1/decision/risk-policy/", body, format="json").status_code, 403)
        bad = {"bands": {"CRITICAL": 0.2, "HIGH": 0.25, "MEDIUM": 0.1}, "rationale": "x"}
        self.assertEqual(self.as_user(self.admin).post("/api/v1/decision/risk-policy/", bad, format="json").status_code, 400)
        self.assertEqual(self.as_user(self.admin).post("/api/v1/decision/risk-policy/", body, format="json").status_code, 201)
        self.assertEqual(get_policy()["bands"]["CRITICAL"], 0.5)
        self.assertFalse(get_policy()["provisional"])


class LifecycleAndMonitoringTests(Base):
    def test_rollback_restores_previous_model(self):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as fh:
            path = fh.name
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        old = ModelVersion.objects.create(name="old", algorithm="RANDOM_FOREST", status="ARCHIVED", artifact_path=path,
                                          promoted_at=timezone.now() - timedelta(days=5))
        with self.assertRaises(ValidationError):
            rollback(self.admin, "   ")
        rollback(self.admin, "Recall dropped on realised outcomes")
        old.refresh_from_db()
        self.model.refresh_from_db()
        self.assertEqual((old.status, self.model.status), ("ACTIVE", "ARCHIVED"))
        self.assertTrue(AuditLog.objects.filter(action="model.rolled_back").exists())

    def test_rollback_endpoint_is_admin_only(self):
        r = self.as_user(self.manager).post("/api/v1/lifecycle/rollback/", {"rationale": "x"}, format="json")
        self.assertEqual(r.status_code, 403)

    def test_realised_performance_uses_future_stock(self):
        as_of = self.today - timedelta(days=20)
        PredictionRecord.objects.create(model=self.model, facility=self.recip, medicine=self.med, as_of_date=as_of,
                                        horizon_days=7, probability=0.9, predicted_stockout=True, threshold=0.5,
                                        features=feats(10))
        InventorySnapshot.objects.bulk_create([
            InventorySnapshot(facility=self.recip, medicine=self.med, date=as_of + timedelta(days=k),
                              stock_on_hand=0 if k == 3 else 5, source="t") for k in range(1, 8)])
        self.model.threshold = 0.5
        res = realized_performance(self.model, self.today)
        self.assertEqual((res["status"], res["n"], res["positives"], res["recall"]), ("ok", 1, 1, 1.0))