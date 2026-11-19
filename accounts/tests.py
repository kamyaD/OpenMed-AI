from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from accounts.models import Role, UserProfile
from inventory.models import Facility


class RbacTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.f1 = Facility.objects.create(code="A1", name="A", region="R")
        cls.f2 = Facility.objects.create(code="B1", name="B", region="R")
        U = get_user_model()

        def mk(name, role, facilities=()):
            u = U.objects.create_user(name, password="x")
            UserProfile.objects.create(user=u, role=role).facilities.set(facilities)
            return u

        cls.admin, cls.analyst = mk("admin", Role.ADMIN), mk("analyst", Role.ANALYST)
        cls.fac = mk("fac", Role.FACILITY_USER, [cls.f1])

    def get(self, user, url):
        self.client.force_authenticate(user=user)
        return self.client.get(url)

    def test_unauthenticated_is_rejected(self):
        self.client.force_authenticate(user=None)
        self.assertEqual(self.client.get("/api/v1/me/").status_code, 401)

    def test_facility_user_only_sees_own_facility(self):
        codes = [f["code"] for f in self.get(self.fac, "/api/v1/facilities/").json()["results"]]
        self.assertEqual(codes, ["A1"])

    def test_facility_user_cannot_use_analyst_endpoints(self):
        self.assertEqual(self.get(self.fac, "/api/v1/quality/report/").status_code, 403)
        self.client.force_authenticate(user=self.fac)
        r = self.client.post("/api/v1/ingestion/inventory/", {"source": "x", "records": []}, format="json")
        self.assertEqual(r.status_code, 403)

    def test_audit_log_is_admin_only(self):
        self.assertEqual(self.get(self.analyst, "/api/v1/audit-logs/").status_code, 403)
        self.assertEqual(self.get(self.admin, "/api/v1/audit-logs/").status_code, 200)

    def test_analyst_can_read_quality_report(self):
        self.assertEqual(self.get(self.analyst, "/api/v1/quality/report/").status_code, 200)