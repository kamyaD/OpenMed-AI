from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from inventory.ingestion import ingest
from inventory.models import ConsumptionRecord, DataQualityIssue, Facility, InventorySnapshot, Medicine


class IngestionTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        Facility.objects.create(code="F1", name="F", region="R")
        Medicine.objects.create(code="M1", name="M")
        cls.today = timezone.localdate()

    def d(self, n):
        return (self.today - timedelta(days=n)).isoformat()

    def row(self, **kw):
        base = {"facility_code": "F1", "medicine_code": "M1", "date": self.d(5), "stock_on_hand": "100"}
        return {**base, **kw}

    def test_dirty_rows_rejected_and_raw_payload_preserved(self):
        rows = [
            self.row(),
            self.row(date=self.d(4), stock_on_hand="90"),
            self.row(date=self.d(4), stock_on_hand="90"),            # duplicate
            self.row(date=self.d(3), stock_on_hand="-5"),            # negative
            self.row(facility_code="NOPE"),                          # unknown facility
            self.row(date="not-a-date"),                             # invalid date
            self.row(date=self.d(2), stock_on_hand=""),              # missing field
        ]
        batch = ingest("INVENTORY", rows, "test")
        self.assertEqual((batch.accepted_rows, batch.rejected_rows), (2, 5))
        self.assertEqual(batch.status, "COMPLETED_WITH_ISSUES")
        types = set(DataQualityIssue.objects.filter(batch=batch).values_list("issue_type", flat=True))
        self.assertEqual(types, {"DUPLICATE", "NEGATIVE_VALUE", "UNKNOWN_FACILITY", "INVALID_DATE", "MISSING_FIELD"})
        self.assertEqual(DataQualityIssue.objects.get(issue_type="NEGATIVE_VALUE").raw_payload["stock_on_hand"], "-5")
        self.assertEqual(InventorySnapshot.objects.count(), 2)

    def test_weekly_consumption_is_normalised_to_daily(self):
        batch = ingest("CONSUMPTION", [{"facility_code": "F1", "medicine_code": "M1", "date": self.d(1),
                                        "quantity_dispensed": "70", "period_days": "7"}], "test")
        self.assertEqual(batch.accepted_rows, 1)
        recs = ConsumptionRecord.objects.all()
        self.assertEqual(recs.count(), 7)
        self.assertTrue(all(abs(r.quantity - 10) < 1e-9 and r.original_quantity == 70 for r in recs))

    def test_inconsistent_lead_time_is_flagged_not_rejected(self):
        batch = ingest("ORDERS", [{"facility_code": "F1", "medicine_code": "M1", "order_number": "O1",
                                   "ordered_date": self.d(10), "quantity_ordered": "50",
                                   "delivered_date": self.d(12)}], "test")
        self.assertEqual(batch.accepted_rows, 1)
        issue = DataQualityIssue.objects.get(issue_type="INCONSISTENT_LEAD_TIME")
        self.assertEqual(issue.severity, "WARNING")