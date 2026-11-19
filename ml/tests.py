from datetime import date, timedelta

import numpy as np
import pandas as pd
from django.conf import settings
from django.test import TestCase

from inventory.models import ConsumptionRecord, Facility, InventorySnapshot, Medicine, Order
from ml.features import build_features
from ml.leakage import leakage_check
from ml.splits import compute_boundaries


class FeatureTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        rng = np.random.default_rng(7)
        cls.start = date(2025, 1, 1)
        f = Facility.objects.create(code="T1", name="Test", region="Coast", level=3)
        m = Medicine.objects.create(code="M1", name="Med", category="Test")
        stock, snaps, cons, orders = 300, [], [], []
        for i in range(260):
            d = cls.start + timedelta(days=i)
            sold = int(min(stock, rng.poisson(10)))
            stock -= sold
            if i % 30 == 29:
                stock += 250
                orders.append(Order(facility=f, medicine=m, order_number=f"O{i}", ordered_date=d - timedelta(days=15),
                                    quantity_ordered=250, delivered_date=d, quantity_delivered=250, source="t"))
            snaps.append(InventorySnapshot(facility=f, medicine=m, date=d, stock_on_hand=stock, source="t"))
            cons.append(ConsumptionRecord(facility=f, medicine=m, date=d, quantity=sold,
                                          original_quantity=sold, source="t"))
        InventorySnapshot.objects.bulk_create(snaps)
        ConsumptionRecord.objects.bulk_create(cons)
        Order.objects.bulk_create(orders)
        cls.stock = {s.date: s.stock_on_hand for s in snaps}

    def test_no_future_leakage(self):
        dates = [self.start + timedelta(days=k) for k in (150, 180, 210)]
        self.assertEqual(leakage_check(dates), [])

    def test_labels_match_future_stock(self):
        t0 = self.start + timedelta(days=150)
        df = build_features(t0, t0 + timedelta(days=40), with_labels=True)
        self.assertGreater(len(df), 0)
        H = settings.MSP["HORIZON_DAYS"]
        for r in df.itertuples():
            d = r.date.date()
            future = [self.stock[d + timedelta(days=k)] for k in range(1, H + 1)]
            self.assertEqual(r.y, 1.0 if min(future) <= 0 else 0.0)

    def test_lead_time_uses_only_past_deliveries(self):
        t = self.start + timedelta(days=100)          # deliveries completed by then: days 29, 59, 89 (lead time 15)
        df = build_features(t, t)
        self.assertEqual(int(df["n_past_deliveries"].iloc[0]), 3)
        self.assertAlmostEqual(float(df["lead_time_mean"].iloc[0]), 15.0, places=3)


class SplitTests(TestCase):
    def test_purge_gap_between_periods(self):
        b = compute_boundaries(pd.date_range("2025-01-01", periods=300), 7)
        T = pd.Timestamp
        self.assertGreater((T(b["val_start"]) - T(b["train_end"])).days, 7)
        self.assertGreater((T(b["val_b_start"]) - T(b["val_a_end"])).days, 7)
        self.assertGreater((T(b["test_start"]) - T(b["val_end"])).days, 7)
        self.assertLess(T(b["train_end"]), T(b["val_start"]))
        self.assertLess(T(b["val_end"]), T(b["test_start"]))