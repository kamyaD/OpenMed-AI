from datetime import timedelta

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from inventory.models import (ConsumptionRecord, DataQualityIssue, Facility, IngestionBatch,
                              InventorySnapshot, Medicine, Order)

REGIONS = {
    "Nairobi": (["Nairobi"], (-1.29, 36.82)),
    "Coast": (["Mombasa", "Kilifi", "Kwale"], (-3.8, 39.6)),
    "Rift Valley": (["Nakuru", "Turkana", "Kajiado", "Uasin Gishu"], (0.3, 36.0)),
    "Western": (["Kakamega", "Bungoma"], (0.5, 34.7)),
    "Central": (["Kiambu", "Nyeri"], (-0.5, 36.9)),
    "Eastern": (["Machakos", "Marsabit"], (-0.9, 37.6)),
    "Nyanza": (["Kisumu", "Migori"], (-0.6, 34.6)),
    "North Eastern": (["Garissa", "Wajir"], (0.5, 39.6)),
}
URBAN = {"Nairobi", "Mombasa", "Nakuru", "Kisumu", "Kiambu", "Uasin Gishu"}
FTYPE = {2: "DISPENSARY", 3: "HEALTH_CENTRE", 4: "SUB_COUNTY_HOSPITAL", 5: "COUNTY_HOSPITAL", 6: "REFERRAL_HOSPITAL"}
LEAD = {2: 20, 3: 16, 4: 11, 5: 8, 6: 6}
MEDICINES = [
    ("AMX500", "Amoxicillin 500mg capsules", "Antibiotics", "capsule"),
    ("ACT020", "Artemether-Lumefantrine 20/120mg", "Antimalarials", "tablet"),
    ("OXY010", "Oxytocin 10IU injection", "Maternal health", "ampoule"),
    ("PCM500", "Paracetamol 500mg tablets", "Analgesics", "tablet"),
    ("ORS001", "Oral rehydration salts", "Paediatric", "sachet"),
    ("ZNC020", "Zinc sulphate 20mg", "Paediatric", "tablet"),
    ("MET500", "Metformin 500mg", "Diabetes", "tablet"),
    ("AML005", "Amlodipine 5mg", "Cardiovascular", "tablet"),
    ("COT480", "Cotrimoxazole 480mg", "Antibiotics", "tablet"),
    ("FEF200", "Ferrous/Folic acid", "Maternal health", "tablet"),
    ("CEF1GM", "Ceftriaxone 1g injection", "Antibiotics", "vial"),
    ("INS100", "Soluble insulin 100IU/ml", "Diabetes", "vial"),
    ("SAL100", "Salbutamol inhaler 100mcg", "Respiratory", "inhaler"),
    ("TLD300", "Tenofovir/Lamivudine/Dolutegravir", "HIV", "tablet"),
    ("RDT001", "Malaria rapid test kit", "Diagnostics", "kit"),
    ("MSP400", "Misoprostol 200mcg", "Maternal health", "tablet"),
    ("MGS500", "Magnesium sulphate 50% injection", "Maternal health", "ampoule"),
    ("GEN080", "Gentamicin 80mg injection", "Antibiotics", "ampoule"),
    ("DIC050", "Diclofenac 50mg", "Analgesics", "tablet"),
    ("ALB400", "Albendazole 400mg", "Antiparasitics", "tablet"),
]


class Command(BaseCommand):
    help = "Generate synthetic facilities, medicines, stock, consumption and orders (development only)"

    def add_arguments(self, p):
        p.add_argument("--facilities", type=int, default=40)
        p.add_argument("--medicines", type=int, default=15)
        p.add_argument("--days", type=int, default=240)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--reset", action="store_true")

    def handle(self, *args, **o):
        if o["reset"]:
            for M in (DataQualityIssue, IngestionBatch, Order, ConsumptionRecord, InventorySnapshot, Medicine, Facility):
                M.objects.all().delete()
        elif Facility.objects.exists():
            raise CommandError("Data already exists. Use --reset to wipe and regenerate.")

        rng = np.random.default_rng(o["seed"])
        days = o["days"]
        today = timezone.localdate()
        start = today - timedelta(days=days)

        regions = list(REGIONS)
        for i in range(o["facilities"]):
            region = regions[i % len(regions)]
            counties, (lat, lon) = REGIONS[region]
            county = str(rng.choice(counties))
            level = int(rng.choice([2, 3, 4, 5, 6], p=[0.45, 0.30, 0.15, 0.08, 0.02]))
            Facility.objects.create(
                code=f"F{i + 1:04d}", region=region, county=county, level=level, facility_type=FTYPE[level],
                name=f"{county} {Facility.Type(FTYPE[level]).label} {i + 1}",
                is_urban=bool(county in URBAN and rng.random() < 0.8),
                latitude=lat + float(rng.normal(0, 0.3)), longitude=lon + float(rng.normal(0, 0.3)))
        for j in range(o["medicines"]):
            code, name, cat, unit = MEDICINES[j] if j < len(MEDICINES) else (f"MED{j:03d}", f"Medicine {j}", "Other", "unit")
            Medicine.objects.create(code=code, name=name, category=cat, unit=unit)

        meds = list(Medicine.objects.all())
        for n, f in enumerate(Facility.objects.all(), start=1):
            snaps, cons, orders = [], [], []
            for m in meds:
                demand = float(rng.lognormal(2.2 + 0.35 * (f.level - 2), 0.6))
                lead_mean = LEAD[f.level] * (0.8 if f.is_urban else 1.3)
                factor = float(rng.uniform(0.6, 1.5))            # <1 means under-ordering -> stockouts
                stock = int(demand * rng.uniform(8, 35))
                pending, n_orders = None, 0
                expiry = start + timedelta(days=int(rng.integers(40, 400)))
                for d in range(days):
                    day = start + timedelta(days=d + 1)
                    if pending and pending[0] <= d:
                        stock += pending[1]
                        pending = None
                    season = 1 + 0.25 * np.sin(2 * np.pi * d / 90)
                    dem = int(rng.poisson(max(demand * season * rng.uniform(0.8, 1.25), 0.1)))
                    if rng.random() < 0.02:
                        dem = int(dem * 2.5)                      # demand spike
                    sold = int(min(stock, dem))
                    stock -= sold
                    if pending is None and stock <= demand * lead_mean * factor:
                        lead = max(2, int(rng.normal(lead_mean, lead_mean * 0.25)))
                        if rng.random() < 0.15:
                            lead += int(rng.integers(5, 15))      # late delivery
                        qty = int(demand * 30)
                        n_orders += 1
                        arrival = d + lead
                        delivered = start + timedelta(days=arrival + 1) if arrival < days else None
                        orders.append(Order(
                            facility=f, medicine=m, order_number=f"{f.code}-{m.code}-{n_orders:03d}",
                            ordered_date=day, quantity_ordered=qty,
                            expected_delivery_date=day + timedelta(days=int(lead_mean)),
                            delivered_date=delivered, quantity_delivered=qty if delivered else None, source="seed"))
                        pending = (arrival, qty)
                    if day > expiry:
                        expiry = day + timedelta(days=int(rng.integers(120, 400)))
                    if rng.random() < 0.97:
                        snaps.append(InventorySnapshot(facility=f, medicine=m, date=day, stock_on_hand=stock,
                                                       earliest_expiry_date=expiry, source="seed"))
                    if rng.random() < 0.97:
                        cons.append(ConsumptionRecord(facility=f, medicine=m, date=day, quantity=sold,
                                                      original_quantity=sold, original_period_days=1, source="seed"))
            InventorySnapshot.objects.bulk_create(snaps, batch_size=5000)
            ConsumptionRecord.objects.bulk_create(cons, batch_size=5000)
            Order.objects.bulk_create(orders, batch_size=5000)
            self.stdout.write(f"facility {n}/{o['facilities']} done")
        self.stdout.write(self.style.SUCCESS("Seed complete."))