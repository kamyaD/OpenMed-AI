from django.conf import settings

DEFAULTS = {
    "REDISTRIBUTION_MAX_DISTANCE_KM": 400,
    "REDISTRIBUTION_MAX_DONORS": 3,
    "REDISTRIBUTION_REQUIRE_EXPIRY": True,   # donors with unknown expiry are excluded
    "REDISTRIBUTION_MIN_DATA_DAYS": 21,      # of the last 28 days
    "SUGGESTION_MAX_AGE_DAYS": 3,            # older suggestions cannot be approved
    "PENDING_ORDER_MAX_AGE_DAYS": 90,
    "MIN_DAILY_CONSUMPTION": 0.05,
    "STALE_FACILITY_SHARE_ALERT": 0.20,
    "PSI_WARN": 0.10,
    "ALERT_RATE_RATIO": 1.5,
    "RECALL_DROP_ALERT": 0.10,
    "MIN_LABELLED_POSITIVES": 30,
    "MODEL_MAX_AGE_DAYS": 90,
    "DRIFT_FEATURES_ALERT_COUNT": 3,
    "MAX_EXPORT_ROWS": 50000,
}


def cfg(key):
    return settings.MSP.get(key, DEFAULTS.get(key))