from django.core.cache import cache
from django.db.models import Q

BANDS_ORDER = ["CRITICAL", "HIGH", "MEDIUM"]
CATEGORIES = ["STOCKED_OUT", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
DEFAULT_BANDS = {"CRITICAL": 0.60, "HIGH": 0.30, "MEDIUM": 0.10}   # provisional, to be approved by stakeholders
CACHE_KEY = "msp:risk_policy"


def get_policy():
    p = cache.get(CACHE_KEY)
    if p is None:
        from .models import RiskPolicy
        obj = RiskPolicy.objects.filter(is_active=True).first()
        p = {"id": obj.pk if obj else None,
             "bands": obj.bands if obj else dict(DEFAULT_BANDS),
             "provisional": obj is None}
        cache.set(CACHE_KEY, p, 60)
    return p


def clear_policy_cache():
    cache.delete(CACHE_KEY)


def validate_bands(bands):
    if not isinstance(bands, dict) or set(bands) != set(BANDS_ORDER):
        raise ValueError("bands must contain exactly CRITICAL, HIGH and MEDIUM")
    vals = []
    for k in BANDS_ORDER:
        try:
            v = float(bands[k])
        except (TypeError, ValueError):
            raise ValueError(f"{k} must be a number")
        if not 0 < v < 1:
            raise ValueError(f"{k} must be between 0 and 1")
        vals.append(v)
    if not (vals[0] > vals[1] > vals[2]):
        raise ValueError("Require CRITICAL > HIGH > MEDIUM")
    return dict(zip(BANDS_ORDER, vals))


def categorize(probability, model_applied=True, policy=None):
    if not model_applied:
        return "STOCKED_OUT"
    b = (policy or get_policy())["bands"]
    for name in BANDS_ORDER:
        if probability >= b[name]:
            return name
    return "LOW"


def category_q(category, policy=None):
    """Q object over PredictionRecord matching a risk category."""
    b = (policy or get_policy())["bands"]
    applied = Q(model_applied=True)
    if category == "STOCKED_OUT":
        return Q(model_applied=False)
    if category == "CRITICAL":
        return applied & Q(probability__gte=b["CRITICAL"])
    if category == "HIGH":
        return applied & Q(probability__gte=b["HIGH"], probability__lt=b["CRITICAL"])
    if category == "MEDIUM":
        return applied & Q(probability__gte=b["MEDIUM"], probability__lt=b["HIGH"])
    if category == "LOW":
        return applied & Q(probability__lt=b["MEDIUM"])
    raise ValueError(f"Unknown category {category}")