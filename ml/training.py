from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from django.conf import settings
from django.utils import timezone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .bundle import ModelBundle, ProbabilityCalibrator
from .evaluation import classification_metrics, expected_calibration_error, select_threshold, subgroup_report
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .model_card import build_model_card
from .models import ModelVersion
from .splits import split_masks
from .utils import code_version

ECE_TARGET = 0.05          # probabilities are called "calibrated" only if test ECE <= this
MIN_CAL_POSITIVES = 30
COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def rule_baseline(df, horizon):
    """Flag if stock covers < horizon days, or if it won't cover the lead time with nothing on order."""
    dos = df["days_of_stock"].fillna(365).to_numpy()
    cov = df["coverage_ratio"].fillna(50).to_numpy()
    pending = df["has_pending_order"].to_numpy() > 0
    flag = (dos < horizon) | ((cov < 1) & ~pending & (dos < 2 * horizon))
    return 1.0 / (1.0 + dos / horizon), flag       # monotone score used for PR-AUC only


def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("sc", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", pre), ("clf", estimator)])


def choose_calibration(p_a, y_a, p_b, y_b):
    info = {"method": "none", "fit_rows": int(len(y_a)), "fit_positives": int(y_a.sum()),
            "ece_raw": round(expected_calibration_error(y_b, p_b), 6)}
    if y_a.sum() < MIN_CAL_POSITIVES or len(set(y_a.tolist())) < 2:
        info["note"] = "Too few positive examples to fit a calibrator; raw probabilities kept."
        return None, info
    best_ece, best_cal = info["ece_raw"], None
    for method in ("sigmoid", "isotonic"):
        cal = ProbabilityCalibrator(method).fit(p_a, y_a)
        e = expected_calibration_error(y_b, cal.predict(p_b))
        info[f"ece_{method}"] = round(e, 6)
        if e < best_ece * 0.9:               # require a 10% relative improvement to justify calibrating
            best_ece, best_cal, info["method"] = e, cal, method
    return best_cal, info


def feature_importance(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    vals = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
    top = sorted(zip(names, vals), key=lambda t: abs(t[1]), reverse=True)[:15]
    return [{"feature": n.replace("num__", "").replace("cat__", ""), "weight": round(float(v), 5)} for n, v in top]


def train_all(dataset, user=None, facility_holdout=0.0, seed=42, max_train_rows=400_000, log=print):
    cfg = settings.MSP
    H, c_fn, c_fp = cfg["HORIZON_DAYS"], cfg["COST_FALSE_NEGATIVE"], cfg["COST_FALSE_POSITIVE"]
    df = pd.read_parquet(dataset.file_path)
    df["date"] = pd.to_datetime(df["date"])
    m = split_masks(df["date"], dataset.split)
    train, val_a, val_b, test = df[m["train"]], df[m["val_a"]], df[m["val_b"]], df[m["test"]]

    holdout = set()
    if facility_holdout > 0:
        ids = np.sort(df["facility_id"].unique())
        rng = np.random.default_rng(seed)
        holdout = set(rng.choice(ids, size=max(1, int(len(ids) * facility_holdout)), replace=False).tolist())
        train = train[~train["facility_id"].isin(holdout)]
    if len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=seed)
    if train["y"].nunique() < 2 or val_b["y"].nunique() < 2 or test["y"].nunique() < 2:
        raise ValueError("Train, validation-B and test sets each need both classes. The seeded data may have too few "
                         "stockouts; re-seed with more days (--days 365).")
    log(f"rows: train={len(train)} val_a={len(val_a)} val_b={len(val_b)} test={len(test)} "
        f"(positive rate train={train['y'].mean():.3f})")

    ya, yb, yt = (x["y"].to_numpy().astype(int) for x in (val_a, val_b, test))
    common = dict(code_version=code_version(), dataset=dataset, trained_at=timezone.now(),
                  library_versions={"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                                    "numpy": np.__version__})

    # ---- rule-based baseline
    def base_eval(part):
        s, flag = rule_baseline(part, H)
        return classification_metrics(part["y"], s, 0.5, pred=flag, probabilistic=False)
    baseline_metrics = {"val_b": base_eval(val_b), "test": base_eval(test)}
    base_mv = ModelVersion.objects.create(
        name="rule_baseline", algorithm="RULE_BASELINE", metrics=baseline_metrics,
        feature_schema={"numeric": ["days_of_stock", "coverage_ratio", "has_pending_order"], "categorical": []},
        model_card={"description": "Deterministic coverage / lead-time rule used as the comparison baseline; "
                                   "it has no artifact and cannot be promoted."}, **common)
    log(f"baseline test: recall={baseline_metrics['test'].get('recall')} precision={baseline_metrics['test'].get('precision')}")

    candidates = {
        "logistic_regression": ("LOGISTIC_REGRESSION", LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": ("RANDOM_FOREST", RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
    }
    created = []
    out_dir = Path(settings.MODEL_ARTIFACT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = timezone.now().strftime("%Y%m%d%H%M%S")

    for name, (algo, est) in candidates.items():
        log(f"training {name} ...")
        pipe = build_pipeline(est).fit(train[COLS], train["y"].astype(int))
        pa, pb, pt = (pipe.predict_proba(x[COLS])[:, 1] for x in (val_a, val_b, test))
        cal, cal_info = choose_calibration(pa, ya, pb, yb)
        if cal is not None:
            pb, pt = cal.predict(pb), cal.predict(pt)
        thr, curve = select_threshold(yb, pb, c_fn, c_fp)
        metrics = {"val_b": classification_metrics(yb, pb, thr), "test": classification_metrics(yt, pt, thr),
                   "train_rows": int(len(train)), "feature_importance": feature_importance(pipe),
                   "threshold_cost_curve": curve, "baseline": baseline_metrics}
        if holdout:
            seen = test["facility_id"].isin(holdout).to_numpy()
            metrics["test_unseen_facilities"] = classification_metrics(yt[seen], pt[seen], thr)
            metrics["test_seen_facilities"] = classification_metrics(yt[~seen], pt[~seen], thr)
        test_ece = metrics["test"].get("ece")
        cal_info["ece_test_final"] = test_ece
        cal_info["probability_calibrated"] = bool(test_ece is not None and test_ece <= ECE_TARGET)
        cal_info["ece_target"] = ECE_TARGET

        sub = subgroup_report(test, pt, thr)
        bundle = ModelBundle(pipe, cal, NUMERIC_FEATURES, CATEGORICAL_FEATURES, thr, algo)
        path = out_dir / f"{name}_{stamp}.joblib"
        joblib.dump(bundle, path)
        chk = joblib.load(path)                                                       # artifact must reload identically
        sample = test[COLS].head(200)
        assert np.allclose(chk.predict_proba(sample), bundle.predict_proba(sample)), "artifact reload mismatch"

        mv = ModelVersion.objects.create(
            name=name, algorithm=algo, artifact_path=str(path), threshold=thr, calibration=cal_info,
            metrics=metrics, subgroup_report=sub, requires_review=sub["requires_review"],
            feature_schema={"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
                            "horizon_days": H, "hyperparameters": est.get_params()},
            model_card=build_model_card(name, algo, dataset, metrics, cal_info, sub, thr), **common)
        created.append(mv)
        log(f"  {name}: test recall={metrics['test'].get('recall')} precision={metrics['test'].get('precision')} "
            f"pr_auc={metrics['test'].get('pr_auc')} calibrated={cal_info['probability_calibrated']} "
            f"threshold={thr} subgroup flags={len(sub['flags'])}")

    # ---- recommendation (a human still has to promote)
    best = min(created, key=lambda v: v.metrics["val_b"]["cost_per_1000"])
    base_val, base_test = baseline_metrics["val_b"]["cost_per_1000"], baseline_metrics["test"]["cost_per_1000"]
    for mv in created:
        wins = mv.metrics["val_b"]["cost_per_1000"] < base_val and mv.metrics["test"]["cost_per_1000"] < base_test
        mv.metrics["selection"] = {
            "recommended": bool(mv.pk == best.pk and wins),
            "reason": ("Lowest validation cost and beats the rule baseline on validation and test."
                       if mv.pk == best.pk and wins else "Not the lowest-cost candidate, or does not beat the baseline."),
            "note": "Threshold is tuned on validation-B, so validation cost is slightly optimistic; test confirms it.",
        }
        mv.save(update_fields=["metrics"])
    return [base_mv] + created


def evaluate_model_subgroups(mv):
    bundle = joblib.load(mv.artifact_path)
    df = pd.read_parquet(mv.dataset.file_path)
    df["date"] = pd.to_datetime(df["date"])
    test = df[split_masks(df["date"], mv.dataset.split)["test"]]
    proba = bundle.predict_proba(test[bundle.numeric + bundle.categorical])
    rep = subgroup_report(test, proba, bundle.threshold)
    mv.subgroup_report, mv.requires_review = rep, rep["requires_review"]
    mv.save(update_fields=["subgroup_report", "requires_review"])
    return rep