import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method):
        self.method, self.model = method, None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        if self.method == "sigmoid":
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(p, float), y)
        return self

    def predict(self, p):
        if self.method == "sigmoid":
            return self.model.predict_proba(self._logit(p))[:, 1]
        return self.model.predict(np.asarray(p, dtype=float))


class ModelBundle:
    def __init__(self, pipeline, calibrator, numeric, categorical, threshold, algorithm):
        self.pipeline, self.calibrator = pipeline, calibrator
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.threshold, self.algorithm = float(threshold), algorithm

    def raw_proba(self, X):
        return self.pipeline.predict_proba(X[self.numeric + self.categorical])[:, 1]

    def predict_proba(self, X):
        p = self.raw_proba(X)
        return self.calibrator.predict(p) if self.calibrator is not None else p