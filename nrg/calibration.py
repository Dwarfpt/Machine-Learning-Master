"""Platt calibration of model scores (kept in its own module so the pickled artifact can be loaded anywhere)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Logistic regression on the logit of the raw score."""

    def fit(self, p, y):
        self.lr = LogisticRegression().fit(self._logit(p), y)
        return self

    def transform(self, p):
        return self.lr.predict_proba(self._logit(p))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)
