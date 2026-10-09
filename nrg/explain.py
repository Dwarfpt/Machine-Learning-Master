"""Explainable AI: SHAP values aggregated back to the original (human readable) features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from .features import feature_groups


class Explainer:
    def __init__(self, pipeline, background: pd.DataFrame):
        self.pre = pipeline[:-1]
        self.model = pipeline[-1]
        self.names = list(self.pre.get_feature_names_out())
        self.groups = feature_groups(self.names)
        bg = self.pre.transform(background)
        if isinstance(self.model, LogisticRegression):
            self._exp = shap.LinearExplainer(self.model, bg)
            self.output_space = "log-odds"
        else:
            self._exp = shap.TreeExplainer(self.model)
            # RandomForest explains probabilities, boosting models explain log-odds
            self.output_space = "probability" if hasattr(self.model, "estimators_") else "log-odds"

    def _raw_shap(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        Xt = self.pre.transform(X)
        sv = self._exp.shap_values(Xt)
        base = self._exp.expected_value
        if isinstance(sv, list):  # older shap API for classifiers
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, features, classes)
            sv = sv[:, :, 1]
        base = np.atleast_1d(base)
        return sv, float(base[-1])

    def grouped_shap(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """SHAP values summed per original feature: DataFrame (rows x original features)."""
        sv, base = self._raw_shap(X)
        df = pd.DataFrame(sv, columns=self.groups, index=X.index)
        return df.T.groupby(level=0, sort=False).sum().T, base

    def explain_row(self, X_row: pd.DataFrame, top_k: int = 8) -> dict:
        g, base = self.grouped_shap(X_row)
        contrib = g.iloc[0].sort_values(key=np.abs, ascending=False)
        factors = [{"feature": f, "value": _py(X_row.iloc[0][f]), "shap": float(v),
                    "direction": "increases_risk" if v > 0 else "decreases_risk"}
                   for f, v in contrib.items()]
        return {"base_value": base, "output_space": self.output_space,
                "factors": factors[:top_k], "all_factors": factors}

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        g, _ = self.grouped_shap(X)
        return (g.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
                .rename(columns={"index": "feature"}))


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v
