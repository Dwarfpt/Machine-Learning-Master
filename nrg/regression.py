"""Regression track: predict the critic SCORE itself (0-100) instead of the yes/no flop label.

Same features, same time split (train < TEST_FROM_YEAR, test >= TEST_FROM_YEAR), 5-fold CV.
Regression metrics: MAE, RMSE, R^2. To compare with the classifiers, the predicted score is also turned into
a flop flag (predicted score < 60) and ranked by "60 - predicted score" -> PR-AUC / recall / precision.

Run:  python -m nrg.regression    -> artifacts/regression_comparison.json (+ MLflow runs tagged task=regression)
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from . import config
from .features import make_preprocessor
from .train import _mlflow, load_dataset

warnings.filterwarnings("ignore")


def regressors() -> dict:
    rs = config.RANDOM_STATE
    return {
        "Baseline (mean score)": (DummyRegressor(), {}),
        "Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"reg__alpha": [1, 10, 30]}),
        "Lasso": (Lasso(max_iter=5000), {"reg__alpha": [0.01, 0.1, 0.3]}),
        "K-Nearest Neighbors": (KNeighborsRegressor(), {"reg__n_neighbors": [25, 50, 100]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=rs), {"reg__max_depth": [5, 8],
                                                                   "reg__min_samples_leaf": [20, 50]}),
        "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=1),
                          {"reg__max_depth": [10, 18], "reg__min_samples_leaf": [3, 10]}),
        "LightGBM": (LGBMRegressor(n_estimators=500, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, random_state=rs, n_jobs=1, verbose=-1),
                     {"reg__num_leaves": [15, 31], "reg__min_child_samples": [20, 50]}),
        "XGBoost": (XGBRegressor(n_estimators=500, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                 random_state=rs, n_jobs=1), {"reg__max_depth": [3, 5]}),
    }


def _metrics(score_true, score_pred) -> dict:
    flop_true = (score_true < config.NEGATIVE_THRESHOLD).astype(int)
    flop_pred = (score_pred < config.NEGATIVE_THRESHOLD).astype(int)
    return {"mae": mean_absolute_error(score_true, score_pred),
            "rmse": float(np.sqrt(mean_squared_error(score_true, score_pred))),
            "r2": r2_score(score_true, score_pred),
            "pr_auc": average_precision_score(flop_true, config.NEGATIVE_THRESHOLD - score_pred),
            "f1": f1_score(flop_true, flop_pred, zero_division=0),
            "recall": recall_score(flop_true, flop_pred, zero_division=0),
            "precision": precision_score(flop_true, flop_pred, zero_division=0)}


def main() -> pd.DataFrame:
    games, X, _, train_mask = load_dataset()
    s = games["metrics_review_score"].astype(float).values
    X_tr, s_tr, X_te, s_te = X[train_mask], s[train_mask], X[~train_mask], s[~train_mask]
    cv = KFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    mlflow = _mlflow()
    rows, fitted = [], {}

    def record(name, pipe, params):
        oof = cross_val_predict(pipe, X_tr, s_tr, cv=cv, n_jobs=-1)
        pipe = clone(pipe).fit(X_tr, s_tr)
        cv_m, te_m = _metrics(s_tr, oof), _metrics(s_te, pipe.predict(X_te))
        rows.append({"model": name, "best_params": params, **{f"cv_{k}": v for k, v in cv_m.items()},
                     **{f"test_{k}": v for k, v in te_m.items()}})
        fitted[name] = pipe
        print(f"{name:26s} CV MAE {cv_m['mae']:.2f} RMSE {cv_m['rmse']:.2f} R2 {cv_m['r2']:.3f} | test MAE "
              f"{te_m['mae']:.2f} R2 {te_m['r2']:.3f} flop PR-AUC {te_m['pr_auc']:.3f} recall {te_m['recall']:.2f}")
        if mlflow:
            with mlflow.start_run(run_name=f"REG {name}"):
                mlflow.set_tags({"task": "regression", "data": "kaggle+corgis"})
                mlflow.log_params({k.replace("reg__", ""): v for k, v in (params or {}).items()})
                mlflow.log_metrics({k: float(v) for k, v in rows[-1].items() if k.startswith(("cv_", "test_"))})

    for name, (reg, grid) in regressors().items():
        pipe = Pipeline([("pre", make_preprocessor()), ("reg", reg)])
        if grid:
            search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1).fit(X_tr, s_tr)
            record(name, search.best_estimator_, search.best_params_)
        else:
            record(name, pipe, {})

    top = pd.DataFrame(rows).iloc[1:].sort_values("cv_mae").head(3)["model"].tolist()
    vote = VotingRegressor([(n.replace(" ", "_"), fitted[n][-1]) for n in top])
    record("Average of top-3", Pipeline([("pre", make_preprocessor()), ("reg", vote)]), {"members": ", ".join(top)})

    table = pd.DataFrame(rows)
    table.to_json(config.ARTIFACTS_DIR / "regression_comparison.json", orient="records", indent=2,
                  default_handler=str)
    best = table.iloc[1:].sort_values("cv_mae").iloc[0]
    summary = {"best_model": best["model"], "test_mae": best["test_mae"], "test_r2": best["test_r2"],
               "test_flop_pr_auc": best["test_pr_auc"]}
    (config.ARTIFACTS_DIR / "regression_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nBest regressor by CV MAE: {best['model']} -> test MAE {best['test_mae']:.2f}, R2 {best['test_r2']:.3f}")
    return table


if __name__ == "__main__":
    main()
