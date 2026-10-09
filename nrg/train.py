"""Train & compare models, pick the best by cross-validated PR-AUC, evaluate on a time-based holdout (2008),
log everything to MLflow and save the production artifact.

Run:  python -m nrg.train
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, db, quality
from .calibration import PlattCalibrator
from .explain import Explainer
from .features import FEATURE_COLS, build_features, games_to_specs, history_from_games, make_preprocessor

warnings.filterwarnings("ignore", category=UserWarning)


def candidate_models(pos_weight: float) -> dict:
    rs = config.RANDOM_STATE
    return {
        "Baseline (majority)": (DummyClassifier(strategy="prior"), {}),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.03, 0.1, 0.3, 1.0]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=rs, n_jobs=-1),
            {"clf__max_depth": [4, 6, 10], "clf__min_samples_leaf": [3, 10]}),
        "LightGBM": (
            LGBMClassifier(n_estimators=300, learning_rate=0.03, scale_pos_weight=pos_weight, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=rs, verbose=-1),
            {"clf__num_leaves": [7, 15], "clf__min_child_samples": [10, 30], "clf__reg_lambda": [0, 5]}),
        "XGBoost": (
            XGBClassifier(n_estimators=300, learning_rate=0.03, scale_pos_weight=pos_weight, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="logloss", random_state=rs, n_jobs=-1),
            {"clf__max_depth": [2, 3, 5], "clf__min_child_weight": [1, 5]}),
    }


def best_threshold(y, proba, beta: float = 1.0) -> float:
    """Threshold maximising F-beta on out-of-fold predictions (F1: balance between missed flops and
    unnecessary interventions; the cost-based decision module then decides WHICH measure is worth it)."""
    grid = np.linspace(0.05, 0.9, 86)
    scores = [fbeta_score(y, proba >= t, beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {"pr_auc": average_precision_score(y, proba), "roc_auc": roc_auc_score(y, proba),
            "f1": f1_score(y, pred, zero_division=0), "f2": fbeta_score(y, pred, beta=2, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "brier": brier_score_loss(y, proba)}


def _mlflow():
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("negative-review-guard")
        return mlflow
    except Exception as e:  # MLflow is a bonus — training must work without it
        print(f"MLflow disabled: {e}")
        return None


def main() -> dict:
    if not config.DB_PATH.exists():
        db.init_db()
    quality.main()
    games = db.load_games()
    history = history_from_games(games)  # all scored games -> publisher track record
    specs = games_to_specs(games)  # on ALL games, so n_platforms counts every platform release
    known = (games["length_main_story_median"] > 0).values  # see leakage note in features.py
    games, specs = games[known].reset_index(drop=True), specs[known].reset_index(drop=True)
    X = build_features(specs, history)
    y = (games["metrics_review_score"] < config.NEGATIVE_THRESHOLD).astype(int).values

    train_mask = (games["release_year"] < config.TEST_YEAR).values
    X_tr, y_tr, X_te, y_te = X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]
    print(f"Labelled games with known length: {len(X)}")
    print(f"Train {len(X_tr)} (neg {y_tr.mean():.1%}) | Test {config.TEST_YEAR}: {len(X_te)} (neg {y_te.mean():.1%})")

    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    pos_weight = (1 - y_tr.mean()) / y_tr.mean()
    mlflow = _mlflow()
    results, fitted, curves = [], {}, {}

    for name, (clf, grid) in candidate_models(pos_weight).items():
        pipe = Pipeline([("pre", make_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid or {"clf__strategy": ["prior"]}, scoring="average_precision", cv=cv,
                              n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        oof = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = best_threshold(y_tr, oof) if name != "Baseline (majority)" else 0.5
        proba_te = best.predict_proba(X_te)[:, 1]
        cv_m = evaluate(y_tr, oof, thr)
        te_m = evaluate(y_te, proba_te, thr)
        row = {"model": name, "best_params": search.best_params_, "threshold": thr,
               **{f"cv_{k}": v for k, v in cv_m.items()}, **{f"test_{k}": v for k, v in te_m.items()}}
        results.append(row)
        fitted[name] = (best, oof)
        fpr, tpr, _ = roc_curve(y_te, proba_te)
        prec, rec, _ = precision_recall_curve(y_te, proba_te)
        curves[name] = {"roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
                        "confusion": confusion_matrix(y_te, proba_te >= thr).tolist()}
        print(f"{name:22s} CV PR-AUC {cv_m['pr_auc']:.3f} | test PR-AUC {te_m['pr_auc']:.3f} "
              f"ROC-AUC {te_m['roc_auc']:.3f} F2 {te_m['f2']:.3f} recall {te_m['recall']:.2f} thr {thr:.2f}")
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
                mlflow.log_param("threshold", round(thr, 3))
                mlflow.log_metrics({k: float(v) for k, v in row.items() if k.startswith(("cv_", "test_"))})

    table = pd.DataFrame(results)
    candidates = table[table["model"] != "Baseline (majority)"]
    best_name = candidates.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]
    best_pipe, best_oof = fitted[best_name]
    print(f"\nSelected by CV PR-AUC: {best_name}")

    # Platt calibration on out-of-fold scores: class weighting inflates probabilities, but the decision module
    # needs realistic ones (expected value = delta p * revenue at risk). SHAP still explains the raw model score.
    calibrator = PlattCalibrator().fit(best_oof, y_tr)
    best_thr = best_threshold(y_tr, calibrator.transform(best_oof))
    cal_te = calibrator.transform(best_pipe.predict_proba(X_te)[:, 1])
    calibrated_metrics = {f"test_{k}": float(v) for k, v in evaluate(y_te, cal_te, best_thr).items()}
    print(f"Calibrated: mean p {cal_te.mean():.3f} (actual {y_te.mean():.3f}), "
          f"Brier {calibrated_metrics['test_brier']:.4f}, threshold {best_thr:.2f}, "
          f"test recall {calibrated_metrics['test_recall']:.2f} precision {calibrated_metrics['test_precision']:.2f}")

    # Refit the selected configuration on ALL labelled data for production
    prod = Pipeline([("pre", make_preprocessor()), ("clf", best_pipe[-1].__class__(**best_pipe[-1].get_params()))])
    prod.fit(X, y)

    background = X.sample(min(200, len(X)), random_state=config.RANDOM_STATE)
    explainer = Explainer(prod, background)
    global_imp = explainer.global_importance(X)
    grouped, _ = explainer.grouped_shap(X_te)

    artifact = {
        "pipeline": prod, "calibrator": calibrator, "calibrated_test_metrics": calibrated_metrics,
        "model_name": best_name, "threshold": best_thr, "feature_cols": FEATURE_COLS,
        "negative_threshold": config.NEGATIVE_THRESHOLD, "trained_at": db.now(), "n_train": int(len(X)),
        "background": background, "reference_features": X,
        "selected_metrics": table[table["model"] == best_name].iloc[0].to_dict(),
    }
    joblib.dump(artifact, config.ARTIFACTS_DIR / "model.joblib")
    table.to_json(config.ARTIFACTS_DIR / "model_comparison.json", orient="records", indent=2)
    (config.ARTIFACTS_DIR / "curves.json").write_text(json.dumps(curves))
    global_imp.to_json(config.ARTIFACTS_DIR / "shap_global.json", orient="records", indent=2)
    grouped.assign(release_year=X_te["release_year"].values).to_csv(config.ARTIFACTS_DIR / "shap_test_values.csv",
                                                                    index=False)
    _save_shap_plots(explainer, X_te)
    if mlflow:
        with mlflow.start_run(run_name=f"PRODUCTION {best_name}"):
            mlflow.log_param("selected_model", best_name)
            mlflow.log_metrics({k: float(v) for k, v in artifact["selected_metrics"].items()
                                if k.startswith(("cv_", "test_"))})
            mlflow.log_artifacts(str(config.ARTIFACTS_DIR), artifact_path="artifacts")
    print(f"Saved model -> {config.ARTIFACTS_DIR / 'model.joblib'}")
    return artifact


def _save_shap_plots(explainer: Explainer, X: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    g, _ = explainer.grouped_shap(X)
    shown = X[g.columns].copy()
    for c in shown.columns:  # categorical -> codes, so the beeswarm can color them
        if not pd.api.types.is_numeric_dtype(shown[c]):
            shown[c] = shown[c].astype("category").cat.codes
    shap.summary_plot(g.values, shown.astype(float), feature_names=list(g.columns), show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "shap_beeswarm.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
