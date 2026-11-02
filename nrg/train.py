"""Train & compare models, pick the best by cross-validated PR-AUC, evaluate on a time-based holdout,
log everything to MLflow and save the production artifact.

Stage 1  11 models of different families (linear, distance-based, probabilistic, kernel, single tree, bagging,
         boosting) + a majority baseline, each with a small GridSearch, 5-fold stratified CV.
Stage 2  hyper-parameter tuning of the 3 best: RandomizedSearch over a wider space.
Stage 3  combinations of the 5 best: majority (hard) vote, soft vote (average probability), stacking.
Winner   = highest CV PR-AUC among all of them; calibrated (Platt) and refit on all labelled data.

Run:  python -m nrg.train
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config, db, quality
from .calibration import PlattCalibrator
from .ensembles import MajorityVoteClassifier
from .explain import Explainer
from .features import FEATURE_COLS, build_features, games_to_specs, history_from_games, make_preprocessor

warnings.filterwarnings("ignore")
BASELINE = "Baseline (majority)"


def candidate_models(pos_weight: float) -> dict:
    """name -> (estimator, small grid, family). Every model runs single-threaded: the parallelism comes from
    the cross-validation (24 CV workers x 24 model threads oversubscribes the CPU and crashes workers on Windows)."""
    rs = config.RANDOM_STATE
    return {
        BASELINE: (DummyClassifier(strategy="prior"), {}, "baseline"),
        "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                                {"clf__C": [0.03, 0.1, 0.3, 1.0]}, "linear"),
        "K-Nearest Neighbors": (KNeighborsClassifier(), {"clf__n_neighbors": [25, 50, 100],
                                                         "clf__weights": ["uniform", "distance"]}, "distance"),
        "Naive Bayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]}, "probabilistic"),
        "SVM (RBF)": (SVC(probability=True, class_weight="balanced", random_state=rs),
                      {"clf__C": [0.3, 1.0], "clf__gamma": ["scale"]}, "kernel"),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=rs),
                          {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [20, 50]}, "tree"),
        "Random Forest": (RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                 random_state=rs, n_jobs=1),
                          {"clf__max_depth": [8, 14], "clf__min_samples_leaf": [3, 10]}, "bagging"),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=400, class_weight="balanced_subsample",
                                             random_state=rs, n_jobs=1),
                        {"clf__max_depth": [10, 18], "clf__min_samples_leaf": [3, 10]}, "bagging"),
        "AdaBoost": (AdaBoostClassifier(random_state=rs), {"clf__n_estimators": [100, 300],
                                                           "clf__learning_rate": [0.1, 0.5]}, "boosting"),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=200, subsample=0.8, random_state=rs),
                              {"clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]}, "boosting"),
        "LightGBM": (LGBMClassifier(n_estimators=400, learning_rate=0.03, scale_pos_weight=pos_weight,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=rs,
                                    n_jobs=1, verbose=-1),
                     {"clf__num_leaves": [15, 31], "clf__min_child_samples": [20, 50]}, "boosting"),
        "XGBoost": (XGBClassifier(n_estimators=400, learning_rate=0.03, scale_pos_weight=pos_weight,
                                  subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=rs,
                                  n_jobs=1),
                    {"clf__max_depth": [3, 5], "clf__min_child_weight": [1, 5]}, "boosting"),
    }


# wider spaces for stage 2 (only for the families that have something to tune)
TUNING_SPACES = {
    "Logistic Regression": {"clf__C": loguniform(1e-3, 10)},
    "K-Nearest Neighbors": {"clf__n_neighbors": randint(10, 200), "clf__weights": ["uniform", "distance"],
                            "clf__p": [1, 2]},
    "SVM (RBF)": {"clf__C": loguniform(0.1, 10), "clf__gamma": loguniform(1e-3, 1e-1)},
    "Decision Tree": {"clf__max_depth": randint(3, 14), "clf__min_samples_leaf": randint(5, 100)},
    "Random Forest": {"clf__max_depth": randint(6, 30), "clf__min_samples_leaf": randint(1, 20),
                      "clf__max_features": ["sqrt", 0.3, 0.5]},
    "Extra Trees": {"clf__max_depth": randint(6, 30), "clf__min_samples_leaf": randint(1, 20),
                    "clf__max_features": ["sqrt", 0.3, 0.5]},
    "AdaBoost": {"clf__n_estimators": randint(50, 500), "clf__learning_rate": loguniform(0.02, 1.0)},
    "Gradient Boosting": {"clf__max_depth": randint(2, 6), "clf__learning_rate": loguniform(0.01, 0.2),
                          "clf__n_estimators": randint(100, 500), "clf__subsample": uniform(0.6, 0.4)},
    "LightGBM": {"clf__num_leaves": randint(7, 63), "clf__min_child_samples": randint(5, 100),
                 "clf__learning_rate": loguniform(0.01, 0.1), "clf__n_estimators": randint(200, 800),
                 "clf__reg_lambda": loguniform(1e-3, 10), "clf__colsample_bytree": uniform(0.5, 0.5)},
    "XGBoost": {"clf__max_depth": randint(2, 8), "clf__min_child_weight": randint(1, 10),
                "clf__learning_rate": loguniform(0.01, 0.1), "clf__n_estimators": randint(200, 800),
                "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.5, 0.5),
                "clf__reg_lambda": loguniform(1e-2, 10)},
}


def best_threshold(y, proba, beta: float = 1.0) -> float:
    """Threshold maximising F-beta on out-of-fold predictions (F1: balance between missed flops and
    unnecessary interventions; the cost-based decision module then decides WHICH measure is worth it)."""
    grid = np.unique(np.concatenate([np.linspace(0.02, 0.95, 94), np.unique(proba)[:200]]))
    scores = [fbeta_score(y, proba >= t, beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {"pr_auc": average_precision_score(y, proba), "roc_auc": roc_auc_score(y, proba),
            "f1": f1_score(y, pred, zero_division=0), "f2": fbeta_score(y, pred, beta=2, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "accuracy": float((pred == y).mean()), "brier": brier_score_loss(y, proba)}


def _mlflow():
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("negative-review-guard")
        return mlflow
    except Exception as e:  # MLflow is a bonus — training must work without it
        print(f"MLflow disabled: {e}")
        return None


def load_dataset():
    """Labelled games -> (games, X, y, train_mask). n_platforms is counted on ALL games (incl. unscored)."""
    if not config.DB_PATH.exists():
        db.init_db()
    games = db.load_games()
    history = history_from_games(games)
    specs = games_to_specs(games)
    scored = games["metrics_review_score"].notna().values
    games, specs = games[scored].reset_index(drop=True), specs[scored].reset_index(drop=True)
    X = build_features(specs, history)
    y = (games["metrics_review_score"] < config.NEGATIVE_THRESHOLD).astype(int).values
    train_mask = (games["release_year"] < config.TEST_FROM_YEAR).values
    return games, X, y, train_mask


def main() -> dict:
    t0 = time.time()
    if not config.DB_PATH.exists():
        db.init_db()
    quality.main()
    games, X, y, train_mask = load_dataset()
    X_tr, y_tr, X_te, y_te = X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]
    print(f"Labelled games: {len(X)} | Train <{config.TEST_FROM_YEAR}: {len(X_tr)} (neg {y_tr.mean():.1%}) | "
          f"Test {config.TEST_FROM_YEAR}+: {len(X_te)} (neg {y_te.mean():.1%})")

    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    pos_weight = (1 - y_tr.mean()) / y_tr.mean()
    mlflow = _mlflow()
    results, fitted, curves = [], {}, {}

    def record(name, stage, family, pipe, params):
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thr = best_threshold(y_tr, oof) if name != BASELINE else 0.5
        pipe = clone(pipe).fit(X_tr, y_tr)
        proba_te = pipe.predict_proba(X_te)[:, 1]
        cv_m, te_m = evaluate(y_tr, oof, thr), evaluate(y_te, proba_te, thr)
        row = {"model": name, "stage": stage, "family": family, "best_params": params, "threshold": thr,
               **{f"cv_{k}": v for k, v in cv_m.items()}, **{f"test_{k}": v for k, v in te_m.items()}}
        results.append(row)
        fitted[name] = (pipe, oof, thr)
        fpr, tpr, _ = roc_curve(y_te, proba_te)
        prec, rec, _ = precision_recall_curve(y_te, proba_te)
        curves[name] = {"roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
                        "confusion": confusion_matrix(y_te, proba_te >= thr).tolist()}
        print(f"[{stage}] {name:32s} CV PR-AUC {cv_m['pr_auc']:.3f} | test PR-AUC {te_m['pr_auc']:.3f} "
              f"ROC {te_m['roc_auc']:.3f} F1 {te_m['f1']:.3f} recall {te_m['recall']:.2f}  ({time.time() - t0:.0f}s)")
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.set_tags({"stage": stage, "family": family, "data": "kaggle+corgis"})
                mlflow.log_params({k.replace("clf__", ""): v for k, v in (params or {}).items()})
                mlflow.log_param("threshold", round(thr, 3))
                mlflow.log_metrics({k: float(v) for k, v in row.items() if k.startswith(("cv_", "test_"))})

    # ---- stage 1: compare many model families
    models = candidate_models(pos_weight)
    for name, (clf, grid, family) in models.items():
        pipe = Pipeline([("pre", make_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid or {"clf__strategy": ["prior"]}, scoring="average_precision", cv=cv,
                              n_jobs=-1).fit(X_tr, y_tr)
        record(name, "1-compare", family, search.best_estimator_, search.best_params_)

    # ---- stage 2: tune the winners harder
    singles = pd.DataFrame(results).query("model != @BASELINE").sort_values("cv_pr_auc", ascending=False)
    for name in singles["model"].head(3):
        if name not in TUNING_SPACES:
            continue
        base = Pipeline([("pre", make_preprocessor()), ("clf", clone(models[name][0]))])
        search = RandomizedSearchCV(base, TUNING_SPACES[name], n_iter=25, scoring="average_precision", cv=cv,
                                    n_jobs=-1, random_state=config.RANDOM_STATE).fit(X_tr, y_tr)
        record(f"{name} (tuned)", "2-tuning", models[name][2], search.best_estimator_,
               {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()})

    # ---- stage 3: combine the 5 best different models
    table = pd.DataFrame(results).query("model != @BASELINE").sort_values("cv_pr_auc", ascending=False)
    top, seen = [], set()
    for name in table["model"]:
        base_name = name.replace(" (tuned)", "")
        if base_name not in seen:
            seen.add(base_name)
            top.append(name)
        if len(top) == 5:
            break
    members = [(n.replace(" ", "_").replace("(", "").replace(")", ""), fitted[n][0][-1]) for n in top]
    thresholds = [fitted[n][2] for n in top]
    combos = {
        "Majority vote (3 of 5)": MajorityVoteClassifier(members, thresholds),
        "Soft voting (average)": VotingClassifier(members, voting="soft"),
        "Stacking (logistic meta)": StackingClassifier(members, final_estimator=LogisticRegression(max_iter=2000),
                                                       cv=5, stack_method="predict_proba", n_jobs=1),
    }
    for name, ens in combos.items():
        record(name, "3-ensemble", "ensemble", Pipeline([("pre", make_preprocessor()), ("clf", ens)]),
               {"members": ", ".join(top)})

    table = pd.DataFrame(results)
    candidates = table[table["model"] != BASELINE]
    best_name = candidates.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]
    best_pipe, best_oof, _ = fitted[best_name]
    print(f"\nSelected by CV PR-AUC: {best_name}")

    # Platt calibration on out-of-fold scores: class weighting / vote shares are not probabilities, but the
    # decision module needs realistic ones (expected value = delta p * revenue at risk).
    calibrator = PlattCalibrator().fit(best_oof, y_tr)
    # production threshold by F2 (recall weighs more): a missed flop costs hundreds of k$ of revenue, a false alarm
    # costs at most a cheap measure (the decision module and the manager filter expensive ones)
    best_thr = best_threshold(y_tr, calibrator.transform(best_oof), beta=2.0)
    cal_te = calibrator.transform(best_pipe.predict_proba(X_te)[:, 1])
    calibrated_metrics = {f"test_{k}": float(v) for k, v in evaluate(y_te, cal_te, best_thr).items()}
    print(f"Calibrated: mean p {cal_te.mean():.3f} (actual {y_te.mean():.3f}), "
          f"Brier {calibrated_metrics['test_brier']:.4f}, threshold {best_thr:.2f}, "
          f"test recall {calibrated_metrics['test_recall']:.2f} precision {calibrated_metrics['test_precision']:.2f}")

    # Refit the selected configuration on ALL labelled data for production
    prod = clone(best_pipe).fit(X, y)

    background = X.sample(min(200, len(X)), random_state=config.RANDOM_STATE)
    explainer = Explainer(prod, background)
    global_imp = explainer.global_importance(X_te)
    shap_sample = X_te.sample(min(300, len(X_te)), random_state=config.RANDOM_STATE)
    grouped, _ = explainer.grouped_shap(shap_sample)

    artifact = {
        "pipeline": prod, "calibrator": calibrator, "calibrated_test_metrics": calibrated_metrics,
        "model_name": best_name, "threshold": best_thr, "feature_cols": FEATURE_COLS,
        "negative_threshold": config.NEGATIVE_THRESHOLD, "trained_at": db.now(), "n_train": int(len(X)),
        "background": background, "reference_features": X, "test_from_year": config.TEST_FROM_YEAR,
        "selected_metrics": table[table["model"] == best_name].iloc[0].to_dict(),
    }
    joblib.dump(artifact, config.ARTIFACTS_DIR / "model.joblib")
    table.to_json(config.ARTIFACTS_DIR / "model_comparison.json", orient="records", indent=2, default_handler=str)
    (config.ARTIFACTS_DIR / "curves.json").write_text(json.dumps(curves))
    global_imp.to_json(config.ARTIFACTS_DIR / "shap_global.json", orient="records", indent=2)
    grouped.assign(release_year=shap_sample["release_year"].values).to_csv(
        config.ARTIFACTS_DIR / "shap_test_values.csv", index=False)
    _save_shap_plots(grouped, shap_sample)
    if mlflow:
        with mlflow.start_run(run_name=f"PRODUCTION {best_name}"):
            mlflow.set_tags({"stage": "production", "data": "kaggle+corgis"})
            mlflow.log_param("selected_model", best_name)
            mlflow.log_metrics({k: float(v) for k, v in artifact["selected_metrics"].items()
                                if k.startswith(("cv_", "test_"))})
            mlflow.log_metrics({f"calibrated_{k}": v for k, v in calibrated_metrics.items()})
            mlflow.log_artifacts(str(config.ARTIFACTS_DIR), artifact_path="artifacts")
    print(f"Saved model -> {config.ARTIFACTS_DIR / 'model.joblib'}  (total {time.time() - t0:.0f}s)")
    return artifact


def _save_shap_plots(g: pd.DataFrame, X: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

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
