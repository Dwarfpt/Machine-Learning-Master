"""AutoML with PyCaret: try ~15 classifiers and ~15 regressors automatically, tune the winner, blend and stack.

Runs in its own Python 3.11 environment (PyCaret 3.3 does not support Python 3.12+):
    .venv-automl\\Scripts\\python automl\\pycaret_automl.py
Input : data/processed/train.csv, test.csv  (made by `python -m nrg.export_dataset` — the SAME features and
        the SAME time split as the main pipeline, so the results are directly comparable)
Output: reports/automl_classification.csv, reports/automl_regression.csv, reports/automl_summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORTS = ROOT / "data" / "processed", ROOT / "reports"
CAT = ["console", "rating", "publisher_main"]
SEED = 42


def classification(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    from pycaret import classification as C

    tr, te = train.drop(columns=["score"]), test.drop(columns=["score"])
    C.setup(data=tr, test_data=te, target="negative", categorical_features=CAT, max_encoding_ohe=40,
            normalize=True, index=False, fold_strategy="stratifiedkfold", fold=5, fold_shuffle=True, session_id=SEED,
            data_split_stratify=True, verbose=False, html=False, n_jobs=-1)
    C.add_metric("pr_auc", "PR-AUC", average_precision_score, target="pred_proba")
    best3 = C.compare_models(sort="PR-AUC", n_select=3, verbose=False,
                             exclude=["lda", "qda", "dummy", "svm", "ridge"])  # no probabilities -> no PR-AUC
    board = C.pull()
    board.to_csv(REPORTS / "automl_classification.csv")
    print(board[["Model", "Accuracy", "AUC", "Recall", "Prec.", "F1", "PR-AUC"]].head(15).to_string())

    tuned = C.tune_model(best3[0], optimize="PR-AUC", n_iter=25, choose_better=True, verbose=False)
    blended = C.blend_models(best3, method="soft", optimize="PR-AUC", choose_better=False, verbose=False)
    stacked = C.stack_models(best3, optimize="PR-AUC", choose_better=False, verbose=False)
    out = {}
    for name, model in [("best single", best3[0]), ("tuned best", tuned), ("blend top-3", blended),
                        ("stack top-3", stacked)]:
        pred = C.predict_model(model, data=te.drop(columns=["negative"]), raw_score=True, verbose=False)
        p = pred["prediction_score_1"].values
        out[name] = {"model": str(model.__class__.__name__) if name != "best single" else board.iloc[0]["Model"],
                     "test_pr_auc": float(average_precision_score(te["negative"], p))}
        print(f"  {name:12s} test PR-AUC {out[name]['test_pr_auc']:.3f}")
    out["leaderboard_top5"] = board.head(5)[["Model", "AUC", "F1", "PR-AUC"]].round(4).to_dict(orient="records")
    return out


def regression(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    from pycaret import regression as R

    tr, te = train.drop(columns=["negative"]), test.drop(columns=["negative"])
    R.setup(data=tr, test_data=te, target="score", categorical_features=CAT, max_encoding_ohe=40, normalize=True, index=False,
            fold=5, fold_shuffle=True, session_id=SEED, verbose=False, html=False, n_jobs=-1)
    best = R.compare_models(sort="MAE", n_select=3, verbose=False, exclude=["dummy"])
    board = R.pull()
    board.to_csv(REPORTS / "automl_regression.csv")
    print(board[["Model", "MAE", "RMSE", "R2"]].head(15).to_string())
    blended = R.blend_models(best, choose_better=False, verbose=False)
    pred = R.predict_model(blended, data=te.drop(columns=["score"]), verbose=False)
    mae = float((pred["prediction_label"] - te["score"].values).abs().mean())
    print(f"  blend top-3 test MAE {mae:.2f}")
    return {"best_model": board.iloc[0]["Model"], "cv_mae": float(board.iloc[0]["MAE"]),
            "cv_r2": float(board.iloc[0]["R2"]), "blend_test_mae": mae,
            "leaderboard_top5": board.head(5)[["Model", "MAE", "RMSE", "R2"]].round(4).to_dict(orient="records")}


def main() -> None:
    t0 = time.time()
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    print(f"train {train.shape}, test {test.shape}")
    summary = {"classification": classification(train, test), "regression": regression(train, test),
               "seconds": round(time.time() - t0)}
    (REPORTS / "automl_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"done in {summary['seconds']} s -> reports/automl_*.csv, automl_summary.json")


if __name__ == "__main__":
    main()
