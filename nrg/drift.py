"""Data drift monitoring: Population Stability Index between training features and features of new requests."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .features import CAT_COLS, HOURS_COLS, NUM_COLS
from .predict import get_predictor
from . import db

MIN_SAMPLES = 20


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if ref.empty or cur.empty:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([ref.min() - 1e-9, ref.median(), ref.max() + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return _psi(r, c)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna()) | set(cur.dropna()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).values
    return _psi(r, c)


def _psi(r, c, eps=1e-4) -> float:
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report() -> dict:
    pred = get_predictor()
    ref = pred.art["reference_features"]
    log = db.list_predictions(limit=1000)
    log = log[log["game_id"].isna()]  # only brand-new games entered by users / API
    if len(log) < MIN_SAMPLES:
        return {"enough_data": False, "n_current": int(len(log)), "min_samples": MIN_SAMPLES}
    cur = pred.features([json.loads(s) for s in log["input_json"]])
    rows = []
    for c in HOURS_COLS + NUM_COLS:
        rows.append({"feature": c, "type": "numeric", "psi": psi_numeric(ref[c], cur[c])})
    for c in CAT_COLS:
        rows.append({"feature": c, "type": "categorical", "psi": psi_categorical(ref[c], cur[c])})
    df = pd.DataFrame(rows)
    df["status"] = pd.cut(df["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "significant"])
    # release_year drifts by construction (new games are newer) -> reported but excluded from the alarm
    alarm = df[(df["status"] == "significant") & (df["feature"] != "release_year")]
    return {"enough_data": True, "n_reference": int(len(ref)), "n_current": int(len(cur)),
            "table": df.astype({"status": str}).to_dict(orient="records"),
            "retrain_recommended": bool(len(alarm)), "drifted_features": alarm["feature"].tolist(),
            "mean_proba_current": float(log["proba_negative"].mean())}
