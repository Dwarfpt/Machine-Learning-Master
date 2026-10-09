"""Prediction service: loads the production model, predicts on new data, explains and logs every prediction."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config, db
from .explain import Explainer
from .features import build_features, games_to_specs, history_from_games

SPEC_FIELDS = ["title", "console", "year", "rating", "genres", "publisher", "max_players", "main_story_hours",
               "n_platforms", "is_sequel"]


class Predictor:
    def __init__(self):
        art = joblib.load(config.ARTIFACTS_DIR / "model.joblib")
        self.art = art
        self.pipeline = art["pipeline"]
        self.model_name = art["model_name"]
        self.threshold = float(art["threshold"])  # in calibrated probability space
        self.calibrator = art.get("calibrator")
        self.explainer = Explainer(self.pipeline, art["background"])
        self.reload_data()

    def reload_data(self):
        if not config.DB_PATH.exists():  # fresh clone: build the DB from the CSV
            db.init_db()
        self.games = db.load_games()
        self.history = history_from_games(self.games)
        self.specs = games_to_specs(self.games)
        self._X_ref = None

    # ---------- inputs ----------
    def spec_for_game(self, game_id: int) -> dict:
        row = self.specs[self.specs["game_id"] == game_id]
        if row.empty:
            raise KeyError(f"game_id {game_id} not found")
        spec = row.iloc[0].to_dict()
        spec.pop("game_id")
        return {k: _py(v) for k, v in spec.items()}

    def features(self, specs: list[dict]) -> pd.DataFrame:
        return build_features(pd.DataFrame(specs), self.history)

    # ---------- outputs ----------
    def risk_level(self, p: float) -> str:
        if p >= self.threshold:
            return "high"
        if p >= 0.6 * self.threshold:
            return "medium"
        return "low"

    def _proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(X)[:, 1]
        return self.calibrator.transform(raw) if self.calibrator is not None else raw

    def predict_proba(self, specs: list[dict]) -> np.ndarray:
        return self._proba(self.features(specs))

    def predict(self, spec: dict, game_id: int | None = None, save: bool = True) -> dict:
        spec = {k: spec.get(k) for k in SPEC_FIELDS}
        X = self.features([spec])
        p = float(self._proba(X)[0])
        exp = self.explainer.explain_row(X)
        result = {"game_id": game_id, "title": spec["title"], "spec": spec, "proba_negative": p,
                  "risk_level": self.risk_level(p), "threshold": self.threshold, "model_name": self.model_name,
                  "explanation": exp, "features": {k: _py(v) for k, v in X.iloc[0].to_dict().items()}}
        if game_id is not None:  # historical game -> show the real outcome for comparison
            g = self.games[self.games["game_id"] == game_id].iloc[0]
            result["actual_score"] = int(g["metrics_review_score"])
            result["actual_negative"] = bool(g["metrics_review_score"] < config.NEGATIVE_THRESHOLD)
        if save:
            result["prediction_id"] = db.save_prediction(game_id, spec["title"], spec, p, result["risk_level"],
                                                         self.model_name, exp["factors"])
        return result

    # ---------- business context ----------
    def revenue_at_risk_k(self, console: str) -> float:
        """Median revenue gap (k$) between well-reviewed and negatively-reviewed games on this console."""
        g = self.games[self.games["release_console"] == console]
        if len(g) < 20:
            g = self.games
        neg = g["metrics_review_score"] < config.NEGATIVE_THRESHOLD
        gap_m_units = max(g.loc[~neg, "metrics_sales"].median() - g.loc[neg, "metrics_sales"].median(), 0.05)
        return float(gap_m_units * 1e6 * config.AVG_UNIT_PRICE_USD / 1e3)

    def similar_games(self, spec: dict, k: int = 5) -> list[dict]:
        """Nearest released games in the model's feature space (with their real outcome)."""
        X_new = self.pipeline[:-1].transform(self.features([spec]))
        known = self.games["length_main_story_median"] > 0
        if getattr(self, "_X_ref", None) is None:  # transformed reference games, computed once
            self._X_ref = self.pipeline[:-1].transform(build_features(self.specs[known], self.history))
        d = np.linalg.norm(self._X_ref - X_new, axis=1)
        idx = np.argsort(d)[:k]
        rows = self.games[known].iloc[idx]
        return [{"game_id": int(r.game_id), "title": r.title, "console": r.release_console,
                 "year": int(r.release_year), "publisher": _py(r.metadata_publishers) or "Unknown",
                 "genres": r.metadata_genres,
                 "review_score": int(r.metrics_review_score), "sales_m_units": float(r.metrics_sales),
                 "distance": round(float(d[i]), 3)} for i, r in zip(idx, rows.itertuples())]

    def publisher_track_record(self, publisher: str) -> dict:
        pub = (publisher or "").split(",")[0].strip() or "Unknown"
        h = self.history[self.history["publisher_main"] == pub]
        if h.empty:
            return {"publisher": pub, "n_games": 0}
        by_year = h.groupby("year").agg(n=("score", "size"), mean_score=("score", "mean"),
                                        negative_rate=("negative", "mean")).round(3)
        return {"publisher": pub, "n_games": int(len(h)), "mean_score": round(float(h["score"].mean()), 1),
                "negative_rate": round(float(h["negative"].mean()), 3),
                "by_year": by_year.reset_index().to_dict(orient="records")}


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


def _py(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v
