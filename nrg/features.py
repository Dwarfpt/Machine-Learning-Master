"""Feature engineering. Only information available BEFORE release is used (no sales, no poll counts).

A "game spec" is the raw input contract shared by the DB pipeline, the web form and the API:
    title, console, year, rating, genres ("Action,Sports"), publisher, max_players,
    main_story_hours, n_platforms, is_sequel (optional)

Leakage note: in the raw data a missing HowLongToBeat length means "nobody logged playtime", i.e. the game was
unpopular (31% negative vs 20%). That is post-release information, so training uses only games with a known
length and the web form requires the planned main-story length.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

CONSOLES = ["Nintendo DS", "Sony PSP", "Nintendo Wii", "X360", "PlayStation 3"]
HANDHELD = {"Nintendo DS", "Sony PSP"}
RATINGS = ["E", "T", "M"]
GENRES = ["Action", "Sports", "Strategy", "Role-Playing (RPG)", "Racing / Driving", "Simulation",
          "Adventure", "Educational"]
GENRE_COLS = ["genre_" + re.sub(r"[^a-z]+", "_", g.lower()).strip("_") for g in GENRES]

HOURS_COLS = ["main_story_hours"]
NUM_COLS = ["release_year", "max_players", "n_platforms", "n_genres", "n_publishers",
            "pub_prior_mean_score", "pub_prior_neg_rate", "pub_prior_n_games"]
BIN_COLS = ["handheld", "is_multiplayer", "is_multiplatform", "is_sequel", "has_subtitle",
            "publisher_unknown"] + GENRE_COLS
CAT_COLS = ["console", "rating", "publisher_main"]
FEATURE_COLS = HOURS_COLS + NUM_COLS + BIN_COLS + CAT_COLS

_SEQUEL_RE = re.compile(r"(\b([2-9]|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\b|\b(0[0-9]|20[0-9]{2})\b|'0[0-9])")
PRIOR_STRENGTH = 5  # Bayesian smoothing of publisher track record


def guess_is_sequel(title: str) -> bool:
    return bool(_SEQUEL_RE.search(title or ""))


def split_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def games_to_specs(games: pd.DataFrame) -> pd.DataFrame:
    """Convert rows of the DB `games` table into game specs."""
    n_platforms = games.groupby("title")["release_console"].transform("nunique")
    return pd.DataFrame({
        "game_id": games["game_id"],
        "title": games["title"],
        "console": games["release_console"],
        "year": games["release_year"],
        "rating": games["release_rating"],
        "genres": games["metadata_genres"],
        "publisher": games["metadata_publishers"].fillna(""),
        "max_players": games["features_max_players"],
        "main_story_hours": games["length_main_story_median"].replace(0, np.nan),
        "n_platforms": n_platforms,
        "is_sequel": games["title"].map(guess_is_sequel),
    })


def history_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """Past games with known outcome, used for the time-aware publisher track record."""
    return pd.DataFrame({
        "publisher_main": games["metadata_publishers"].map(lambda p: (split_list(p) or ["Unknown"])[0]),
        "year": games["release_year"],
        "score": games["metrics_review_score"],
        "negative": (games["metrics_review_score"] < config.NEGATIVE_THRESHOLD).astype(int),
    })


def _publisher_prior(publisher: str, year: int, history: pd.DataFrame) -> tuple[float, float, int]:
    past = history[history["year"] < year]
    if past.empty:  # no history at all -> neutral global prior
        return 70.0, 0.2, 0
    g_mean, g_neg = past["score"].mean(), past["negative"].mean()
    own = past[past["publisher_main"] == publisher]
    n = len(own)
    mean = (own["score"].sum() + PRIOR_STRENGTH * g_mean) / (n + PRIOR_STRENGTH)
    neg = (own["negative"].sum() + PRIOR_STRENGTH * g_neg) / (n + PRIOR_STRENGTH)
    return float(mean), float(neg), int(n)


def build_features(specs: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Turn game specs into the model feature frame. `history` must only contain games with known scores;
    for every row only games released in EARLIER years are used (no target leakage)."""
    s = specs.copy()
    out = pd.DataFrame(index=s.index)
    out["main_story_hours"] = pd.to_numeric(s["main_story_hours"], errors="coerce").replace(0, np.nan)
    out["release_year"] = s["year"].astype(int)
    out["max_players"] = s["max_players"].astype(int).clip(1, 64)
    out["n_platforms"] = s["n_platforms"].fillna(1).astype(int).clip(1, 5)

    genres = s["genres"].map(split_list)
    out["n_genres"] = genres.map(len)
    for g, col in zip(GENRES, GENRE_COLS):
        out[col] = genres.map(lambda gs, g=g: int(g in gs))

    pubs = s["publisher"].map(split_list)
    out["n_publishers"] = pubs.map(len)
    out["publisher_unknown"] = (out["n_publishers"] == 0).astype(int)
    out["publisher_main"] = pubs.map(lambda p: p[0] if p else "Unknown")
    priors = [_publisher_prior(p, y, history) for p, y in zip(out["publisher_main"], out["release_year"])]
    out["pub_prior_mean_score"] = [p[0] for p in priors]
    out["pub_prior_neg_rate"] = [p[1] for p in priors]
    out["pub_prior_n_games"] = [p[2] for p in priors]

    out["handheld"] = s["console"].isin(HANDHELD).astype(int)
    out["is_multiplayer"] = (out["max_players"] > 1).astype(int)
    out["is_multiplatform"] = (out["n_platforms"] > 1).astype(int)
    seq = s["is_sequel"] if "is_sequel" in s else pd.Series([None] * len(s), index=s.index)
    out["is_sequel"] = [int(bool(v)) if v is not None and not (isinstance(v, float) and np.isnan(v))
                        else int(guess_is_sequel(t)) for v, t in zip(seq, s["title"])]
    out["has_subtitle"] = s["title"].fillna("").str.contains(r":| - ").astype(int)
    out["console"] = s["console"]
    out["rating"] = s["rating"]
    return out[FEATURE_COLS]


def make_preprocessor() -> ColumnTransformer:
    hours = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                      ("scale", StandardScaler())])
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=15, sparse_output=False)
    return ColumnTransformer([("hours", hours, HOURS_COLS), ("num", num, NUM_COLS),
                              ("bin", "passthrough", BIN_COLS), ("cat", cat, CAT_COLS)],
                             verbose_feature_names_out=True)


def feature_groups(transformed_names: list[str]) -> list[str]:
    """Map each transformed column (e.g. 'cat__console_X360') back to its original feature ('console')."""
    groups = []
    for name in transformed_names:
        _, raw = name.split("__", 1)
        if name.startswith("cat__"):
            raw = next(c for c in CAT_COLS if raw.startswith(c + "_"))
        groups.append(raw)
    return groups
