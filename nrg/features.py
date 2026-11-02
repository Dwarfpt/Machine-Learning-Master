"""Feature engineering. Only information available BEFORE release is used (no sales, scores, user counts).

A "game spec" is the raw input contract shared by the DB pipeline, the web form and the API:
    title, console, year, rating, genres ("Action,Sports"), publisher, developer, max_players,
    main_story_hours, n_platforms, is_sequel (optional)

Data: Kaggle (Metacritic scores, 31 platforms, 1980-2016) merged with CORGIS (play time, players), see sources.py.
Play time and max players are known only for the CORGIS part (~900 / 1 200 games) -> median-imputed, so a
missing value carries no signal (it mostly means "not in CORGIS", not "unpopular game").
Seasonal / cyclical component: the dataset has release YEAR only (no month), so "season" is modelled as the
console life cycle — years since the platform launch (early titles vs late-generation titles) — plus the year trend.
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
from .sources import HANDHELDS, PLATFORM_LAUNCH, PLATFORMS

CONSOLES = list(dict.fromkeys(PLATFORMS.values()))
RATINGS = ["E", "E10+", "T", "M"]
GENRES = ["Action", "Sports", "Shooter", "Role-Playing", "Racing", "Misc", "Platform", "Simulation", "Strategy",
          "Fighting", "Adventure", "Puzzle"]
GENRE_COLS = ["genre_" + re.sub(r"[^a-z]+", "_", g.lower()).strip("_") for g in GENRES]

HOURS_COLS = ["main_story_hours"]
NUM_COLS = ["release_year", "console_age", "max_players", "n_platforms", "n_genres",
            "pub_prior_mean_score", "pub_prior_neg_rate", "pub_prior_n_games",
            "dev_prior_mean_score", "dev_prior_n_games"]
BIN_COLS = ["handheld", "is_multiplatform", "is_sequel", "has_subtitle", "publisher_unknown",
            "developer_is_publisher"] + GENRE_COLS
CAT_COLS = ["console", "rating", "publisher_main"]
FEATURE_COLS = HOURS_COLS + NUM_COLS + BIN_COLS + CAT_COLS

_SEQUEL_RE = re.compile(r"(\b([2-9]|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\b|\b(0[0-9]|20[0-9]{2})\b|'0[0-9])")
PRIOR_STRENGTH = 5  # Bayesian smoothing of publisher / developer track record


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
        "developer": games["metadata_developers"].fillna(""),
        "max_players": games["features_max_players"],
        "main_story_hours": games["length_main_story_median"],
        "n_platforms": n_platforms,
        "is_sequel": games["title"].map(guess_is_sequel),
    })


def history_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """Past games with known outcome, used for the time-aware publisher / developer track record."""
    g = games[games["metrics_review_score"].notna()]
    return pd.DataFrame({
        "publisher_main": g["metadata_publishers"].map(lambda p: (split_list(p) or ["Unknown"])[0]),
        "developer_main": g["metadata_developers"].map(lambda p: (split_list(p) or ["Unknown"])[0]),
        "year": g["release_year"],
        "score": g["metrics_review_score"],
        "negative": (g["metrics_review_score"] < config.NEGATIVE_THRESHOLD).astype(int),
    })


def _priors(keys: pd.Series, years: pd.Series, history: pd.DataFrame, col: str) -> pd.DataFrame:
    """Smoothed mean score / negative rate / count of a publisher or developer, using ONLY earlier years.
    Vectorised: cumulative per-year statistics, then a lookup of 'everything before year y'."""
    h = history[history[col].notna()]
    year_tot = h.groupby("year").agg(s=("score", "sum"), n=("score", "size"), k=("negative", "sum")).sort_index()
    cum_tot = year_tot.cumsum()
    own = h.groupby([col, "year"]).agg(s=("score", "sum"), n=("score", "size"), k=("negative", "sum"))
    own_cum = own.groupby(level=0).cumsum()
    out = []
    for key, y in zip(keys, years):
        tot = cum_tot[cum_tot.index < y]
        if tot.empty:
            out.append((70.0, 0.2, 0))
            continue
        t = tot.iloc[-1]
        g_mean, g_neg = t["s"] / t["n"], t["k"] / t["n"]
        s = n = k = 0.0
        if key in own_cum.index.get_level_values(0):
            o = own_cum.loc[key]
            o = o[o.index < y]
            if not o.empty:
                s, n, k = o.iloc[-1]["s"], o.iloc[-1]["n"], o.iloc[-1]["k"]
        out.append(((s + PRIOR_STRENGTH * g_mean) / (n + PRIOR_STRENGTH),
                    (k + PRIOR_STRENGTH * g_neg) / (n + PRIOR_STRENGTH), int(n)))
    return pd.DataFrame(out, columns=["mean", "neg", "n"], index=keys.index)


def build_features(specs: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Turn game specs into the model feature frame. `history` must only contain games with known scores;
    for every row only games released in EARLIER years are used (no target leakage)."""
    s = specs.copy()
    for col in ("developer", "max_players", "main_story_hours"):
        if col not in s:
            s[col] = None
    out = pd.DataFrame(index=s.index)
    out["main_story_hours"] = pd.to_numeric(s["main_story_hours"], errors="coerce").replace(0, np.nan)
    out["release_year"] = s["year"].astype(int)
    launch = s["console"].map(PLATFORM_LAUNCH)
    out["console_age"] = (out["release_year"] - launch).clip(lower=0)
    out["max_players"] = pd.to_numeric(s["max_players"], errors="coerce").clip(1, 64)
    out["n_platforms"] = pd.to_numeric(s["n_platforms"], errors="coerce").fillna(1).clip(1, 8)

    genres = s["genres"].map(split_list)
    out["n_genres"] = genres.map(len)
    for g, col in zip(GENRES, GENRE_COLS):
        out[col] = genres.map(lambda gs, g=g: int(g in gs))

    pubs = s["publisher"].map(split_list)
    out["publisher_unknown"] = (pubs.map(len) == 0).astype(int)
    out["publisher_main"] = pubs.map(lambda p: p[0] if p else "Unknown")
    devs = s["developer"].map(split_list).map(lambda d: d[0] if d else None)
    pp = _priors(out["publisher_main"], out["release_year"], history, "publisher_main")
    dp = _priors(devs.fillna("Unknown"), out["release_year"], history, "developer_main")
    out["pub_prior_mean_score"], out["pub_prior_neg_rate"], out["pub_prior_n_games"] = pp["mean"], pp["neg"], pp["n"]
    out["dev_prior_mean_score"], out["dev_prior_n_games"] = dp["mean"].where(devs.notna()), dp["n"].where(devs.notna(), 0)
    out["developer_is_publisher"] = [int(bool(d) and d.lower() in p.lower())
                                     for d, p in zip(devs.fillna(""), out["publisher_main"])]

    out["handheld"] = s["console"].isin(HANDHELDS).astype(int)
    out["is_multiplatform"] = (out["n_platforms"] > 1).astype(int)
    seq = s["is_sequel"] if "is_sequel" in s else pd.Series([None] * len(s), index=s.index)
    out["is_sequel"] = [int(bool(v)) if v is not None and not (isinstance(v, float) and np.isnan(v))
                        else int(guess_is_sequel(t)) for v, t in zip(seq, s["title"])]
    out["has_subtitle"] = s["title"].fillna("").str.contains(r":| - ").astype(int)
    out["console"] = s["console"]
    out["rating"] = s["rating"].fillna("Unknown")
    return out[FEATURE_COLS]


def make_preprocessor() -> ColumnTransformer:
    hours = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                      ("scale", StandardScaler())])
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False)
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
