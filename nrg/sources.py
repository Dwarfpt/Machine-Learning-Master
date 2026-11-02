"""Two data sources merged into one `games` table.

1. Kaggle "Video Game Sales with Ratings" (VGChartz sales + Metacritic scores, Dec 2016): 16 719 rows,
   31 platforms, 1980-2016; critic score for 8 137 rows; developer, ESRB rating, sales by region, user score.
2. CORGIS Video Games (2004-2008, 5 consoles, 1 212 rows): HowLongToBeat play time, max players.

Matching: normalised title + platform. Kaggle gives the critic score, genre, publisher, developer and sales;
CORGIS adds the main-story length and max players where the game exists in both. CORGIS games missing from
Kaggle are added with their own score. One row of `games` = one game on one platform.
Post-release columns (sales, user score, critic/user counts) are kept for EDA and business maths only —
features.py never uses them.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

KAGGLE_CSV = config.DATA_DIR / "raw" / "kaggle" / "Video_Games_Sales_as_at_22_Dec_2016.csv"
# original: kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings; public mirror used for automatic download
KAGGLE_URL = "https://raw.githubusercontent.com/gexijin/learnR/master/datasets/video-game-sales-at-22-Dec-2016.csv"

# Kaggle platform code -> display name (the 5 CORGIS consoles keep their CORGIS names)
PLATFORMS = {
    "DS": "Nintendo DS", "PSP": "Sony PSP", "Wii": "Nintendo Wii", "X360": "X360", "PS3": "PlayStation 3",
    "PS2": "PlayStation 2", "PS4": "PlayStation 4", "PS": "PlayStation", "PSV": "PS Vita", "XB": "Xbox",
    "XOne": "Xbox One", "GC": "GameCube", "GBA": "Game Boy Advance", "GB": "Game Boy", "3DS": "Nintendo 3DS",
    "WiiU": "Wii U", "N64": "Nintendo 64", "PC": "PC", "DC": "Dreamcast", "SAT": "Saturn", "SNES": "SNES",
    "NES": "NES", "GEN": "Genesis", "2600": "Atari 2600", "SCD": "Sega CD", "WS": "WonderSwan", "NG": "Neo Geo",
    "TG16": "TurboGrafx-16", "3DO": "3DO", "GG": "Game Gear", "PCFX": "PC-FX",
}
# release year of each platform -> console lifecycle feature (early / mid / late generation)
PLATFORM_LAUNCH = {
    "Nintendo DS": 2004, "Sony PSP": 2004, "Nintendo Wii": 2006, "X360": 2005, "PlayStation 3": 2006,
    "PlayStation 2": 2000, "PlayStation 4": 2013, "PlayStation": 1994, "PS Vita": 2011, "Xbox": 2001,
    "Xbox One": 2013, "GameCube": 2001, "Game Boy Advance": 2001, "Game Boy": 1989, "Nintendo 3DS": 2011,
    "Wii U": 2012, "Nintendo 64": 1996, "PC": 1985, "Dreamcast": 1998, "Saturn": 1994, "SNES": 1990,
    "NES": 1983, "Genesis": 1988, "Atari 2600": 1977,
}
HANDHELDS = {"Nintendo DS", "Sony PSP", "PS Vita", "Game Boy Advance", "Game Boy", "Nintendo 3DS",
             "WonderSwan", "Game Gear"}
# CORGIS genre names -> the Kaggle genre vocabulary
CORGIS_GENRE = {"Role-Playing (RPG)": "Role-Playing", "Racing / Driving": "Racing", "Educational": "Misc"}
# rare ESRB codes folded into the main four
RATING_MAP = {"K-A": "E", "EC": "E", "AO": "M", "RP": None}


def norm_title(t: str) -> str:
    t = str(t).lower().replace("&", "and")
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t.replace("pokmon", "pokemon")


def _snake(col: str) -> str:
    col = col.replace("?", "").replace("+", "plus")
    return re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()


def load_corgis() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV)
    df.columns = [_snake(c) for c in df.columns]
    return df


def ensure_kaggle() -> None:
    """Download the Kaggle CSV on a fresh clone (the file is not committed: the dataset has no explicit licence)."""
    if KAGGLE_CSV.exists():
        return
    import urllib.request
    KAGGLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = KAGGLE_CSV.with_suffix(".part")
    urllib.request.urlretrieve(KAGGLE_URL, tmp)
    head = tmp.read_text(encoding="utf-8").splitlines()
    if not head[0].startswith("Name,Platform,Year_of_Release") or len(head) < 16000:
        tmp.unlink()
        raise RuntimeError(f"unexpected content downloaded from {KAGGLE_URL}")
    tmp.replace(KAGGLE_CSV)


def load_kaggle() -> pd.DataFrame:
    ensure_kaggle()
    df = pd.read_csv(KAGGLE_CSV)
    df.columns = [_snake(c) for c in df.columns]
    return df


def build_games(corgis: pd.DataFrame, kaggle: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Unified `games` table + an integration report (row counts, matches, conflicts)."""
    k = kaggle.copy()
    k = k[k["name"].notna() & k["year_of_release"].notna()]
    k = k[k["year_of_release"] <= 2016]  # one mis-dated row (2020) in the source
    k["release_console"] = k["platform"].map(PLATFORMS).fillna(k["platform"])
    k["key"] = k["name"].map(norm_title) + "|" + k["release_console"]
    k = k.sort_values("global_sales", ascending=False).drop_duplicates("key")  # 2 duplicated game-platform rows
    k["user_score"] = pd.to_numeric(k["user_score"].replace("tbd", np.nan), errors="coerce")

    c = corgis.copy()
    c["key"] = c["title"].map(norm_title) + "|" + c["release_console"]
    c = c.drop_duplicates("key")
    c_hours = c["length_main_story_median"].replace(0, np.nan)
    c_genres = c["metadata_genres"].fillna("").map(
        lambda g: ",".join(dict.fromkeys(CORGIS_GENRE.get(x.strip(), x.strip()) for x in g.split(",") if x.strip())))
    corg = pd.DataFrame({"key": c["key"], "c_title": c["title"], "c_console": c["release_console"],
                         "c_year": c["release_year"], "c_rating": c["release_rating"], "c_genres": c_genres,
                         "c_publisher": c["metadata_publishers"], "c_players": c["features_max_players"],
                         "c_hours": c_hours, "c_score": c["metrics_review_score"], "c_sales": c["metrics_sales"]})

    m = k.merge(corg, on="key", how="outer", indicator=True)
    both, k_only, c_only = (m["_merge"] == "both"), (m["_merge"] == "left_only"), (m["_merge"] == "right_only")
    pick = lambda a, b: m[a].where(m[a].notna(), m[b])  # noqa: E731  Kaggle first, CORGIS as fallback
    genres_k = m["genre"].fillna("")
    genres = [",".join(dict.fromkeys([g for g in [gk] + (gc.split(",") if isinstance(gc, str) else []) if g]))
              for gk, gc in zip(genres_k, m["c_genres"])]
    rating = pick("rating", "c_rating").map(lambda r: RATING_MAP.get(r, r) if isinstance(r, str) else None)
    games = pd.DataFrame({
        "title": pick("name", "c_title"),
        "release_console": pick("release_console", "c_console"),
        "release_year": pick("year_of_release", "c_year").astype(int),
        "release_rating": rating,
        "metadata_genres": genres,
        "metadata_publishers": pick("publisher", "c_publisher").replace({"Unknown": np.nan}),
        "metadata_developers": m["developer"],
        "features_max_players": m["c_players"],
        "length_main_story_median": m["c_hours"],
        "metrics_review_score": pick("critic_score", "c_score"),
        "metrics_sales": pick("global_sales", "c_sales"),
        "metrics_sales_na": m["na_sales"], "metrics_sales_eu": m["eu_sales"], "metrics_sales_jp": m["jp_sales"],
        "metrics_critic_count": m["critic_count"], "metrics_user_score": m["user_score"],
        "metrics_user_count": m["user_count"],
        "source": np.select([both, k_only, c_only], ["both", "kaggle", "corgis"], "?"),
    })
    games = games.sort_values(["release_year", "title"]).reset_index(drop=True)
    games.insert(0, "game_id", range(1, len(games) + 1))

    scored = games["metrics_review_score"].notna()
    conflict = (m.loc[both, "critic_score"] - m.loc[both, "c_score"]).abs()
    report = {
        "kaggle_rows": int(len(kaggle)), "corgis_rows": int(len(corgis)), "games_rows": int(len(games)),
        "matched_both": int(both.sum()), "kaggle_only": int(k_only.sum()), "corgis_only": int(c_only.sum()),
        "with_critic_score": int(scored.sum()), "with_play_time": int(games["length_main_story_median"].notna().sum()),
        "platforms": int(games["release_console"].nunique()),
        "years": [int(games["release_year"].min()), int(games["release_year"].max())],
        "score_conflicts_gt5": int((conflict > 5).sum()), "score_mean_abs_diff": round(float(conflict.mean()), 2),
    }
    return games, report
