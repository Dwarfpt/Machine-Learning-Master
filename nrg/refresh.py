"""RPA: automatic data refresh — new games arrive as CSV files, the system ingests and scores them.

Flow (triggered every morning by the n8n Schedule node, or by hand: POST /data/refresh):
  1. pick up every *.csv in data/incoming/
  2. validate each row (required columns, known platform, year / score / players ranges) — bad rows are reported
  3. append valid rows to the `games` table (source = "incoming:<file>"), skipping games already in the DB
  4. games WITHOUT a critic score are upcoming releases -> predict the risk, save the predictions
  5. refresh the knowledge-base insights (RAG) and the drift report
  6. move the file to data/incoming/processed/, write a row to `refresh_log`, return a summary for n8n
     (n8n e-mails the high-risk list and logs the run to Google Sheets)

Expected CSV columns: title, console, year, genres, publisher  (+ optional: rating, developer, max_players,
main_story_hours, critic_score, sales_m). Example: docs/examples/incoming_example.csv
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from . import config, db
from .features import CONSOLES, split_list
from .sources import norm_title

INCOMING = config.DATA_DIR / "incoming"
REQUIRED = ["title", "console", "year", "genres", "publisher"]
LOG_SQL = """CREATE TABLE IF NOT EXISTS refresh_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, file TEXT, rows INTEGER, added INTEGER,
    invalid INTEGER, duplicates INTEGER, upcoming_scored INTEGER, high_risk INTEGER, summary_json TEXT)"""


def _validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    errors = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return df.iloc[0:0], [{"row": None, "error": f"missing columns {missing}"}]
    ok = pd.Series(True, index=df.index)
    checks = [
        (df["title"].isna() | (df["title"].astype(str).str.strip() == ""), "empty title"),
        (~df["console"].isin(CONSOLES), "unknown console"),
        (~pd.to_numeric(df["year"], errors="coerce").between(1980, 2035), "year outside 1980-2035"),
    ]
    if "critic_score" in df:
        sc = pd.to_numeric(df["critic_score"], errors="coerce")
        checks.append((sc.notna() & ~sc.between(0, 100), "critic_score outside 0-100"))
    if "max_players" in df:
        mp = pd.to_numeric(df["max_players"], errors="coerce")
        checks.append((mp.notna() & ~mp.between(1, 64), "max_players outside 1-64"))
    for bad, msg in checks:
        for i in df.index[bad & ok]:
            errors.append({"row": int(i) + 2, "title": str(df.at[i, "title"]), "error": msg})
        ok &= ~bad
    return df[ok], errors


def _to_games_rows(df: pd.DataFrame, source: str, start_id: int) -> pd.DataFrame:
    col = lambda c: df[c] if c in df else pd.Series([np.nan] * len(df), index=df.index)  # noqa: E731
    return pd.DataFrame({
        "game_id": range(start_id, start_id + len(df)),
        "title": df["title"].astype(str).str.strip(), "release_console": df["console"],
        "release_year": pd.to_numeric(df["year"]).astype(int), "release_rating": col("rating"),
        "metadata_genres": df["genres"].map(lambda g: ",".join(split_list(g))),
        "metadata_publishers": df["publisher"], "metadata_developers": col("developer"),
        "features_max_players": pd.to_numeric(col("max_players"), errors="coerce"),
        "length_main_story_median": pd.to_numeric(col("main_story_hours"), errors="coerce"),
        "metrics_review_score": pd.to_numeric(col("critic_score"), errors="coerce"),
        "metrics_sales": pd.to_numeric(col("sales_m"), errors="coerce"),
        "source": source,
    })


def run(move_files: bool = True) -> dict:
    from .predict import get_predictor
    INCOMING.mkdir(parents=True, exist_ok=True)
    files = sorted(INCOMING.glob("*.csv"))
    summary = {"started_at": db.now(), "files": [], "added": 0, "invalid": 0, "duplicates": 0,
               "upcoming_scored": 0, "high_risk": []}
    if not files:
        summary["message"] = "no new files in data/incoming"
        return summary
    with db.connect() as conn:
        conn.execute(LOG_SQL)
    games = db.load_games()
    known = set(games["title"].map(norm_title) + "|" + games["release_console"])
    p = get_predictor()
    for f in files:
        raw = pd.read_csv(f)
        valid, errors = _validate(raw)
        keys = valid["title"].map(norm_title) + "|" + valid["console"]
        dup = keys.isin(known) | keys.duplicated()
        new = valid[~dup]
        rows = _to_games_rows(new, f"incoming:{f.name}", int(games["game_id"].max()) + 1)
        with db.connect() as conn:
            rows.to_sql("games", conn, if_exists="append", index=False)
        known |= set(keys[~dup])
        games = pd.concat([games, rows], ignore_index=True)
        p.reload_data()
        upcoming = rows[rows["metrics_review_score"].isna()]
        high = []
        for gid in upcoming["game_id"]:
            r = p.predict(p.spec_for_game(int(gid)), save=True)
            summary["upcoming_scored"] += 1
            if r["risk_level"] == "high":
                high.append({"title": r["title"], "console": r["spec"]["console"], "year": r["spec"]["year"],
                             "proba_negative": round(r["proba_negative"], 3), "prediction_id": r["prediction_id"]})
        file_sum = {"file": f.name, "rows": int(len(raw)), "added": int(len(rows)), "invalid": len(errors),
                    "duplicates": int(dup.sum()), "errors": errors[:20], "upcoming_scored": int(len(upcoming)),
                    "high_risk": len(high)}
        summary["files"].append(file_sum)
        summary["added"] += file_sum["added"]
        summary["invalid"] += file_sum["invalid"]
        summary["duplicates"] += file_sum["duplicates"]
        summary["high_risk"] += high
        with db.connect() as conn:
            conn.execute("INSERT INTO refresh_log (created_at, file, rows, added, invalid, duplicates, "
                         "upcoming_scored, high_risk, summary_json) VALUES (?,?,?,?,?,?,?,?,?)",
                         (db.now(), f.name, file_sum["rows"], file_sum["added"], file_sum["invalid"],
                          file_sum["duplicates"], file_sum["upcoming_scored"], len(high), json.dumps(file_sum)))
        if move_files:
            done = INCOMING / "processed"
            done.mkdir(exist_ok=True)
            shutil.move(str(f), done / f"{db.now().replace(':', '-')}_{f.name}")
    try:  # keep the RAG knowledge base in sync with the new data
        from .rag import build_data_insights
        build_data_insights()
    except Exception as e:
        summary["kb_error"] = str(e)
    summary["high_risk"].sort(key=lambda r: -r["proba_negative"])
    summary["high_risk_count"] = len(summary["high_risk"])
    summary["finished_at"] = db.now()
    return summary


def history(limit: int = 50) -> pd.DataFrame:
    with db.connect() as conn:
        conn.execute(LOG_SQL)
        return pd.read_sql("SELECT * FROM refresh_log ORDER BY id DESC LIMIT ?", conn, params=(limit,))


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
