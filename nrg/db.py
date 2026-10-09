"""SQLite data layer: loading raw CSV into the DB and CRUD for predictions / actions / agent runs."""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    game_id INTEGER,              -- NULL for brand new (unreleased) games entered in the UI
    title TEXT,
    input_json TEXT NOT NULL,     -- raw features used for the prediction
    proba_negative REAL NOT NULL,
    risk_level TEXT NOT NULL,
    model_name TEXT NOT NULL,
    explanation_json TEXT
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    prediction_id INTEGER REFERENCES predictions(id),
    title TEXT,
    action_code TEXT NOT NULL,
    rationale TEXT,
    cost_k REAL,
    expected_benefit_k REAL,
    requires_approval INTEGER NOT NULL,
    status TEXT NOT NULL,         -- pending_approval | approved | rejected | executed
    decided_by TEXT,
    decided_at TEXT,
    source TEXT                   -- llm_agent | rule_based_agent | manual
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    prediction_id INTEGER,
    mode TEXT NOT NULL,           -- llm | rule_based
    language TEXT,
    trace_json TEXT,              -- tool calls made by the agent
    final_answer TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _snake(col: str) -> str:
    col = col.replace("?", "").replace("+", "plus")
    return re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()


def init_db(csv_path=None) -> int:
    """(Re)load the raw CSV into the `games` table and create the operational tables."""
    csv_path = csv_path or config.RAW_CSV
    if not config.RAW_CSV.exists():
        config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        fallback = config.ROOT.parent / "Video_Games_Data.csv"
        if fallback.exists():
            shutil.copy(fallback, config.RAW_CSV)
    df = pd.read_csv(csv_path)
    df.columns = [_snake(c) for c in df.columns]
    df.insert(0, "game_id", range(1, len(df) + 1))
    with connect() as conn:
        df.to_sql("games", conn, if_exists="replace", index=False)
        conn.executescript(SCHEMA)
    return len(df)


def load_games() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql("SELECT * FROM games", conn)


def get_game(game_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
    return dict(row) if row else None


def search_games(query: str, limit: int = 20) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql(
            "SELECT game_id, title, release_console, release_year, metrics_review_score "
            "FROM games WHERE title LIKE ? ORDER BY title LIMIT ?",
            conn, params=(f"%{query}%", limit),
        )


def save_prediction(game_id, title, input_row: dict, proba: float, risk: str, model_name: str,
                    explanation: list | None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (created_at, game_id, title, input_json, proba_negative, risk_level, "
            "model_name, explanation_json) VALUES (?,?,?,?,?,?,?,?)",
            (now(), game_id, title, json.dumps(input_row, default=str), proba, risk, model_name,
             json.dumps(explanation, default=str) if explanation is not None else None),
        )
        return cur.lastrowid


def get_prediction(prediction_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["input"] = json.loads(d.pop("input_json"))
    d["explanation"] = json.loads(d.pop("explanation_json") or "null")
    return d


def list_predictions(limit: int = 200) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(limit,))


def create_action(prediction_id, title, action_code, rationale, cost_k, expected_benefit_k,
                  requires_approval: bool, source: str) -> int:
    status = "pending_approval" if requires_approval else "approved"
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO actions (created_at, prediction_id, title, action_code, rationale, cost_k, "
            "expected_benefit_k, requires_approval, status, source) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now(), prediction_id, title, action_code, rationale, cost_k, expected_benefit_k,
             int(requires_approval), status, source),
        )
        return cur.lastrowid


def get_action(action_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
    return dict(row) if row else None


def set_action_status(action_id: int, status: str, decided_by: str = "manager") -> None:
    with connect() as conn:
        conn.execute("UPDATE actions SET status = ?, decided_by = ?, decided_at = ? WHERE id = ?",
                     (status, decided_by, now(), action_id))


def list_actions(status: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM actions"
    params: tuple = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    with connect() as conn:
        return pd.read_sql(sql + " ORDER BY id DESC", conn, params=params)


def save_agent_run(prediction_id, mode, language, trace, final_answer) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO agent_runs (created_at, prediction_id, mode, language, trace_json, final_answer) "
            "VALUES (?,?,?,?,?,?)",
            (now(), prediction_id, mode, language, json.dumps(trace, default=str, ensure_ascii=False),
             final_answer),
        )
        return cur.lastrowid


if __name__ == "__main__":
    print(f"Loaded {init_db()} games into {config.DB_PATH}")
