"""Data quality checks on BOTH raw sources (CORGIS, Kaggle) and on the merged `games` table.
Produces reports/data_quality.json (every finding is tagged with its source)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, db

LENGTH_COLS_PREFIX = "length_"
BOOL_COLS = ["features_handheld", "features_multiplatform", "features_online", "metadata_licensed",
             "metadata_sequel", "release_re_release"]


def run_quality_checks(df: pd.DataFrame) -> dict:
    """Checks of the raw CORGIS table."""
    report: dict = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "issues": []}

    def issue(severity, check, detail, affected=None, decision=""):
        report["issues"].append({"severity": severity, "check": check, "detail": detail,
                                 "affected_rows": None if affected is None else int(affected),
                                 "decision": decision})

    # 1. Missing values
    missing = df.isna().sum()
    report["missing"] = {k: int(v) for k, v in missing[missing > 0].items()}
    for col, n in report["missing"].items():
        issue("medium", "missing_values", f"{col}: {n} missing ({n / len(df):.1%})", n,
              "Publisher -> category 'Unknown'" if "publisher" in col else "median imputation in the model")

    # 2. Duplicates
    full_dups = int(df.drop(columns=["game_id"]).duplicated().sum())
    title_dups = int(df["title"].duplicated().sum())
    report["duplicates"] = {"full_rows": full_dups, "same_title_other_platform": title_dups}
    if full_dups:
        issue("high", "duplicates", f"{full_dups} fully duplicated rows", full_dups, "drop")
    issue("info", "duplicates", f"{title_dups} titles appear on several consoles (not true duplicates: "
          "one row = one game x platform)", title_dups, "Use as feature n_platforms")

    # 3. Constant / non-informative columns
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    report["constant_columns"] = constant
    for c in constant:
        issue("high", "constant_column", f"{c} has a single value ({df[c].iloc[0]!r}) for all rows",
              len(df), "Drop; re-derive (handheld from console, multiplatform from titles, "
              "sequel from title pattern)" if c in BOOL_COLS else "Drop")

    # 4. Range / validity checks
    rules = {
        "metrics_review_score": (0, 100),
        "release_year": (1990, 2030),
        "features_max_players": (1, 64),
        "metrics_sales": (0, None),
        "metrics_used_price": (0, None),
    }
    report["range_violations"] = {}
    for col, (lo, hi) in rules.items():
        bad = df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
        report["range_violations"][col] = int(bad.sum())
        if bad.any():
            issue("high", "range", f"{col}: {int(bad.sum())} values outside [{lo}, {hi}]", bad.sum(), "clip")

    # 5. Zeros that actually mean "no data" in HowLongToBeat length columns
    length_cols = [c for c in df.columns if c.startswith(LENGTH_COLS_PREFIX) and not c.endswith("polled")]
    zero_main = int((df["length_main_story_median"] == 0).sum())
    report["length_zero_as_missing"] = {c: int((df[c] == 0).sum()) for c in length_cols}
    issue("medium", "zero_as_missing", f"length_main_story_median == 0 for {zero_main} games "
          f"({zero_main / len(df):.1%}) — nobody polled, 0 means 'unknown'", zero_main,
          "Replace 0 by NaN, median imputation inside the model pipeline")

    # 6. Logical consistency: rushed <= median <= leisure for main story
    m = df[(df["length_main_story_median"] > 0)]
    inconsistent = int(((m["length_main_story_rushed"] > m["length_main_story_leisure"] + 1e-9)).sum())
    report["length_inconsistent"] = inconsistent
    if inconsistent:
        issue("low", "consistency", f"{inconsistent} games where rushed > leisure play time", inconsistent,
              "keep (poll noise)")

    # 7. Outliers (IQR) on main story length
    s = m["length_main_story_median"]
    q1, q3 = s.quantile([0.25, 0.75])
    out = int(((s < q1 - 3 * (q3 - q1)) | (s > q3 + 3 * (q3 - q1))).sum())
    report["length_outliers_3iqr"] = out
    issue("low", "outliers", f"{out} extreme main-story lengths (>3 IQR)", out,
          "keep (RPGs are long by nature); trees are robust, log1p for linear model")

    # 8. Leakage audit: columns only known AFTER release
    leakage = ["metrics_sales", "metrics_used_price"] + [c for c in df.columns if c.endswith("polled")]
    report["post_release_columns"] = leakage
    issue("high", "target_leakage", "Sales, used price and HowLongToBeat poll counts are only known after "
          "release", None, "Excluded from model features; sales used only for business impact estimate")

    # 9. Target distribution
    neg = (df["metrics_review_score"] < config.NEGATIVE_THRESHOLD)
    report["target"] = {"threshold": config.NEGATIVE_THRESHOLD, "negative_rate": float(neg.mean()),
                        "n_negative": int(neg.sum())}
    report["score_stats"] = {k: float(v) for k, v in df["metrics_review_score"].describe().items()}
    report["generated_at"] = db.now()
    return report


def run_kaggle_checks(df: pd.DataFrame) -> dict:
    """Checks of the raw Kaggle table (VGChartz + Metacritic)."""
    report: dict = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "issues": []}

    def issue(severity, check, detail, affected=None, decision=""):
        report["issues"].append({"severity": severity, "check": check, "detail": detail,
                                 "affected_rows": None if affected is None else int(affected), "decision": decision})

    n = len(df)
    for col, decision in [("critic_score", "no target -> not used for training (kept for sales / n_platforms)"),
                          ("year_of_release", "drop row"), ("name", "drop row"),
                          ("developer", "developer prior = global prior (median imputation)"),
                          ("rating", "category 'Unknown'")]:
        k = int(df[col].isna().sum())
        issue("high" if col == "critic_score" else "medium", "missing_values",
              f"{col}: {k} missing ({k / n:.1%})", k, decision)
    tbd = int((df["user_score"] == "tbd").sum())
    issue("medium", "invalid_value", f"user_score = 'tbd' in {tbd} rows (text in a numeric column)", tbd,
          "-> NaN; user score is post-release anyway (not a feature)")
    future = int((df["year_of_release"] > 2016).sum())
    issue("high", "range", f"{future} release year(s) after the dataset date (Dec 2016)", future, "drop row")
    dup = int(df.duplicated(["name", "platform"]).sum())
    issue("medium", "duplicates", f"{dup} duplicated (name, platform) rows", dup, "keep the row with higher sales")
    rare = df["rating"].value_counts()
    rare = rare[rare < 10]
    issue("low", "rare_category", f"rare ESRB codes {rare.to_dict()}", int(rare.sum()),
          "K-A, EC -> E; AO -> M; RP (pending) -> Unknown")
    out = int((df["global_sales"] > df["global_sales"].quantile(0.99) * 5).sum())
    issue("low", "outliers", f"{out} extreme sales values (Wii Sports etc.)", out, "keep: real hits; sales only "
          "used for business impact (medians)")
    issue("high", "target_leakage", "Sales, user score, critic/user counts are only known after release", None,
          "Excluded from model features")
    return report


def main() -> dict:
    corgis, kaggle, games = db.load_raw("corgis_raw"), db.load_raw("kaggle_raw"), db.load_games()
    c, k = run_quality_checks(corgis.assign(game_id=range(len(corgis)))), run_kaggle_checks(kaggle)
    integ_path = config.REPORTS_DIR / "data_integration.json"
    integration = json.loads(integ_path.read_text(encoding="utf-8")) if integ_path.exists() else {}
    issues = [{"source": "CORGIS", **i} for i in c["issues"]] + [{"source": "Kaggle", **i} for i in k["issues"]]
    issues.append({"source": "merge", "severity": "info", "check": "integration",
                   "detail": f"{integration.get('matched_both')} games matched in both sources (title + platform); "
                             f"critic score mean abs. difference {integration.get('score_mean_abs_diff')}",
                   "affected_rows": integration.get("matched_both"),
                   "decision": "Kaggle score/genre/publisher/developer, CORGIS play time + max players"})
    scored = games[games["metrics_review_score"].notna()]
    neg = scored["metrics_review_score"] < config.NEGATIVE_THRESHOLD
    rep = {"n_rows": int(len(games)), "n_labelled": int(len(scored)), "issues": issues,
           "sources": {"corgis": {k_: v for k_, v in c.items() if k_ != "issues"},
                       "kaggle": {k_: v for k_, v in k.items() if k_ != "issues"}},
           "integration": integration,
           "target": {"threshold": config.NEGATIVE_THRESHOLD, "negative_rate": float(neg.mean()),
                      "n_negative": int(neg.sum())},
           "score_stats": {k_: float(v) for k_, v in scored["metrics_review_score"].describe().items()},
           "generated_at": db.now()}
    out = config.REPORTS_DIR / "data_quality.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=lambda o: o.item()
                              if isinstance(o, np.generic) else str(o)), encoding="utf-8")
    print(f"Data quality: {len(issues)} findings -> {out}")
    for i in issues:
        print(f"  [{i['source']:>6}|{i['severity']:>6}] {i['check']}: {i['detail']}")
    return rep


if __name__ == "__main__":
    main()
