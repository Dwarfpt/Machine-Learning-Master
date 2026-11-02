"""End-to-end checks: data integration, leakage guards, prediction, XAI, ensembles, decision module, agent,
RAG, RPA data refresh, HITL tokens.
Run:  pytest -q   (requires a trained model: python -m nrg.train)"""
import shutil

import pandas as pd
import pytest

from nrg import actions, automation, config, db
from nrg.features import FEATURE_COLS, build_features, guess_is_sequel, history_from_games

SPEC = {"title": "Test Racer 2", "console": "Nintendo Wii", "year": 2009, "rating": "E", "genres": "Racing",
        "publisher": "THQ", "developer": "", "max_players": 1, "main_story_hours": 4.0, "n_platforms": 1,
        "is_sequel": True}


@pytest.fixture(scope="session")
def predictor(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db") / "games.db"
    shutil.copy(config.DB_PATH, tmp)
    config.DB_PATH = tmp  # never touch the real DB
    config.OUTBOX_PATH = tmp.with_name("outbox.jsonl")
    config.N8N_WEBHOOK_URL = config.N8N_DECISION_WEBHOOK_URL = ""
    from nrg.predict import Predictor, get_predictor
    get_predictor.cache_clear()
    return get_predictor()


def test_no_post_release_columns_in_features():
    leaky = ("sales", "used_price", "polled", "review_score")
    assert not [c for c in FEATURE_COLS if any(k in c for k in leaky)]


def test_publisher_prior_uses_only_past_years():
    hist = pd.DataFrame({"publisher_main": ["A", "A"], "developer_main": ["D", "D"], "year": [2005, 2006],
                         "score": [10, 90], "negative": [1, 0]})
    spec = pd.DataFrame([{**SPEC, "publisher": "A", "year": 2006}])
    f = build_features(spec, hist).iloc[0]
    assert f["pub_prior_n_games"] == 1  # the 2006 game itself is not visible
    assert f["pub_prior_mean_score"] < 70


def test_sequel_heuristic():
    assert guess_is_sequel("Madden NFL 08") and guess_is_sequel("Final Fantasy XII") and not guess_is_sequel("Zoo Keeper")


def test_predict_and_explain(predictor):
    r = predictor.predict(SPEC)
    assert 0 <= r["proba_negative"] <= 1
    assert r["risk_level"] in {"low", "medium", "high"}
    assert r["explanation"]["factors"] and {"feature", "shap", "value"} <= set(r["explanation"]["factors"][0])
    assert db.get_prediction(r["prediction_id"])["title"] == SPEC["title"]


def test_measures_and_approval_rule(predictor):
    ev = actions.evaluate_measures(predictor, SPEC)
    codes = {m["code"] for m in ev["measures"]}
    assert "EXTENDED_QA" in codes and "ADD_MULTIPLAYER" in codes
    assert all(m["requires_approval"] == (m["cost_k"] > config.APPROVAL_COST_THRESHOLD_K or
                                          m["code"] == "DELAY_RELEASE_POLISH") for m in ev["measures"])
    assert ev["p_after_plan"] <= ev["p_before"] + 1e-9


def test_rule_based_agent_creates_actions(predictor):
    from nrg.agent import run_agent
    pid = predictor.predict(SPEC)["prediction_id"]
    out = run_agent(pid, "en", force_rule_based=True)
    tools = [s["tool"] for s in out["trace"]]
    assert {"get_prediction", "explain_prediction", "evaluate_preventive_measures", "submit_action_plan"} <= set(tools)
    assert out["created_actions"]
    for a in out["created_actions"]:
        expected = "pending_approval" if actions.requires_approval(actions.BY_CODE[a["code"]]) else "approved"
        assert db.get_action(a["action_id"])["status"] == expected


def test_invalid_codes_are_ignored(predictor):
    from nrg.agent import submit_action_plan
    pid = predictor.predict(SPEC)["prediction_id"]
    res = submit_action_plan(pid, ["EXTENDED_QA", "DELETE_DATABASE"], "test")
    assert res["ignored_invalid_codes"] == ["DELETE_DATABASE"]
    assert "error" in submit_action_plan(pid, ["A", "B"], "placeholder codes")  # nothing created, LLM must retry


def test_approval_token():
    tok = automation.approval_token(42)
    assert automation.verify_token(42, tok) and not automation.verify_token(43, tok)


def test_history_target_definition(predictor):
    h = history_from_games(predictor.games)
    assert ((h["score"] < config.NEGATIVE_THRESHOLD) == h["negative"].astype(bool)).all()


def test_low_risk_policy_forces_monitoring(predictor):
    from nrg.agent import submit_action_plan
    g = predictor.games
    gid = int(g[(g["title"] == "Super Mario Galaxy") & (g["release_console"] == "Nintendo Wii")]["game_id"].iloc[0])
    r = predictor.predict(predictor.spec_for_game(gid), game_id=gid)  # score 97: expected low risk
    if r["risk_level"] != "low":
        pytest.skip("example is not low-risk for the current model")
    res = submit_action_plan(r["prediction_id"], ["EXTENDED_QA"], "test")
    assert [a["code"] for a in res["created_actions"]] == ["MONITOR_ONLY"] and "policy_override" in res


def test_cjk_is_stripped_from_llm_text():
    from nrg.agent import _strip_cjk
    assert _strip_cjk("Риски, связанные с 负面评价。 Менеджеру") == "Риски, связанные с. Менеджеру"
    assert _strip_cjk("Plan: QA，PR") == "Plan: QA,PR"
    assert _strip_cjk("Обычный текст, без изменений!") == "Обычный текст, без изменений!"


def test_two_sources_are_merged():
    import json
    rep = json.loads((config.REPORTS_DIR / "data_integration.json").read_text(encoding="utf-8"))
    assert rep["matched_both"] > 1000 and rep["with_critic_score"] > 8000 and rep["platforms"] > 20
    assert rep["years"][1] <= 2016 and rep["score_mean_abs_diff"] < 1  # same Metacritic score in both sources
    games = db.load_games()
    assert set(games["source"]) >= {"kaggle", "corgis", "both"}


def test_majority_vote_needs_three_of_five():
    import numpy as np
    from sklearn.base import BaseEstimator, ClassifierMixin
    from nrg.ensembles import MajorityVoteClassifier

    class Const(ClassifierMixin, BaseEstimator):
        def __init__(self, p=0.0):
            self.p = p
        def fit(self, X, y):
            return self
        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])

    X, y = np.zeros((4, 1)), np.array([0, 1, 0, 1])
    yes3 = MajorityVoteClassifier([(str(i), Const(p)) for i, p in enumerate([.9, .9, .9, .1, .1])]).fit(X, y)
    yes2 = MajorityVoteClassifier([(str(i), Const(p)) for i, p in enumerate([.9, .9, .1, .1, .1])]).fit(X, y)
    assert yes3.predict(X).tolist() == [1] * 4 and yes2.predict(X).tolist() == [0] * 4
    assert abs(yes3.predict_proba(X)[0, 1] - 0.6) < 1e-9  # share of votes


def test_rag_finds_relevant_policy():
    from nrg import rag
    hits = rag.search("licensed movie tie-in game release delay", k=3)
    assert hits and any("Licensed" in h["title"] for h in hits)


def test_agent_uses_knowledge_base(predictor):
    from nrg.agent import run_agent
    pid = predictor.predict(SPEC)["prediction_id"]
    out = run_agent(pid, "ru", force_rule_based=True)
    kb = [s for s in out["trace"] if s["tool"] == "search_knowledge_base"]
    assert kb and kb[0]["result"]["passages"]
    assert "RAG" in out["answer"]


def test_data_refresh_ingests_and_validates(predictor, tmp_path, monkeypatch):
    from nrg import refresh
    monkeypatch.setattr(refresh, "INCOMING", tmp_path)
    shutil.copy(config.ROOT / "docs" / "examples" / "incoming_example.csv", tmp_path / "new_games.csv")
    out = refresh.run()
    assert out["added"] == 7 and out["invalid"] == 2 and out["upcoming_scored"] == 7
    assert (tmp_path / "processed").exists() and not (tmp_path / "new_games.csv").exists()
    again = shutil.copy(config.ROOT / "docs" / "examples" / "incoming_example.csv", tmp_path / "again.csv")
    assert again and refresh.run()["duplicates"] == 7  # the same games are not added twice
