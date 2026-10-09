"""End-to-end checks: leakage guards, prediction, XAI, decision module, agent, HITL tokens.
Run:  pytest -q   (requires a trained model: python -m nrg.train)"""
import shutil

import pandas as pd
import pytest

from nrg import actions, automation, config, db
from nrg.features import FEATURE_COLS, build_features, guess_is_sequel, history_from_games

SPEC = {"title": "Test Racer 2", "console": "Nintendo Wii", "year": 2009, "rating": "E", "genres": "Racing / Driving",
        "publisher": "THQ", "max_players": 1, "main_story_hours": 4.0, "n_platforms": 1, "is_sequel": True}


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
    hist = pd.DataFrame({"publisher_main": ["A", "A"], "year": [2005, 2006], "score": [10, 90], "negative": [1, 0]})
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
    r = predictor.predict(predictor.spec_for_game(1), game_id=1)  # Super Mario 64 DS: low risk
    if r["risk_level"] != "low":
        pytest.skip("example is not low-risk for the current model")
    res = submit_action_plan(r["prediction_id"], ["EXTENDED_QA"], "test")
    assert [a["code"] for a in res["created_actions"]] == ["MONITOR_ONLY"] and "policy_override" in res
