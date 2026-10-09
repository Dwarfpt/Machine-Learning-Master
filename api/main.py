"""REST API — used by n8n (and any external system): prediction, explanation, agent, HITL approvals.

Run:  uvicorn api.main:app --reload --port 8000     (docs: http://localhost:8000/docs)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from nrg import actions, automation, db  # noqa: E402
from nrg.agent import llm_available, run_agent  # noqa: E402
from nrg.drift import drift_report  # noqa: E402
from nrg.i18n import measure_name  # noqa: E402
from nrg.predict import get_predictor  # noqa: E402

app = FastAPI(title="Negative Review Guard API", version="1.0",
              description="Prediction of negative critic reviews for video games + automated preventive measures")


class GameSpec(BaseModel):
    title: str = Field(examples=["Galactic Racer 2"])
    console: str = Field(examples=["Nintendo Wii"], description="Nintendo DS | Sony PSP | Nintendo Wii | X360 | PlayStation 3")
    year: int = Field(examples=[2009])
    rating: str = Field(examples=["E"], description="E | T | M")
    genres: str = Field(examples=["Racing / Driving"], description="comma separated")
    publisher: str = Field("", examples=["THQ"])
    max_players: int = Field(1, ge=1, le=64)
    main_story_hours: float = Field(gt=0, examples=[6.0])
    n_platforms: int = Field(1, ge=1, le=5)
    is_sequel: bool | None = None


class AgentRequest(BaseModel):
    prediction_id: int
    language: str = "ru"
    force_rule_based: bool = False


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "model": p.model_name, "threshold": p.threshold, "llm_available": llm_available()}


@app.post("/predict")
def predict(spec: GameSpec):
    return get_predictor().predict(spec.model_dump())


@app.post("/predict/game/{game_id}")
def predict_existing(game_id: int):
    p = get_predictor()
    try:
        return p.predict(p.spec_for_game(game_id), game_id=game_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/predictions/{prediction_id}")
def prediction(prediction_id: int):
    p = db.get_prediction(prediction_id)
    if not p:
        raise HTTPException(404, "not found")
    return p


@app.get("/predictions/{prediction_id}/measures")
def measures(prediction_id: int):
    p = db.get_prediction(prediction_id)
    if not p:
        raise HTTPException(404, "not found")
    risky = [f["feature"] for f in (p["explanation"] or []) if f["shap"] > 0][:5]
    return actions.evaluate_measures(get_predictor(), p["input"], p["proba_negative"], risky)


@app.post("/agent/run")
def agent(req: AgentRequest):
    if not db.get_prediction(req.prediction_id):
        raise HTTPException(404, "prediction not found")
    return run_agent(req.prediction_id, req.language, req.force_rule_based)


@app.get("/actions")
def list_actions(status: str | None = None):
    return db.list_actions(status).to_dict(orient="records")


_PAGE = """<html><body style="font-family:sans-serif;max-width:560px;margin:60px auto">
<h2>{icon} Action #{id}: {name}</h2><p>Game: <b>{title}</b></p><p>Status: <b>{status}</b></p>
<p style="color:#666">Negative Review Guard · Human-in-the-loop</p></body></html>"""


@app.get("/actions/{action_id}/decision", response_class=HTMLResponse)
def decide(action_id: int, decision: str = Query(pattern="^(approved|rejected)$"), token: str = "",
           by: str = "manager (email link)"):
    """Human-in-the-loop endpoint: the approve / reject links in the n8n e-mail point here."""
    a = db.get_action(action_id)
    if not a:
        raise HTTPException(404, "action not found")
    if not automation.verify_token(action_id, token):
        raise HTTPException(403, "invalid token")
    if a["status"] == "pending_approval":
        db.set_action_status(action_id, decision, by)
        automation.notify_decision(action_id)
    a = db.get_action(action_id)
    icon = "✅" if a["status"] in ("approved", "executed") else "❌"
    return _PAGE.format(icon=icon, id=action_id, name=measure_name(a["action_code"], "en"), title=a["title"],
                        status=a["status"])


@app.post("/actions/{action_id}/execute")
def execute(action_id: int, token: str = ""):
    """Called by n8n after the task was created in the external system (Sheets / Trello / e-mail)."""
    a = db.get_action(action_id)
    if not a:
        raise HTTPException(404, "action not found")
    if not automation.verify_token(action_id, token):
        raise HTTPException(403, "invalid token")
    if a["status"] != "approved":
        raise HTTPException(409, f"action is {a['status']}, only approved actions can be executed")
    db.set_action_status(action_id, "executed", "n8n")
    return db.get_action(action_id)


@app.get("/monitoring/drift")
def drift():
    return drift_report()
