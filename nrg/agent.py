"""AI agent: analyses the ML result and selects preventive measures.

Tools (the same functions are used by the LLM agent and by the rule-based fallback):
  1. get_prediction               – DB: prediction, risk level, game spec
  2. explain_prediction           – XAI: top SHAP factors
  3. find_similar_games           – DB + model space: nearest released games and their real reviews/sales
  4. get_publisher_track_record   – DB: publisher history
  5. evaluate_preventive_measures – ML what-if simulation + cost/benefit of each measure
  6. submit_action_plan           – writes actions to DB and triggers n8n (costly ones -> human approval)

LLM: local GGUF model (default Qwen2.5-3B-Instruct) via llama-cpp-python, Qwen-native tool calling (see nrg/llm.py).
If the model is not available the deterministic rule-based agent runs the same tools in a fixed order, so the system
always produces a decision.
"""
from __future__ import annotations

import json
import re

from . import actions, automation, config, db, llm
from .i18n import factor_sentence, feature_label, measure_name, t
from .predict import get_predictor

LANG_NAMES = {"ru": "Russian", "ro": "Romanian", "en": "English"}
# CJK ideographs, kana, CJK punctuation ("。", "，") and full-width forms: small Qwen models code-switch into Chinese
_CJK = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]")
_CJK_PUNCT = str.maketrans({"\u3002": ".", "\uff0c": ",", "\u3001": ",", "\uff1a": ":", "\uff1b": ";", "\uff01": "!",
                            "\uff1f": "?", "\uff08": "(", "\uff09": ")"})


def _strip_cjk(text: str) -> str:
    text = _CJK.sub("", text.translate(_CJK_PUNCT))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\s+([.,;:!?)])", r"\1", text).strip()


# ------------------------------------------------------------------ tools
def get_prediction(prediction_id: int) -> dict:
    p = db.get_prediction(int(prediction_id))
    if not p:
        return {"error": f"prediction {prediction_id} not found"}
    pred = get_predictor()
    return {"prediction_id": p["id"], "title": p["title"], "game_spec": p["input"],
            "proba_negative": round(p["proba_negative"], 3), "risk_level": p["risk_level"],
            "decision_threshold": round(pred.threshold, 3), "model": p["model_name"],
            "negative_review_definition": f"critic score < {config.NEGATIVE_THRESHOLD}/100"}


def explain_prediction(prediction_id: int, top_k: int = 6) -> dict:
    p = db.get_prediction(int(prediction_id))
    if not p:
        return {"error": f"prediction {prediction_id} not found"}
    factors = (p["explanation"] or [])[: int(top_k)]
    return {"explanation_method": "SHAP (TreeExplainer), contributions summed per original feature",
            "factors": [{"feature": f["feature"], "label": feature_label(f["feature"], "en"),
                         "value": f["value"], "shap": round(f["shap"], 4), "direction": f["direction"]}
                        for f in factors]}


def find_similar_games(prediction_id: int, k: int = 5) -> dict:
    p = db.get_prediction(int(prediction_id))
    if not p:
        return {"error": f"prediction {prediction_id} not found"}
    games = get_predictor().similar_games(p["input"], int(k))
    neg = sum(g["review_score"] < config.NEGATIVE_THRESHOLD for g in games)
    return {"similar_games": games, "negative_among_similar": f"{neg}/{len(games)}"}


def get_publisher_track_record(prediction_id: int) -> dict:
    p = db.get_prediction(int(prediction_id))
    if not p:
        return {"error": f"prediction {prediction_id} not found"}
    return get_predictor().publisher_track_record(p["input"].get("publisher") or "")


def evaluate_preventive_measures(prediction_id: int, _lang: str = "en") -> dict:
    p = db.get_prediction(int(prediction_id))
    if not p:
        return {"error": f"prediction {prediction_id} not found"}
    risky = [f["feature"] for f in (p["explanation"] or []) if f["shap"] > 0][:5]
    ev = actions.evaluate_measures(get_predictor(), p["input"], p["proba_negative"], risky)
    r = lambda x: round(x, 3)  # noqa: E731
    return {"risk_level": ev["risk_level"], "p_before": r(ev["p_before"]),
            "revenue_at_risk_k_usd": round(ev["revenue_at_risk_k"]),
            "approval_rule": f"cost > {config.APPROVAL_COST_THRESHOLD_K} k$ or release delay -> manager approval",
            "measures": [{"code": m["code"], "name": measure_name(m["code"], _lang), "kind": m["kind"],
                          "cost_k_usd": m["cost_k"], "p_after": r(m["p_after"]),
                          "expected_benefit_k_usd": round(m["expected_benefit_k"]),
                          "net_value_k_usd": round(m["net_value_k"]), "requires_approval": m["requires_approval"],
                          "targets_top_risk_factor": m["targets_top_risk_factor"]} for m in ev["measures"]],
            "recommended_plan": ev["recommended_plan"], "p_after_plan": r(ev["p_after_plan"]),
            "plan_cost_k_usd": ev["plan_cost_k"], "plan_expected_benefit_k_usd": round(ev["plan_expected_benefit_k"])}


def submit_action_plan(prediction_id: int, action_codes: list[str], rationale: str, _source: str = "llm_agent") -> dict:
    p = db.get_prediction(int(prediction_id))
    if not p:
        return {"error": f"prediction {prediction_id} not found"}
    rationale = _strip_cjk(str(rationale or ""))  # stored in the DB, shown to the manager and e-mailed
    if isinstance(action_codes, str):
        action_codes = [c.strip() for c in action_codes.replace(";", ",").split(",")]
    valid = [c for c in dict.fromkeys(action_codes) if c in actions.BY_CODE or c == actions.MONITOR_ONLY]
    rejected = [c for c in action_codes if c not in valid]
    if len(valid) > 1 and actions.MONITOR_ONLY in valid:  # monitoring together with real measures is meaningless
        valid.remove(actions.MONITOR_ONLY)
    if action_codes and not valid:
        return {"error": f"none of the codes {action_codes} exist; nothing was created. Use codes from "
                         f"evaluate_preventive_measures: {[m.code for m in actions.CATALOG] + [actions.MONITOR_ONLY]}"}
    policy_note = None
    if p["risk_level"] == "low" and valid != [actions.MONITOR_ONLY]:  # business policy: low risk -> no spending
        policy_note = f"policy: risk is low, {valid} replaced by MONITOR_ONLY"
        valid = [actions.MONITOR_ONLY]
    ev = {m["code"]: m for m in actions.evaluate_measures(get_predictor(), p["input"], p["proba_negative"])["measures"]}
    created = []
    for code in valid or [actions.MONITOR_ONLY]:
        m = actions.BY_CODE.get(code)
        cost = m.cost_k if m else 0.0
        need = actions.requires_approval(m) if m else False
        aid = db.create_action(p["id"], p["title"], code, rationale, cost,
                               ev.get(code, {}).get("expected_benefit_k", 0.0), need, _source)
        delivery = automation.notify_action_created(aid, {"proba_negative": p["proba_negative"],
                                                          "risk_level": p["risk_level"]})
        created.append({"action_id": aid, "code": code, "status": "pending_approval" if need else "approved",
                        "delivery": delivery["channel"]})
    res = {"created_actions": created, "ignored_invalid_codes": rejected}
    if policy_note:
        res["policy_override"] = policy_note
    return res


TOOLS = {f.__name__: f for f in [get_prediction, explain_prediction, find_similar_games, get_publisher_track_record,
                                 evaluate_preventive_measures, submit_action_plan]}


def _schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


_PID = {"prediction_id": {"type": "integer", "description": "ID of the prediction"}}
TOOL_SCHEMAS = [
    _schema("get_prediction", "Get the ML prediction (probability of a negative review, risk level, game spec).",
            _PID, ["prediction_id"]),
    _schema("explain_prediction", "Get SHAP explanation: which features push the risk up or down.",
            {**_PID, "top_k": {"type": "integer"}}, ["prediction_id"]),
    _schema("find_similar_games", "Find the most similar already released games with their real critic score and sales.",
            {**_PID, "k": {"type": "integer"}}, ["prediction_id"]),
    _schema("get_publisher_track_record", "Get historical review scores of the game's publisher.",
            _PID, ["prediction_id"]),
    _schema("evaluate_preventive_measures", "Simulate every preventive measure through the ML model and compute "
            "cost, expected benefit and net value in k$. Returns a recommended plan.", _PID, ["prediction_id"]),
    _schema("submit_action_plan", "Submit the final action plan. Creates tasks and triggers the n8n automation; "
            "expensive measures are sent to a manager for approval.",
            {**_PID, "action_codes": {"type": "array", "items": {
                "type": "string", "enum": [m.code for m in actions.CATALOG] + [actions.MONITOR_ONLY]},
                "description": "Measure codes from evaluate_preventive_measures (only these exact values)"},
             "rationale": {"type": "string", "description": "Short business justification"}},
            ["prediction_id", "action_codes", "rationale"]),
]

SYSTEM_PROMPT = """You are a game-publishing risk analyst agent. A machine-learning model has estimated the
probability that a video game will receive a NEGATIVE critic review (score < {thr}/100). Your job is to turn
this prediction into a decision.

Procedure (use the tools, never invent numbers):
Step 1 - call these five tools together in ONE answer: get_prediction, explain_prediction, find_similar_games,
         get_publisher_track_record, evaluate_preventive_measures.
Step 2 - call submit_action_plan exactly once.
Step 3 - write the final comment (no more tool calls).

Decision rules:
- action_codes must be real codes such as "EXTENDED_QA" copied from evaluate_preventive_measures (never "A", "B").
- Choose only measure codes returned by evaluate_preventive_measures, prefer positive net value and measures that
  target the top SHAP risk factors. Max 3 measures. If risk is low, submit ["MONITOR_ONLY"].
- You may deviate from the recommended plan, but explain why.
- Never claim an expensive measure is approved: they go to a human manager (Human-in-the-loop).

Final answer (after submit_action_plan): an analyst comment of 3-5 sentences in {lang}: why you chose these
measures, which SHAP risk factors they address, what the manager should watch. Do not repeat tables and do not
invent numbers - the system adds all figures from the tools itself. Write only in {lang}."""


# ------------------------------------------------------------------ LLM agent
def llm_available() -> bool:
    return llm.available()


def _call_tool(name: str, args: dict, prediction_id: int, state: dict) -> dict:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    args = {k: v for k, v in args.items() if not k.startswith("_")}  # internal params are not for the LLM
    if "prediction_id" in fn.__code__.co_varnames:
        args["prediction_id"] = prediction_id  # the agent works on exactly one prediction
    if name == "submit_action_plan" and state["submitted"]:
        return {"error": "plan already submitted"}
    if name == "evaluate_preventive_measures":
        args["_lang"] = state["lang"]
    try:
        res = fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}
    if name == "submit_action_plan" and res.get("created_actions"):
        state["submitted"] = True
    return res


def _run_llm(prediction_id: int, lang: str, max_steps: int = 8) -> tuple[str, list]:
    system = llm.tools_system_prompt(SYSTEM_PROMPT.format(thr=config.NEGATIVE_THRESHOLD, lang=LANG_NAMES[lang]),
                                     TOOL_SCHEMAS)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Analyse prediction_id={prediction_id} and decide what to do. "
                                            f"Write the final report in {LANG_NAMES[lang]}."}]
    trace, state, final = [], {"submitted": False, "lang": lang}, ""
    for _ in range(max_steps):
        text = llm.chat(messages, max_tokens=320 if state["submitted"] else 512)
        calls = llm.parse_tool_calls(text)
        messages.append({"role": "assistant", "content": text})
        if not calls:
            final = text.strip()
            break
        results = []
        for c in calls:
            res = _call_tool(c["name"], c["arguments"], prediction_id, state)
            trace.append({"tool": c["name"], "args": c["arguments"], "result": res})
            results.append((c["name"], res))
        messages.append(llm.tool_responses_message(results))
    else:
        final = "(agent stopped: step limit reached)"
    if lang != "en" and _CJK.search(final):  # small Qwen models sometimes code-switch into Chinese
        messages.append({"role": "user", "content": f"Rewrite your report strictly in {LANG_NAMES[lang]}, "
                                                    "without any Chinese characters. Keep the same content."})
        final = _strip_cjk(llm.chat(messages))
    if not state["submitted"]:  # guardrail: a decision must always be recorded
        ev = evaluate_preventive_measures(prediction_id)
        res = submit_action_plan(prediction_id, ev["recommended_plan"],
                                 "Guardrail: LLM did not submit a plan; decision module plan used.", "llm_guardrail")
        trace.append({"tool": "submit_action_plan", "args": {"guardrail": True}, "result": res})
    return final, trace


# ------------------------------------------------------------------ rule-based agent
def _run_rules(prediction_id: int, lang: str) -> tuple[str, list]:
    trace = []

    def call(name, **kw):
        res = TOOLS[name](**kw)
        trace.append({"tool": name, "args": kw, "result": res})
        return res

    pred = call("get_prediction", prediction_id=prediction_id)
    exp = call("explain_prediction", prediction_id=prediction_id, top_k=6)
    sim = call("find_similar_games", prediction_id=prediction_id, k=5)
    pub = call("get_publisher_track_record", prediction_id=prediction_id)
    ev = call("evaluate_preventive_measures", prediction_id=prediction_id)
    plan = ev["recommended_plan"]
    by_code = {m["code"]: m for m in ev["measures"]}
    rationale = "; ".join(f"{c}: net {by_code[c]['net_value_k_usd']} k$" for c in plan if c in by_code) or \
        "low risk – monitoring only"
    sub = call("submit_action_plan", prediction_id=prediction_id, action_codes=plan, rationale=rationale,
               _source="rule_based_agent")
    return _compose_report(pred, exp, sim, pub, ev, sub, lang), trace


_H = {
    "ru": ("Риск", "Главные причины (SHAP)", "Подтверждения", "Решение", "Ограничения",
           "Похожие вышедшие игры: негативных {neg}.", "Издатель {p}: {n} игр, средняя оценка {m}, доля провалов {r:.0%}.",
           "Издатель без истории в базе.", "Ожидает согласования менеджера", "Запущено автоматически",
           "Модель обучена на ~900 играх 2004–2008 (PR-AUC ≈ 0.39 на тесте против 0.18 у базовой); эффект процессных мер "
           "(QA, плейтест, PR) — экспертное допущение, а не вывод модели.",
           "Вероятность негативного отзыва: {p:.0%} (порог {thr:.0%}), уровень риска: {risk}."),
    "ro": ("Risc", "Cauzele principale (SHAP)", "Dovezi", "Decizie", "Limitări",
           "Jocuri similare lansate: negative {neg}.", "Editor {p}: {n} jocuri, scor mediu {m}, rata eșecurilor {r:.0%}.",
           "Editor fără istoric în bază.", "Așteaptă aprobarea managerului", "Lansat automat",
           "Modelul este antrenat pe ~900 de jocuri din 2004–2008 (PR-AUC ≈ 0.39 vs 0.18 la bază); efectul "
           "măsurilor de proces (QA, playtest, PR) este o ipoteză de expert, nu o ieșire a modelului.",
           "Probabilitatea unei recenzii negative: {p:.0%} (prag {thr:.0%}), nivel de risc: {risk}."),
    "en": ("Risk", "Main reasons (SHAP)", "Evidence", "Decision", "Caveats",
           "Similar released games: {neg} negative.", "Publisher {p}: {n} games, mean score {m}, flop rate {r:.0%}.",
           "Publisher has no history in the DB.", "Waiting for manager approval", "Started automatically",
           "The model is trained on ~900 games from 2004–2008 (PR-AUC ≈ 0.39 vs 0.18 baseline); the effect of "
           "process measures (QA, playtest, PR) is an expert assumption, not a model output.",
           "Probability of a negative review: {p:.0%} (threshold {thr:.0%}), risk level: {risk}."),
}


def _compose_report(pred, exp, sim, pub, ev, sub, lang) -> str:
    h = _H[lang]
    lines = [f"### {pred['title']}", f"**{h[0]}.** " + h[11].format(p=pred["proba_negative"],
             thr=pred["decision_threshold"], risk=t("risk_" + pred["risk_level"], lang))]
    lines.append(f"\n**{h[1]}:**")
    lines += [f"- {factor_sentence(f, lang)}" for f in exp["factors"][:5]]
    lines.append(f"\n**{h[2]}:**")
    lines.append("- " + h[5].format(neg=sim["negative_among_similar"]) + " " +
                 ", ".join(f"{g['title']} ({g['console']}, {g['review_score']})" for g in sim["similar_games"][:3]))
    lines.append("- " + (h[6].format(p=pub["publisher"], n=pub["n_games"], m=pub["mean_score"], r=pub["negative_rate"])
                         if pub.get("n_games") else h[7]))
    # totals for the measures that were ACTUALLY created (the LLM may deviate from the recommended plan)
    codes = [a["code"] for a in sub["created_actions"] if a["code"] in actions.BY_CODE]
    p1 = actions.plan_probability(get_predictor(), pred["game_spec"], ev["p_before"], codes)
    lines.append(f"\n**{h[3]}:** " + t("plan_summary", lang, p0=ev["p_before"], p1=p1,
                                       cost=sum(actions.BY_CODE[c].cost_k for c in codes),
                                       ben=(ev["p_before"] - p1) * ev["revenue_at_risk_k_usd"]))
    for a in sub["created_actions"]:
        status = h[8] if a["status"] == "pending_approval" else h[9]
        lines.append(f"- **{measure_name(a['code'], lang)}** (#{a['action_id']}) — {status}")
    lines.append(f"\n**{h[4]}:** {h[10]}")
    return "\n".join(lines)


_COMMENT = {"ru": "Комментарий AI-агента (LLM)", "ro": "Comentariul agentului AI (LLM)",
            "en": "AI agent comment (LLM)"}
_GUARDRAIL_NOTE = {
    "ru": "LLM не передала корректный план — сработал guardrail: план зафиксирован модулем решений.",
    "ro": "LLM nu a transmis un plan valid — guardrail activat: planul a fost fixat de modulul de decizie.",
    "en": "The LLM did not submit a valid plan — guardrail triggered: the decision module's plan was recorded."}


# ------------------------------------------------------------------ entry point
def run_agent(prediction_id: int, lang: str = "ru", force_rule_based: bool = False) -> dict:
    lang = lang if lang in LANG_NAMES else "ru"
    mode = "rule_based"
    if not force_rule_based and llm_available():
        try:
            answer, trace = _run_llm(prediction_id, lang)
            mode = "llm"
        except Exception as e:  # LLM failure must not break the business flow
            answer, trace = _run_rules(prediction_id, lang)
            trace.insert(0, {"tool": "llm_error", "args": {}, "result": str(e)})
    else:
        answer, trace = _run_rules(prediction_id, lang)
    created = [a for step in trace if step["tool"] == "submit_action_plan"
               for a in step["result"].get("created_actions", [])]
    if mode == "llm":  # every figure comes from the tools / DB; the LLM contributes the decision and a comment
        pred = get_prediction(prediction_id)
        facts = _compose_report(pred, explain_prediction(prediction_id), find_similar_games(prediction_id),
                                get_publisher_track_record(prediction_id),
                                evaluate_preventive_measures(prediction_id, lang), {"created_actions": created}, lang)
        answer = f"{facts}\n\n**{_COMMENT[lang]}:**\n\n{answer.strip()}"
        if any(step["args"].get("guardrail") for step in trace):
            answer += "\n\n⚠️ " + _GUARDRAIL_NOTE[lang]
    run_id = db.save_agent_run(prediction_id, mode, lang, trace, answer)
    return {"run_id": run_id, "mode": mode, "model": config.LLM_NAME if mode == "llm" else None,
            "answer": answer, "trace": trace, "created_actions": created}
