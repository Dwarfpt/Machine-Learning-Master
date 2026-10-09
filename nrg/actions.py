"""Decision module: catalog of preventive measures and their cost/benefit evaluation.

Two kinds of measures:
* "simulated"  — change an actionable feature of the game spec and re-run the ML model (counterfactual what-if).
                 The effect is estimated by the model itself.
* "assumption" — process measures the dataset cannot observe (QA, playtests, PR). Their effect is an explicit,
                 configurable relative risk reduction (documented assumption, not a model output).

Expected benefit (k$) = (p_before - p_after) * revenue_at_risk_k ;  net = benefit - cost.
Measures above APPROVAL_COST_THRESHOLD_K (or flagged) need human approval (Human-in-the-loop).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import config


@dataclass
class Measure:
    code: str
    cost_k: float
    kind: str  # simulated | assumption
    addresses: list[str]  # features (SHAP groups) this measure acts on
    modify: Callable[[dict], dict | None] | None = None  # returns modified spec or None if not applicable
    relative_risk_reduction: float = 0.0
    always_approval: bool = False
    tags: list[str] = field(default_factory=list)


def _more_content(s):
    h = s.get("main_story_hours") or 0
    return {**s, "main_story_hours": round(h * 1.5, 1)} if h else None


def _add_multiplayer(s):
    return {**s, "max_players": 4} if (s.get("max_players") or 1) < 4 else None


def _port(s):
    n = s.get("n_platforms") or 1
    return {**s, "n_platforms": n + 1} if n < 5 else None


CATALOG: list[Measure] = [
    Measure("EXTEND_CONTENT", 180, "simulated", ["main_story_hours"], modify=_more_content),
    Measure("ADD_MULTIPLAYER", 120, "simulated", ["max_players", "is_multiplayer"], modify=_add_multiplayer),
    Measure("MULTIPLATFORM_PORT", 250, "simulated", ["n_platforms", "is_multiplatform", "console"], modify=_port),
    Measure("DELAY_RELEASE_POLISH", 300, "assumption", ["release_year", "is_sequel", "pub_prior_mean_score"],
            relative_risk_reduction=0.25, always_approval=True),
    Measure("EXTENDED_QA", 60, "assumption", ["pub_prior_mean_score", "pub_prior_neg_rate", "publisher_main"],
            relative_risk_reduction=0.15),
    Measure("PLAYTEST_FOCUS_GROUP", 25, "assumption", ["genre_*", "rating", "main_story_hours"],
            relative_risk_reduction=0.10),
    Measure("CRITIC_PREVIEW_PR", 15, "assumption", ["is_sequel", "has_subtitle", "publisher_unknown"],
            relative_risk_reduction=0.05),
]
MONITOR_ONLY = "MONITOR_ONLY"
BY_CODE = {m.code: m for m in CATALOG}


def requires_approval(m: Measure) -> bool:
    return m.always_approval or m.cost_k > config.APPROVAL_COST_THRESHOLD_K


def catalog_as_dicts() -> list[dict]:
    return [{"code": m.code, "cost_k": m.cost_k, "kind": m.kind, "addresses": m.addresses,
             "relative_risk_reduction": m.relative_risk_reduction if m.kind == "assumption" else None,
             "requires_approval": requires_approval(m)} for m in CATALOG]


def evaluate_measures(predictor, spec: dict, p_before: float | None = None,
                      top_factors: list[str] | None = None) -> dict:
    """Score every measure for one game. Returns the ranking and a recommended plan."""
    if p_before is None:
        p_before = float(predictor.predict_proba([spec])[0])
    risk_k = predictor.revenue_at_risk_k(spec["console"])
    risky = set(top_factors or [])
    rows = []
    sim = [(m, m.modify(spec)) for m in CATALOG if m.kind == "simulated"]
    sim = [(m, s) for m, s in sim if s is not None]
    sim_p = predictor.predict_proba([s for _, s in sim]) if sim else []
    sim_p = {m.code: float(p) for (m, _), p in zip(sim, sim_p)}

    for m in CATALOG:
        if m.kind == "simulated":
            if m.code not in sim_p:
                continue
            p_after = sim_p[m.code]
        else:
            p_after = p_before * (1 - m.relative_risk_reduction)
        delta = p_before - p_after
        benefit = delta * risk_k
        targets = any(a in risky or (a.endswith("*") and any(r.startswith(a[:-1]) for r in risky))
                      for a in m.addresses)
        rows.append({"code": m.code, "kind": m.kind, "cost_k": m.cost_k, "p_before": p_before, "p_after": p_after,
                     "delta_p": delta, "expected_benefit_k": benefit, "net_value_k": benefit - m.cost_k,
                     "roi": benefit / m.cost_k if m.cost_k else 0.0, "targets_top_risk_factor": targets,
                     "requires_approval": requires_approval(m)})
    rows.sort(key=lambda r: r["net_value_k"], reverse=True)

    risk_level = predictor.risk_level(p_before)
    if risk_level == "low":
        plan = []
    else:
        plan = [r["code"] for r in rows if r["net_value_k"] > 0][:3]
    p_plan = plan_probability(predictor, spec, p_before, plan)
    return {"p_before": p_before, "risk_level": risk_level, "revenue_at_risk_k": risk_k, "measures": rows,
            "recommended_plan": plan or [MONITOR_ONLY], "p_after_plan": p_plan,
            "plan_cost_k": sum(BY_CODE[c].cost_k for c in plan),
            "plan_expected_benefit_k": (p_before - p_plan) * risk_k}


def plan_probability(predictor, spec: dict, p_before: float, codes: list[str]) -> float:
    """Combine measures: apply all spec changes at once through the model, then process-measure reductions."""
    s = dict(spec)
    changed = False
    for c in codes:
        m = BY_CODE[c]
        if m.kind == "simulated":
            new = m.modify(s)
            if new:
                s, changed = new, True
    p = float(predictor.predict_proba([s])[0]) if changed else p_before
    for c in codes:
        m = BY_CODE[c]
        if m.kind == "assumption":
            p *= 1 - m.relative_risk_reduction
    return p
