"""Offline evaluation of the AI agent against the decision module (the "oracle" plan).

Runs the LLM agent on a stratified sample of test-year games (low / medium / high risk) in a sandbox: a temporary
copy of the DB, no n8n (events go to a temporary outbox), so nothing reaches the real action log or mailboxes.

    python -m nrg.eval_agent --tag 3b --per-level 10          # LLM server must be up: LLM_BASE_URL=http://localhost:8081
    -> reports/agent_eval_3b.json  (+ printed summary)

Metrics
    llm_success       share of runs finished in LLM mode (no exception -> no fallback to the rule-based agent)
    guardrail_rate    LLM did not submit a plan, the decision module plan was recorded instead
    policy_override   LLM proposed spending on a low-risk game, the business policy replaced it by MONITOR_ONLY
    exact_match       recorded plan == decision-module plan (as sets)
    jaccard           mean set overlap between recorded and decision-module plan
    net_value_ratio   net value (benefit - cost, k$) of the recorded plan / of the decision-module plan (high+medium)
    cjk_rate          final comment contained Chinese characters (small Qwen models code-switch)
    latency_s         wall time of one agent run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sandbox() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="nrg_eval_"))
    src = ROOT / "data" / "games.db"
    if src.exists():
        shutil.copy(src, tmp / "games.db")
    os.environ.update(NRG_DB_PATH=str(tmp / "games.db"), NRG_OUTBOX_PATH=str(tmp / "outbox.jsonl"),
                      N8N_WEBHOOK_URL="", N8N_DECISION_WEBHOOK_URL="")
    os.environ.setdefault("LLM_BASE_URL", "http://localhost:8081")
    return tmp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="llm", help="name of the run, e.g. 3b / 7b")
    ap.add_argument("--per-level", type=int, default=10, help="games per risk level")
    ap.add_argument("--lang", default="ru")
    args = ap.parse_args(argv)

    tmp = _sandbox()
    from . import actions, config, db  # noqa: E402  (imported after the sandbox env is set)
    from .agent import run_agent
    from .agent import llm_available
    from .predict import get_predictor

    if not llm_available():
        sys.exit(f"LLM server is not reachable at {config.LLM_BASE_URL}")
    p = get_predictor()
    test_ids = p.specs.loc[p.specs["year"] == config.TEST_YEAR, "game_id"]
    specs = {int(gid): p.spec_for_game(int(gid)) for gid in test_ids}
    proba = dict(zip(specs, p.predict_proba(list(specs.values()))))
    by_level = {"low": [], "medium": [], "high": []}
    for gid, pr in sorted(proba.items(), key=lambda kv: kv[0]):
        by_level[p.risk_level(float(pr))].append(gid)
    sample = [(lvl, gid) for lvl, ids in by_level.items() for gid in ids[::max(1, len(ids) // args.per_level)][:args.per_level]]

    from .agent import _CJK as cjk
    rows = []
    for i, (lvl, gid) in enumerate(sample, 1):
        pred = p.predict(specs[gid], game_id=gid)
        ev = actions.evaluate_measures(p, pred_spec := specs[gid], pred["proba_negative"])
        oracle = set(ev["recommended_plan"])
        t0 = time.time()
        res = run_agent(pred["prediction_id"], args.lang)
        dt = time.time() - t0
        trace = res["trace"]
        subs = [s for s in trace if s["tool"] == "submit_action_plan"]
        chosen = {a["code"] for a in res["created_actions"]}
        proposed = set()
        for s in subs:
            codes = s["args"].get("action_codes") or []
            proposed |= set([codes] if isinstance(codes, str) else codes)

        def net(codes):
            codes = [c for c in codes if c in actions.BY_CODE]
            p1 = actions.plan_probability(p, pred_spec, ev["p_before"], codes)
            return (ev["p_before"] - p1) * ev["revenue_at_risk_k"] - sum(actions.BY_CODE[c].cost_k for c in codes)

        rows.append({
            "game_id": gid, "title": pred.get("title"), "risk_level": lvl, "p": round(float(pred["proba_negative"]), 3),
            "mode": res["mode"], "llm_error": next((s["result"] for s in trace if s["tool"] == "llm_error"), None),
            "guardrail": any(s["args"].get("guardrail") for s in subs),
            "policy_override": any("policy_override" in s["result"] for s in subs),
            "proposed": sorted(proposed), "recorded": sorted(chosen), "oracle": sorted(oracle),
            "exact_match": chosen == oracle, "jaccard": len(chosen & oracle) / len(chosen | oracle) if chosen | oracle else 1.0,
            "net_recorded_k": round(net(chosen), 1), "net_oracle_k": round(net(oracle), 1),
            "tools_called": [s["tool"] for s in trace], "cjk": bool(cjk.search(res["answer"])), "latency_s": round(dt, 1),
        })
        r = rows[-1]
        print(f"[{i:2}/{len(sample)}] {lvl:6} p={r['p']:.2f} {str(r['title'])[:28]:28} {dt:5.1f}s "
              f"llm={r['proposed']} -> {r['recorded']} | oracle={r['oracle']}"
              + (" GUARDRAIL" if r["guardrail"] else "") + (" POLICY" if r["policy_override"] else ""))

    n = len(rows)
    spend = [r for r in rows if r["risk_level"] != "low"]
    summary = {
        "tag": args.tag, "model": config.LLM_NAME, "n_games": n, "per_level": {k: sum(r["risk_level"] == k for r in rows) for k in by_level},
        "llm_success": sum(r["mode"] == "llm" for r in rows) / n,
        "guardrail_rate": sum(r["guardrail"] for r in rows) / n,
        "policy_override_rate": sum(r["policy_override"] for r in rows) / n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "exact_match_by_level": {k: round(statistics.mean([r["exact_match"] for r in rows if r["risk_level"] == k]), 3)
                                 for k in by_level if any(r["risk_level"] == k for r in rows)},
        "mean_jaccard": statistics.mean(r["jaccard"] for r in rows),
        "net_value_ratio_med_high": (sum(r["net_recorded_k"] for r in spend) / sum(r["net_oracle_k"] for r in spend)
                                     if spend and sum(r["net_oracle_k"] for r in spend) else None),
        "cjk_rate": sum(r["cjk"] for r in rows) / n,
        "latency_s_mean": statistics.mean(r["latency_s"] for r in rows),
        "latency_s_p90": sorted(r["latency_s"] for r in rows)[int(0.9 * (n - 1))],
    }
    out = config.REPORTS_DIR / f"agent_eval_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
