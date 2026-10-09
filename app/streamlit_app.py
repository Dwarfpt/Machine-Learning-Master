"""Web application (Streamlit): dashboard, model comparison + XAI, prediction form, AI agent, HITL approvals,
monitoring. Three languages: RU / RO / EN.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from nrg import actions, automation, config, db  # noqa: E402
from nrg.features import CONSOLES, GENRES, RATINGS, split_list  # noqa: E402
from nrg.i18n import LANGS, factor_sentence, feature_label, format_value, measure_desc, measure_name, t  # noqa: E402

# Palette (dataviz reference palette): categorical order, diverging blue<->red, reserved status colors
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RISK_UP, RISK_DOWN = "#d03b3b", "#2a78d6"
STATUS = {"low": ("#0ca30c", "🟢"), "medium": ("#fab219", "🟠"), "high": ("#d03b3b", "🔴")}

st.set_page_config(page_title="Negative Review Guard", page_icon="🎮", layout="wide")

lang = st.sidebar.selectbox("🌐 Language / Язык / Limba", list(LANGS), format_func=LANGS.get, index=0)
st.sidebar.title("🎮 Negative Review Guard")
st.sidebar.caption(t("app_subtitle", lang))
PAGES = ["page_dashboard", "page_models", "page_predict", "page_approvals", "page_monitoring"]
page = st.sidebar.radio("nav", PAGES, format_func=lambda k: t(k, lang), label_visibility="collapsed")


@st.cache_resource
def predictor():
    from nrg.predict import Predictor
    return Predictor()


@st.cache_data
def games() -> pd.DataFrame:
    if not config.DB_PATH.exists():
        db.init_db()
    return db.load_games()


def load_json(name):
    p = config.ARTIFACTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def model_ready() -> bool:
    if not (config.ARTIFACTS_DIR / "model.joblib").exists():
        st.error(t("no_model", lang))
        return False
    return True


def md(text: str) -> None:
    """st.markdown treats $...$ as LaTeX — escape currency signs (k$)."""
    st.markdown(text.replace("$", "\\$"))


def risk_badge(level: str) -> str:
    color, icon = STATUS[level]
    return f"{icon} <span style='color:{color};font-weight:700'>{t('risk_' + level, lang).upper()}</span>"


def shap_chart(factors: list[dict], top: int = 10):
    df = pd.DataFrame(factors[:top])[::-1]
    df["label"] = [f"{feature_label(f, lang)} = {format_value(v)}" for f, v in zip(df.feature, df.value)]
    fig = go.Figure(go.Bar(x=df["shap"], y=df["label"], orientation="h",
                           marker_color=[RISK_UP if v > 0 else RISK_DOWN for v in df["shap"]],
                           hovertemplate="%{y}<br>SHAP %{x:+.4f}<extra></extra>"))
    fig.update_layout(height=40 + 32 * len(df), margin=dict(l=10, r=10, t=10, b=10), xaxis_title="SHAP",
                      bargap=0.35)
    fig.add_vline(x=0, line_color="#888", line_width=1)
    return fig


# =============================================================== DASHBOARD
def page_dashboard():
    st.title(t("app_title", lang))
    df = games().copy()
    thr = config.NEGATIVE_THRESHOLD
    df["negative"] = df["metrics_review_score"] < thr
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("kpi_games", lang), len(df))
    c2.metric(t("kpi_negative", lang, thr=thr), f"{df['negative'].mean():.1%}")
    c3.metric(t("kpi_avg_score", lang), f"{df['metrics_review_score'].mean():.1f}")
    med = df.groupby("negative")["metrics_sales"].median()
    c4.metric(t("kpi_sales_gap", lang), f"{med[False]:.2f}M vs {med[True]:.2f}M")

    left, right = st.columns(2)
    with left:
        st.subheader(t("score_dist", lang))
        fig = px.histogram(df, x="metrics_review_score", nbins=40, color_discrete_sequence=[SERIES[0]])
        fig.add_vline(x=thr, line_dash="dash", line_color=RISK_UP, annotation_text=f"< {thr} = negative")
        fig.update_layout(height=330, margin=dict(t=10, b=10), bargap=0.08, yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader(t("neg_by", lang))
        dim = st.selectbox(" ", ["release_console", "genre", "release_year", "release_rating", "publisher"],
                           format_func=lambda d: {"release_console": t("console", lang), "genre": t("genres", lang),
                                                  "release_year": t("year", lang), "release_rating": t("rating", lang),
                                                  "publisher": t("publisher", lang)}[d], label_visibility="collapsed")
        d = df.copy()
        if dim == "genre":
            d["genre"] = d["metadata_genres"].map(split_list)
            d = d.explode("genre")
        if dim == "publisher":
            d["publisher"] = d["metadata_publishers"].map(lambda p: (split_list(p) or ["Unknown"])[0])
            top = d["publisher"].value_counts().head(12).index
            d = d[d["publisher"].isin(top)]
        agg = d.groupby(dim).agg(rate=("negative", "mean"), n=("negative", "size")).reset_index()
        agg = agg.sort_values("rate")
        agg[dim] = agg[dim].astype(str)
        fig = px.bar(agg, x="rate", y=dim, orientation="h", hover_data={"n": True, "rate": ":.1%"},
                     color_discrete_sequence=[SERIES[0]])
        fig.add_vline(x=df["negative"].mean(), line_dash="dot", line_color="#888")
        fig.update_layout(height=330, margin=dict(t=10, b=10), xaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(t("sales_vs_score", lang))
    d = df.assign(outcome=df["negative"].map({True: f"< {thr}", False: f"≥ {thr}"}))
    fig = px.scatter(d, x="metrics_review_score", y="metrics_sales", color="outcome", hover_name="title",
                     hover_data=["release_console", "release_year"], log_y=True, opacity=0.7,
                     color_discrete_map={f"< {thr}": SERIES[1], f"≥ {thr}": SERIES[0]})
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_layout(height=380, margin=dict(t=10, b=10), xaxis_title="Review score", yaxis_title="Sales, M units")
    st.plotly_chart(fig, use_container_width=True)

    rep = json.loads((config.REPORTS_DIR / "data_quality.json").read_text(encoding="utf-8")) \
        if (config.REPORTS_DIR / "data_quality.json").exists() else None
    if rep:
        with st.expander(f"🧹 {t('data_quality', lang)} — {len(rep['issues'])} findings", expanded=False):
            st.dataframe(pd.DataFrame(rep["issues"]), use_container_width=True, hide_index=True)


# =============================================================== MODELS
def page_models():
    st.title(t("page_models", lang))
    if not model_ready():
        return
    comp = pd.DataFrame(load_json("model_comparison.json"))
    p = predictor()
    st.success(f"**{t('selected_model', lang)}: {p.model_name}** · threshold (calibrated) = {p.threshold:.2f}")
    st.caption(t("selection_rule", lang))
    cols = ["model", "cv_pr_auc", "cv_roc_auc", "test_pr_auc", "test_roc_auc", "test_f1", "test_recall",
            "test_precision", "test_brier"]
    st.subheader(t("model_comparison", lang))
    st.dataframe(comp[cols].style.format({c: "{:.3f}" for c in cols[1:]}).highlight_max(
        subset=["cv_pr_auc", "test_pr_auc", "test_roc_auc"], color="#cde2fb"), use_container_width=True, hide_index=True)
    cal = p.art.get("calibrated_test_metrics")
    if cal:
        st.caption("Calibrated (Platt) production model on test 2008: " +
                   " · ".join(f"{k.replace('test_', '')} {v:.3f}" for k, v in cal.items()))

    curves = load_json("curves.json")
    c1, c2 = st.columns(2)
    roc, pr = go.Figure(), go.Figure()
    for i, (name, cv) in enumerate(curves.items()):
        width = 3 if name == p.model_name else 2
        roc.add_trace(go.Scatter(x=cv["roc"]["fpr"], y=cv["roc"]["tpr"], name=name, mode="lines",
                                 line=dict(color=SERIES[i], width=width)))
        pr.add_trace(go.Scatter(x=cv["pr"]["recall"], y=cv["pr"]["precision"], name=name, mode="lines",
                                line=dict(color=SERIES[i], width=width)))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#aaa", dash="dot"), showlegend=False))
    roc.update_layout(title="ROC (test 2008)", xaxis_title="FPR", yaxis_title="TPR", height=380,
                      hovermode="x unified")
    pr.update_layout(title="Precision–Recall (test 2008)", xaxis_title="Recall", yaxis_title="Precision",
                     height=380, hovermode="x unified")
    c1.plotly_chart(roc, use_container_width=True)
    c2.plotly_chart(pr, use_container_width=True)

    st.subheader(t("global_importance", lang))
    imp = pd.DataFrame(load_json("shap_global.json")).head(15)[::-1]
    imp["label"] = imp["feature"].map(lambda f: feature_label(f, lang))
    fig = px.bar(imp, x="mean_abs_shap", y="label", orientation="h", color_discrete_sequence=[SERIES[0]])
    fig.update_layout(height=480, margin=dict(t=10, b=10), xaxis_title="mean |SHAP|", yaxis_title="", bargap=0.3)
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(fig, use_container_width=True)
    bees = config.REPORTS_DIR / "shap_beeswarm.png"
    if bees.exists():
        c2.image(str(bees), caption="SHAP beeswarm (test 2008)")


# =============================================================== PREDICT
def game_form(default: dict | None = None) -> dict | None:
    d = default or {}
    with st.form("game"):
        c1, c2, c3 = st.columns(3)
        title = c1.text_input(t("title", lang), d.get("title", "My New Game"))
        console = c2.selectbox(t("console", lang), CONSOLES, index=CONSOLES.index(d.get("console", "Nintendo Wii")))
        year = c3.number_input(t("year", lang), 2004, 2015, int(d.get("year", 2009)))
        c1, c2, c3 = st.columns(3)
        genres = c1.multiselect(t("genres", lang), GENRES, default=split_list(d.get("genres", "Action")))
        pubs = sorted(set(games()["metadata_publishers"].dropna().map(lambda p: split_list(p)[0])))
        pub_default = (split_list(d.get("publisher")) or ["(unknown)"])[0]
        opts = ["(unknown)"] + pubs
        publisher = c2.selectbox(t("publisher", lang), opts, index=opts.index(pub_default) if pub_default in opts else 0)
        rating = c3.selectbox(t("rating", lang), RATINGS, index=RATINGS.index(d.get("rating", "E")))
        c1, c2, c3, c4 = st.columns(4)
        max_players = c1.number_input(t("max_players", lang), 1, 16, int(d.get("max_players", 1)))
        hours = c2.number_input(t("main_story_hours", lang), 0.5, 200.0, float(d.get("main_story_hours") or 8.0), 0.5)
        n_platforms = c3.number_input(t("n_platforms", lang), 1, 5, int(d.get("n_platforms", 1)))
        is_sequel = c4.checkbox(t("is_sequel", lang), bool(d.get("is_sequel", False)))
        if not st.form_submit_button(t("predict_btn", lang), type="primary"):
            return None
    return {"title": title, "console": console, "year": int(year), "rating": rating,
            "genres": ",".join(genres) or "Action", "publisher": "" if publisher == "(unknown)" else publisher,
            "max_players": int(max_players), "main_story_hours": float(hours), "n_platforms": int(n_platforms),
            "is_sequel": is_sequel}


def page_predict():
    st.title(t("page_predict", lang))
    if not model_ready():
        return
    p = predictor()
    mode = st.radio(" ", ["new", "existing"], horizontal=True, label_visibility="collapsed",
                    format_func=lambda m: t("mode_new" if m == "new" else "mode_existing", lang))
    if mode == "existing":
        g = games()
        q = st.text_input(t("search", lang), "")
        cand = g[g["title"].str.contains(q, case=False, regex=False)] if q else g.sample(30, random_state=1)
        labels = {r.game_id: f"{r.title} · {r.release_console} · {r.release_year} · score {r.metrics_review_score}"
                  for r in cand.head(200).itertuples()}
        if not labels:
            st.info("—")
            return
        gid = st.selectbox(t("title", lang), list(labels), format_func=labels.get)
        if st.button(t("predict_btn", lang), type="primary"):
            st.session_state["result"] = p.predict(p.spec_for_game(int(gid)), game_id=int(gid))
            st.session_state.pop("agent", None)
    else:
        spec = game_form()
        if spec:
            st.session_state["result"] = p.predict(spec)
            st.session_state.pop("agent", None)

    res = st.session_state.get("result")
    if not res:
        return
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric(t("proba", lang), f"{res['proba_negative']:.1%}", help=f"threshold {res['threshold']:.2f}")
    c2.markdown(f"**{t('risk', lang)}**<br><span style='font-size:1.6rem'>{risk_badge(res['risk_level'])}</span>",
                unsafe_allow_html=True)
    if "actual_score" in res:
        c3.metric(t("actual", lang), res["actual_score"],
                  delta="negative" if res["actual_negative"] else "ok",
                  delta_color="inverse" if res["actual_negative"] else "normal")
    if res["spec"].get("main_story_hours") is None:
        st.caption("ℹ️ " + t("hours_unknown", lang))

    st.subheader(t("why", lang))
    st.caption(t("shap_note", lang))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(shap_chart(res["explanation"]["all_factors"]), use_container_width=True)
    with c2:
        md("\n".join(f"- {factor_sentence(f, lang)}" for f in res["explanation"]["factors"][:6]))

    st.subheader(t("measures", lang))
    risky = [f["feature"] for f in res["explanation"]["factors"] if f["shap"] > 0][:5]
    ev = actions.evaluate_measures(p, res["spec"], res["proba_negative"], risky)
    tbl = pd.DataFrame(ev["measures"])
    tbl.insert(0, "measure", tbl["code"].map(lambda c: measure_name(c, lang)))
    tbl["approval"] = tbl["requires_approval"].map({True: "🔒 " + t("needs_approval", lang), False: "⚡ " + t("auto", lang)})
    st.dataframe(tbl[["measure", "kind", "cost_k", "p_before", "p_after", "expected_benefit_k", "net_value_k",
                      "approval"]].style.format({"p_before": "{:.1%}", "p_after": "{:.1%}", "cost_k": "{:.0f}",
                                                 "expected_benefit_k": "{:.0f}", "net_value_k": "{:+.0f}"}),
                 use_container_width=True, hide_index=True)
    md(f"**{t('recommended_plan', lang)}:** " + ", ".join(measure_name(c, lang) for c in ev["recommended_plan"])
                + "  \n" + t("plan_summary", lang, p0=ev["p_before"], p1=ev["p_after_plan"], cost=ev["plan_cost_k"],
                             ben=ev["plan_expected_benefit_k"]))

    from nrg.agent import llm_available, run_agent
    llm = llm_available()
    st.caption(t("agent_mode_llm", lang, model=config.LLM_NAME) if llm else t("agent_mode_rule", lang))
    if st.button(t("run_agent", lang), type="primary"):
        with st.spinner("🤖 ..."):
            st.session_state["agent"] = run_agent(res["prediction_id"], lang)
    out = st.session_state.get("agent")
    if out:
        st.subheader(t("agent_report", lang))
        md(out["answer"])
        st.markdown(f"**{t('created_actions', lang)}:**")
        for a in out["created_actions"]:
            icon = "🔒" if a["status"] == "pending_approval" else "⚡"
            st.write(f"{icon} #{a['action_id']} {measure_name(a['code'], lang)} — `{a['status']}` → {a['delivery']}")
        with st.expander(t("agent_trace", lang)):
            for i, step in enumerate(out["trace"], 1):
                st.markdown(f"**{i}. `{step['tool']}`** `{json.dumps(step['args'], ensure_ascii=False)}`")
                st.json(step["result"], expanded=False)


# =============================================================== APPROVALS (HITL)
def page_approvals():
    st.title(t("page_approvals", lang))
    pend = db.list_actions("pending_approval")
    st.subheader(f"{t('pending', lang)} ({len(pend)})")
    if pend.empty:
        st.info(t("no_pending", lang))
    for a in pend.itertuples():
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                md(f"**#{a.id} · {a.title}** — {measure_name(a.action_code, lang)}  \n"
                   f"{measure_desc(a.action_code, lang)}  \n"
                   f"💰 {a.cost_k:.0f} k$ · 📈 {a.expected_benefit_k:.0f} k$ · _{a.rationale}_")
            if c2.button("✅ " + t("approve", lang), key=f"ap{a.id}"):
                db.set_action_status(a.id, "approved", "manager (web UI)")
                automation.notify_decision(a.id)
                st.rerun()
            if c3.button("❌ " + t("reject", lang), key=f"rj{a.id}"):
                db.set_action_status(a.id, "rejected", "manager (web UI)")
                automation.notify_decision(a.id)
                st.rerun()
    st.subheader(t("all_actions", lang))
    allx = db.list_actions()
    if not allx.empty:
        allx["measure"] = allx["action_code"].map(lambda c: measure_name(c, lang))
        st.dataframe(allx[["id", "created_at", "title", "measure", "cost_k", "expected_benefit_k", "status",
                           "decided_by", "source"]], use_container_width=True, hide_index=True)


# =============================================================== MONITORING
def page_monitoring():
    st.title(t("page_monitoring", lang))
    if not model_ready():
        return
    from nrg.drift import drift_report
    rep = drift_report()
    st.subheader(t("drift", lang))
    st.caption(t("drift_note", lang))
    if not rep["enough_data"]:
        st.info(t("not_enough", lang, n=rep["min_samples"]) + f" ({rep['n_current']})")
    else:
        df = pd.DataFrame(rep["table"])
        df["label"] = df["feature"].map(lambda f: feature_label(f, lang))
        color = {"stable": "#0ca30c", "moderate": "#fab219", "significant": "#d03b3b"}
        fig = px.bar(df.sort_values("psi"), x="psi", y="label", color="status", orientation="h",
                     color_discrete_map=color)
        fig.add_vline(x=0.1, line_dash="dot", line_color="#888")
        fig.add_vline(x=0.25, line_dash="dot", line_color="#888")
        fig.update_layout(height=420, margin=dict(t=10, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        if rep["retrain_recommended"]:
            st.warning(f"⚠️ Retrain recommended: {', '.join(rep['drifted_features'])}")
        else:
            st.success("✅ No significant drift (release_year drifts by design and is excluded from the alarm)")
    st.subheader(t("predictions_log", lang))
    log = db.list_predictions()
    if not log.empty:
        st.dataframe(log[["id", "created_at", "title", "proba_negative", "risk_level", "model_name"]],
                     use_container_width=True, hide_index=True)
    outbox = config.OUTBOX_PATH
    if outbox.exists():
        with st.expander("📤 n8n outbox (events not delivered to n8n)"):
            st.code("\n".join(outbox.read_text(encoding="utf-8").splitlines()[-10:]), language="json")


{"page_dashboard": page_dashboard, "page_models": page_models, "page_predict": page_predict,
 "page_approvals": page_approvals, "page_monitoring": page_monitoring}[page]()
