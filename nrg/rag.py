"""RAG (retrieval-augmented generation): the agent searches a small knowledge base before deciding.

Knowledge base = Markdown files in docs/kb/ (measures playbook, company policy, insights computed from the data)
+ the live log of past manager decisions from the `actions` table. Documents are split into sections and
indexed with TF-IDF on character n-grams (works for EN/RU/RO, no extra model or API needed). The agent tool
`search_knowledge_base` returns the most relevant passages, and the LLM uses them as grounded context.

    python -m nrg.rag --build        regenerate docs/kb/data_insights.md from the games table
    python -m nrg.rag "short story"  search from the command line
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, db

KB_DIR = config.ROOT / "docs" / "kb"


def _sections(path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"\n(?=## )", text)
    doc_title = parts[0].splitlines()[0].lstrip("# ").strip()
    out = []
    for p in parts[1:] if len(parts) > 1 else parts:
        lines = p.strip().splitlines()
        out.append({"source": path.name, "title": f"{doc_title} — {lines[0].lstrip('# ').strip()}",
                    "text": " ".join(l.strip() for l in lines[1:] if l.strip())})
    return out


def _decision_docs() -> list[dict]:
    """Past manager decisions as retrievable 'cases' (the knowledge base grows with every decision)."""
    try:
        acts = db.list_actions()
    except Exception:
        return []
    acts = acts[acts["status"].isin(["executed", "rejected"]) & acts["decided_by"].notna()].head(200)
    return [{"source": "actions_log", "title": f"Past decision #{a.id}: {a.action_code} for {a.title}",
             "text": f"Manager decision: {a.status}. Measure {a.action_code}, cost {a.cost_k} k$, expected benefit "
                     f"{round(a.expected_benefit_k or 0)} k$. Rationale: {a.rationale}"} for a in acts.itertuples()]


@lru_cache(maxsize=1)
def _static_index():
    docs = [s for p in sorted(KB_DIR.glob("*.md")) for s in _sections(p)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
    return docs, vec


def search(query: str, k: int = 3) -> list[dict]:
    docs, vec = _static_index()
    docs = docs + _decision_docs()
    if not docs or not query.strip():
        return []
    mat = vec.fit_transform([d["title"] + " " + d["text"] for d in docs] + [query])
    scores = (mat[:-1] @ mat[-1].T).toarray().ravel()
    best = np.argsort(scores)[::-1][:k]
    return [{**docs[i], "score": round(float(scores[i]), 3)} for i in best if scores[i] > 0]


def build_data_insights() -> str:
    """Write docs/kb/data_insights.md: flop rates computed from the merged dataset (facts, not opinions)."""
    from .features import split_list
    g = db.load_games()
    g = g[g["metrics_review_score"].notna()].copy()
    g["neg"] = g["metrics_review_score"] < config.NEGATIVE_THRESHOLD
    base = g["neg"].mean()
    lines = ["# Insights from the data", "",
             f"Computed from {len(g)} games with a critic score (Kaggle + CORGIS). Overall share of negative "
             f"reviews (score < {config.NEGATIVE_THRESHOLD}): {base:.0%}.", ""]

    def block(title, s: pd.DataFrame, min_n=60):
        s = s[s["n"] >= min_n].sort_values("neg", ascending=False)
        lines.extend([f"## {title}", " ".join(f"{i}: {r.neg:.0%} negative of {int(r.n)} games;"
                                             for i, r in s.iterrows()), ""])

    gg = g.assign(genre=g["metadata_genres"].map(split_list)).explode("genre")
    block("Negative share by genre", gg.groupby("genre")["neg"].agg(n="size", neg="mean"))
    block("Negative share by platform", g.groupby("release_console")["neg"].agg(n="size", neg="mean"), 150)
    block("Negative share by ESRB rating", g.groupby("release_rating")["neg"].agg(n="size", neg="mean"))
    h = g[g["length_main_story_median"].notna()]
    bins = pd.cut(h["length_main_story_median"], [0, 5, 10, 20, 200], labels=["<5 h", "5-10 h", "10-20 h", "20+ h"])
    block("Negative share by main-story length (short story)", h.groupby(bins, observed=True)["neg"].agg(
        n="size", neg="mean"), 30)
    seq = g["title"].str.contains(r"\b(?:2|3|4|II|III|IV)\b", regex=True)
    block("Sequels versus new games", g.groupby(seq.map({True: "sequel", False: "new IP"}))["neg"].agg(
        n="size", neg="mean"))
    pubs = g.assign(pub=g["metadata_publishers"].fillna("Unknown")).groupby("pub")["neg"].agg(n="size", neg="mean")
    block("Publishers with the highest share of negative reviews (at least 80 games)", pubs.head(400), 80)
    text = "\n".join(lines)
    (KB_DIR / "data_insights.md").write_text(text, encoding="utf-8")
    _static_index.cache_clear()
    return text


if __name__ == "__main__":
    if "--build" in sys.argv:
        print(build_data_insights())
    else:
        for r in search(" ".join(sys.argv[1:]) or "short story length extend content"):
            print(f"[{r['score']}] {r['source']} :: {r['title']}\n    {r['text'][:200]}")
