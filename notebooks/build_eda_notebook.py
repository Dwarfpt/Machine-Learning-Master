"""Builds and executes notebooks/01_EDA_and_modeling.ipynb (run from the project root)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

cells = [
    ("md", """# Negative Review Guard — EDA, качество данных и моделирование
**Тема:** Реализация системы прогнозирования негативных отзывов покупателей и автоматизированного выбора мер по их предотвращению.

**Данные:** CORGIS *Video Games* (1212 игр, 2004–2008, 5 платформ) — открытый датасет. Целевая переменная:
*негативный отзыв* = оценка критиков `Metrics.Review Score` **< 60**.

**Бизнес-постановка:** издатель до релиза оценивает риск провала у критиков и выбирает меры (QA, плейтест, контент, перенос релиза), окупающиеся по ожидаемой выгоде.

> RO: Sistem de prognozare a recenziilor negative și selecție automată a măsurilor de prevenire.
> EN: System for predicting negative reviews and automatically selecting preventive measures."""),
    ("code", """import os, sys, json
sys.path.insert(0, os.path.abspath('..'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from nrg import config, db, quality
pd.set_option('display.max_columns', 50)
if not config.DB_PATH.exists():
    db.init_db()
games = db.load_games()            # загрузка из SQLite (таблица games)
games.shape"""),
    ("md", "## 1. Подключение к БД и первичный осмотр"),
    ("code", "games.head(3).T.head(20)"),
    ("code", "games.describe().T[['mean','std','min','50%','max']].round(2)"),
    ("md", "## 2. Проверка качества данных"),
    ("code", """rep = quality.run_quality_checks(games)
pd.DataFrame(rep['issues'])[['severity','check','detail','decision']]"""),
    ("md", """**Выводы по качеству:**
* 6 булевых признаков (`Handheld?`, `Multiplatform?`, `Online?`, `Licensed?`, `Sequel?`, `Re-release?`) **константны (всегда True)** — известная ошибка CORGIS. Удаляем и **восстанавливаем**: handheld из платформы, multiplatform из повторов названия на разных консолях, sequel — регулярным выражением по названию (цифры, римские числа, '08).
* 21.8% издателей пропущено → категория `Unknown`.
* Длина игры = 0 у 25% игр — это «нет данных», а не 0 часов."""),
    ("md", "## 3. EDA: целевая переменная"),
    ("code", """games['negative'] = (games.metrics_review_score < config.NEGATIVE_THRESHOLD).astype(int)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
games.metrics_review_score.hist(bins=40, ax=ax[0], color='#2a78d6'); ax[0].axvline(60, color='#d03b3b', ls='--')
ax[0].set_title('Распределение оценок (порог 60)')
games.groupby('release_year').negative.mean().plot.bar(ax=ax[1], color='#2a78d6'); ax[1].set_title('Доля негативных по годам')
plt.tight_layout(); print('Доля негативных:', round(games.negative.mean(), 3))"""),
    ("code", """fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
games.groupby('release_console').negative.mean().sort_values().plot.barh(ax=ax[0], color='#2a78d6'); ax[0].set_title('Платформа')
games.groupby('release_rating').negative.mean().plot.bar(ax=ax[1], color='#2a78d6'); ax[1].set_title('Рейтинг ESRB')
g = games.assign(genre=games.metadata_genres.str.split(',')).explode('genre')
g.groupby('genre').negative.mean().sort_values().plot.barh(ax=ax[2], color='#2a78d6'); ax[2].set_title('Жанр')
plt.tight_layout()"""),
    ("md", "## 4. Бизнес-цена негативного отзыва"),
    ("code", """print(games.groupby('negative').metrics_sales.agg(['mean','median']).round(3))
plt.figure(figsize=(7,3.5)); plt.scatter(games.metrics_review_score, games.metrics_sales, s=8, alpha=.5, c='#2a78d6')
plt.yscale('log'); plt.xlabel('Review score'); plt.ylabel('Sales, M units'); plt.title('Продажи vs оценка');"""),
    ("md", """Медианные продажи игр с негативными отзывами в ~2.4 раза ниже → «выручка под риском» используется агентом для оценки окупаемости мер.
**Но продажи известны только после релиза** — в признаки модели не входят (утечка)."""),
    ("md", "## 5. Аудит утечки данных (leakage)"),
    ("code", """m = games.length_main_story_median == 0
pd.DataFrame({'neg_rate': [games.negative[m].mean(), games.negative[~m].mean()], 'n': [m.sum(), (~m).sum()]},
             index=['длина неизвестна', 'длина известна']).round(3)"""),
    ("md", """**Важная находка:** если длина игры неизвестна (никто не прошёл её на HowLongToBeat), доля провалов 31% против 20%.
Это сигнал *непопулярности после релиза* → утечка. Решение: обучаемся только на играх с известной длиной (906),
а в форме для новой игры плановая длина — обязательное поле. Поля `Sales`, `Used Price`, `*.Polled` исключены."""),
    ("md", "## 6. Признаки и time-based split"),
    ("code", """from nrg.features import games_to_specs, history_from_games, build_features, FEATURE_COLS
history = history_from_games(games)
specs = games_to_specs(games)
known = (games.length_main_story_median > 0).values
X = build_features(specs[known].reset_index(drop=True), history)
y = games.negative[known].values
print(len(FEATURE_COLS), 'признаков'); X.head()"""),
    ("md", """`pub_prior_*` — репутация издателя, посчитанная **только по играм прошлых лет** (Bayesian smoothing), без утечки.
Train: 2004–2007, Test: 2008 (модель проверяется «на будущем»)."""),
    ("md", "## 7. Сравнение моделей (результаты `python -m nrg.train`)"),
    ("code", """comp = pd.read_json(config.ARTIFACTS_DIR / 'model_comparison.json')
comp[['model','cv_pr_auc','cv_roc_auc','test_pr_auc','test_roc_auc','test_f1','test_recall','test_precision']].round(3)"""),
    ("md", """Выбор модели — по **PR-AUC на кросс-валидации** (а не на тесте, чтобы не подгонять под тест). PR-AUC выбран, т.к. классы
несбалансированы (≈21% негативных). Все модели в ~2 раза лучше базовой линии (PR-AUC 0.18).
Вероятности калибруются (Platt) — они нужны модулю решений для расчёта ожидаемой выгоды."""),
    ("md", "## 8. Explainable AI: SHAP"),
    ("code", """from nrg.predict import Predictor
P = Predictor()
imp = P.explainer.global_importance(X)
imp.head(12).iloc[::-1].plot.barh(x='feature', y='mean_abs_shap', figsize=(7,4), color='#2a78d6', legend=False)
plt.title('Глобальная важность (mean |SHAP|)');"""),
    ("code", """# Локальное объяснение одного прогноза
spec = {'title': 'Galactic Racer 2', 'console': 'Nintendo Wii', 'year': 2009, 'rating': 'E', 'genres': 'Racing / Driving',
        'publisher': 'THQ', 'max_players': 1, 'main_story_hours': 4, 'n_platforms': 1, 'is_sequel': True}
r = P.predict(spec, save=False)
print('P(negative) =', round(r['proba_negative'], 3), '| risk:', r['risk_level'])
pd.DataFrame(r['explanation']['factors'])"""),
    ("md", "## 9. От прогноза к действию: модуль решений"),
    ("code", """from nrg.actions import evaluate_measures
ev = evaluate_measures(P, spec, r['proba_negative'])
print('Выручка под риском, k$:', round(ev['revenue_at_risk_k']), '| план:', ev['recommended_plan'])
pd.DataFrame(ev['measures'])[['code','kind','cost_k','p_after','expected_benefit_k','net_value_k','requires_approval']].round(3)"""),
    ("md", """Меры типа *simulated* оцениваются **самой моделью** (what-if: меняем признак и пересчитываем вероятность),
меры типа *assumption* — экспертное допущение об относительном снижении риска. Дорогие меры (>100 k$ или перенос релиза)
уходят на согласование менеджеру (Human-in-the-loop) через n8n.

Далее: AI-агент (`nrg/agent.py`), веб-приложение (`app/streamlit_app.py`), API (`api/main.py`), n8n (`n8n/nrg_workflow.json`)."""),
]

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("01_EDA_and_modeling.ipynb")
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
nbf.write(nb, out)
print(f"Executed and saved {out}")
