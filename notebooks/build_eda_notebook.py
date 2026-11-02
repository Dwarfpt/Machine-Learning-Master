"""Builds and executes notebooks/01_EDA_and_modeling.ipynb (run from the project root, after `python -m nrg.train`
and `python -m nrg.regression`)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

cells = [
    ("md", """# Negative Review Guard — данные, EDA, качество и моделирование
**Тема:** Реализация системы прогнозирования негативных отзывов покупателей и автоматизированного выбора мер по их предотвращению.

**Данные — два открытых источника, объединённые в одну БД (SQLite):**
1. **Kaggle «Video Game Sales with Ratings»** (VGChartz + Metacritic, декабрь 2016): 16 719 строк, 31 платформа, 1980–2016;
   оценка критиков, жанр, издатель, разработчик, рейтинг ESRB, продажи по регионам, оценки пользователей.
2. **CORGIS Video Games** (2004–2008, 5 консолей, 1 212 строк): длина игры (HowLongToBeat), число игроков.

Целевая переменная: *негативный отзыв* = оценка критиков **< 60**.

> RO: Sistem de prognozare a recenziilor negative și selecție automată a măsurilor de prevenire.
> EN: System for predicting negative reviews and automatically selecting preventive measures."""),
    ("code", """import os, sys, json
sys.path.insert(0, os.path.abspath('..'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from nrg import config, db, quality
pd.set_option('display.max_columns', 50)
if not config.DB_PATH.exists():
    db.init_db()
games = db.load_games()                       # объединённая таблица games
kaggle, corgis = db.load_raw('kaggle_raw'), db.load_raw('corgis_raw')   # исходные таблицы как есть
print('kaggle_raw', kaggle.shape, '| corgis_raw', corgis.shape, '| games', games.shape)"""),
    ("md", "## 1. Объединение источников"),
    ("code", """integ = json.load(open(config.REPORTS_DIR / 'data_integration.json', encoding='utf-8'))
pd.Series(integ).to_frame('value')"""),
    ("md", """Игры сопоставляются по **нормализованному названию + платформе**. Из Kaggle берём оценку, жанр, издателя, разработчика
и продажи; из CORGIS — длину сюжета и число игроков. Оценки в общих играх совпадают (средняя разница ≈ 0.1 балла) —
это один и тот же Metacritic, т.е. источники согласованы."""),
    ("code", "games.source.value_counts()"),
    ("md", "## 2. Проверка качества данных (оба источника)"),
    ("code", """rep = quality.main()
pd.DataFrame(rep['issues'])[['source','severity','check','detail','decision']]"""),
    ("md", """**Главное:** у 51% строк Kaggle нет оценки критиков (не используются для обучения, но нужны для подсчёта платформ);
`user_score = 'tbd'` — текст в числовой колонке; 4 игры «из будущего» (после 2016); 6 константных признаков в CORGIS;
пропуски издателя/разработчика/рейтинга → категория Unknown / медиана."""),
    ("md", "## 3. EDA: целевая переменная"),
    ("code", """g = games[games.metrics_review_score.notna()].copy()
g['negative'] = (g.metrics_review_score < config.NEGATIVE_THRESHOLD).astype(int)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
g.metrics_review_score.hist(bins=50, ax=ax[0], color='#2a78d6'); ax[0].axvline(60, color='#d03b3b', ls='--')
ax[0].set_title('Распределение оценок (порог 60)')
by_year = g[g.release_year >= 1998].groupby('release_year').negative.agg(['mean', 'size'])
by_year['mean'].plot.bar(ax=ax[1], color='#2a78d6'); ax[1].set_title('Доля негативных по годам (тренд)')
plt.tight_layout(); print(len(g), 'игр с оценкой; доля негативных:', round(g.negative.mean(), 3))"""),
    ("code", """fig, ax = plt.subplots(1, 3, figsize=(16, 4))
top = g.release_console.value_counts().head(14).index
g[g.release_console.isin(top)].groupby('release_console').negative.mean().sort_values().plot.barh(ax=ax[0], color='#2a78d6'); ax[0].set_title('Платформа')
g.groupby('release_rating').negative.mean().plot.bar(ax=ax[1], color='#2a78d6'); ax[1].set_title('Рейтинг ESRB')
gg = g.assign(genre=g.metadata_genres.str.split(',')).explode('genre')
gg.groupby('genre').negative.mean().sort_values().plot.barh(ax=ax[2], color='#2a78d6'); ax[2].set_title('Жанр')
plt.tight_layout()"""),
    ("md", """## 4. Сезонная / циклическая компонента
В данных есть только **год** релиза (месяца нет), поэтому «сезонность» в классическом смысле (праздничный сезон) посчитать
нельзя. Цикл рынка игр — это **жизненный цикл консоли**: ранние игры поколения, расцвет, игры в конце поколения.
Признак `console_age` = год релиза − год выхода платформы; плюс тренд по годам (`release_year`)."""),
    ("code", """from nrg.sources import PLATFORM_LAUNCH
g['console_age'] = (g.release_year - g.release_console.map(PLATFORM_LAUNCH)).clip(lower=0)
cyc = g[g.console_age <= 10].groupby('console_age').negative.agg(['mean', 'size'])
ax = cyc['mean'].plot(marker='o', figsize=(7, 3.4), color='#2a78d6'); ax.set_title('Доля негативных vs возраст платформы на релизе')
ax.set_xlabel('лет с выхода платформы'); cyc.round(3).T"""),
    ("md", "## 5. Бизнес-цена негативного отзыва"),
    ("code", """s = g[g.metrics_sales.notna()]
print(s.groupby('negative').metrics_sales.agg(['mean','median']).round(3))
plt.figure(figsize=(7,3.5)); plt.scatter(s.metrics_review_score, s.metrics_sales, s=4, alpha=.3, c='#2a78d6')
plt.yscale('log'); plt.xlabel('Review score'); plt.ylabel('Sales, M units'); plt.title('Продажи vs оценка');"""),
    ("md", """Продажи — основа «выручки под риском» в модуле решений. **Но продажи известны только после релиза** —
в признаки модели они не входят (утечка), как и оценки и число отзывов пользователей."""),
    ("md", "## 6. Аудит утечки данных (leakage)"),
    ("code", """h = g.length_main_story_median.notna()
pd.DataFrame({'neg_rate': [g.negative[~h].mean(), g.negative[h].mean()], 'n': [(~h).sum(), h.sum()]},
             index=['длина неизвестна', 'длина известна']).round(3)"""),
    ("md", """Длина игры известна только для игр из CORGIS. Пропуск теперь в основном значит «игры нет в CORGIS», а не
«игру никто не прошёл», и почти не несёт сигнала → пропуски заполняются медианой внутри пайплайна (без флага пропуска).
Репутация издателя и разработчика (`pub_prior_*`, `dev_prior_*`) считается **только по играм прошлых лет**."""),
    ("md", """## 7. Методология выборки
* **Единица наблюдения:** игра на платформе; в обучение идут только игры с оценкой критиков.
* **Разбиение по времени:** обучение — релизы до 2012, тест — 2013–2016 (модель проверяется «на будущем», как в жизни).
* **Кросс-валидация:** 5-fold *стратифицированная* (в каждом фолде та же доля провалов) — на ней подбираются
  гиперпараметры и выбирается модель. Тест в выборе не участвует.
* **Дисбаланс классов** (≈25% провалов в обучении, 14% в тесте): веса классов (`class_weight`, `scale_pos_weight`),
  порог подбирается по F1 на out-of-fold прогнозах, вероятности калибруются (Platt)."""),
    ("code", """from nrg.train import load_dataset
gl, X, y, train_mask = load_dataset()
print(X.shape[1], 'признаков | train', train_mask.sum(), 'neg', round(y[train_mask].mean(), 3),
      '| test', (~train_mask).sum(), 'neg', round(y[~train_mask].mean(), 3)); X.head()"""),
    ("md", "## 8. Сравнение моделей: 11 моделей → подбор гиперпараметров → ансамбли"),
    ("code", """comp = pd.read_json(config.ARTIFACTS_DIR / 'model_comparison.json')
comp.sort_values(['stage', 'cv_pr_auc'], ascending=[True, False])[
    ['stage','model','family','cv_pr_auc','cv_roc_auc','test_pr_auc','test_roc_auc','test_f1','test_recall','test_precision']].round(3)"""),
    ("md", """* **Этап 1:** модели разных семейств — линейная (логистическая регрессия), по соседям (KNN), вероятностная (Naive Bayes),
  ядерная (SVM), дерево, бэггинг (Random Forest, Extra Trees), бустинг (AdaBoost, Gradient Boosting, LightGBM, XGBoost).
* **Этап 2:** для трёх лучших — RandomizedSearch по широкому пространству гиперпараметров.
* **Этап 3:** комбинации пяти лучших: **голосование большинством** (каждая модель говорит 1/0, если ≥ 3 из 5 сказали
  «провал» — провал), усреднение вероятностей (soft voting), стекинг (логистическая регрессия поверх прогнозов моделей).
* Победитель — по PR-AUC на кросс-валидации; затем калибровка и переобучение на всех данных."""),
    ("md", "## 9. Регрессия: прогноз самой оценки"),
    ("code", """reg = pd.read_json(config.ARTIFACTS_DIR / 'regression_comparison.json')
reg[['model','cv_mae','cv_rmse','cv_r2','test_mae','test_rmse','test_r2','test_pr_auc','test_recall']].round(3)"""),
    ("md", """Регрессия предсказывает балл 0–100 (MAE — средняя ошибка в баллах). Чтобы сравнить с классификацией, прогноз < 60
считается провалом. Классификатор, обученный именно на «провал/не провал», ловит провалы лучше — поэтому в системе
используется классификация, а регрессия — как дополнительная проверка."""),
    ("md", "## 10. AutoML (PyCaret) — независимая проверка"),
    ("code", """ac = pd.read_csv(config.REPORTS_DIR / 'automl_classification.csv')
ac[[c for c in ['Model','Accuracy','AUC','Recall','Prec.','F1','PR-AUC'] if c in ac]]"""),
    ("md", "Подробно — ноутбук `02_AutoML_PyCaret.ipynb` (Python 3.11, отдельное окружение `.venv-automl`)."),
    ("md", "## 11. Explainable AI: SHAP"),
    ("code", """imp = pd.read_json(config.ARTIFACTS_DIR / 'shap_global.json')
imp.head(12).iloc[::-1].plot.barh(x='feature', y='mean_abs_shap', figsize=(7,4), color='#2a78d6', legend=False)
plt.title('Глобальная важность (mean |SHAP|), тест');"""),
    ("code", """from nrg.predict import Predictor
P = Predictor()
spec = {'title': 'Movie Hero: The Game', 'console': 'PlayStation 4', 'year': 2017, 'rating': 'T', 'genres': 'Action',
        'publisher': 'THQ', 'developer': '', 'max_players': 1, 'main_story_hours': 4, 'n_platforms': 1, 'is_sequel': False}
r = P.predict(spec, save=False)
print(P.model_name, '| P(negative) =', round(r['proba_negative'], 3), '| risk:', r['risk_level'])
pd.DataFrame(r['explanation']['factors'])"""),
    ("md", "## 12. От прогноза к действию: модуль решений"),
    ("code", """from nrg.actions import evaluate_measures
ev = evaluate_measures(P, spec, r['proba_negative'])
print('Выручка под риском, k$:', round(ev['revenue_at_risk_k']), '| план:', ev['recommended_plan'])
pd.DataFrame(ev['measures'])[['code','kind','cost_k','p_after','expected_benefit_k','net_value_k','requires_approval']].round(3)"""),
    ("md", """Дальше: AI-агент с RAG (`nrg/agent.py`, `nrg/rag.py`), RPA-обновление данных (`nrg/refresh.py` + n8n), веб-приложение
(`app/streamlit_app.py`), API (`api/main.py`), n8n (`n8n/nrg_workflow.json`)."""),
]

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("01_EDA_and_modeling.ipynb")
NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
nbf.write(nb, out)
print(f"Executed and saved {out}")
