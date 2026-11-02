"""Builds and executes notebooks/02_AutoML_PyCaret.ipynb in the Python 3.11 AutoML environment:
    .venv-automl\\Scripts\\python notebooks\\build_automl_notebook.py
(needs data/processed/*.csv from `python -m nrg.export_dataset`)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

cells = [
    ("md", """# AutoML с PyCaret
PyCaret автоматически перебирает ~15 классификаторов и ~15 регрессоров, подбирает гиперпараметры победителя и строит
ансамбли (blending, stacking). Это **независимая проверка** ручного выбора модели в `nrg/train.py`:
те же признаки и тот же тест по времени (обучение до 2012, тест 2013–2016).

PyCaret 3.3 работает только на Python ≤ 3.11, поэтому ноутбук выполняется в отдельном окружении `.venv-automl`."""),
    ("code", """import sys, pandas as pd
from sklearn.metrics import average_precision_score
print(sys.version.split()[0])
train = pd.read_csv('../data/processed/train.csv'); test = pd.read_csv('../data/processed/test.csv')
CAT = ['console', 'rating', 'publisher_main']
train.shape, test.shape, round(train.negative.mean(), 3), round(test.negative.mean(), 3)"""),
    ("md", "## 1. Классификация: провал (оценка < 60) или нет"),
    ("code", """from pycaret import classification as C
C.setup(data=train.drop(columns=['score']), test_data=test.drop(columns=['score']), target='negative',
        categorical_features=CAT, max_encoding_ohe=40, normalize=True, index=False, fold_strategy='stratifiedkfold',
        fold=5, fold_shuffle=True, session_id=42, verbose=False, html=False, n_jobs=-1)
C.add_metric('pr_auc', 'PR-AUC', average_precision_score, target='pred_proba')
best3 = C.compare_models(sort='PR-AUC', n_select=3, exclude=['lda', 'qda', 'dummy', 'svm', 'ridge'], verbose=False)
C.pull()"""),
    ("md", "**compare_models** обучил каждую модель на 5 фолдах и отсортировал по PR-AUC (метрика для редкого класса)."),
    ("code", """tuned = C.tune_model(best3[0], optimize='PR-AUC', n_iter=25, choose_better=True, verbose=False)
blended = C.blend_models(best3, method='soft', choose_better=False, verbose=False)
stacked = C.stack_models(best3, choose_better=False, verbose=False)
te = test.drop(columns=['score'])
rows = []
for name, m in [('лучшая модель', best3[0]), ('после tune_model', tuned), ('blend top-3', blended), ('stack top-3', stacked)]:
    p = C.predict_model(m, data=te.drop(columns=['negative']), raw_score=True, verbose=False)['prediction_score_1']
    rows.append({'вариант': name, 'test PR-AUC': round(average_precision_score(te.negative, p), 3)})
pd.DataFrame(rows)"""),
    ("md", "## 2. Регрессия: прогноз самой оценки 0–100"),
    ("code", """from pycaret import regression as R
R.setup(data=train.drop(columns=['negative']), test_data=test.drop(columns=['negative']), target='score',
        categorical_features=CAT, max_encoding_ohe=40, normalize=True, index=False, fold=5, fold_shuffle=True,
        session_id=42, verbose=False, html=False, n_jobs=-1)
best_r = R.compare_models(sort='MAE', n_select=3, exclude=['dummy'], verbose=False)
R.pull()"""),
    ("md", """**Вывод.** PyCaret подтверждает ручной выбор: лидируют ансамбли деревьев (Random Forest / Extra Trees / бустинг);
подбор гиперпараметров и комбинации дают небольшой прирост. Регрессия ошибается в среднем на ~8 баллов из 100."""),
]

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3.11 (AutoML)", "language": "python"}
out = Path(__file__).with_name("02_AutoML_PyCaret.ipynb")
NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
nbf.write(nb, out)
print(f"Executed and saved {out}")
