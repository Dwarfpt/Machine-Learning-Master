"""Translations: Russian / Romanian / English. Every string is a (ru, ro, en) tuple."""
from __future__ import annotations

LANGS = {"ru": "Русский", "ro": "Română", "en": "English"}
_IDX = {"ru": 0, "ro": 1, "en": 2}

STR = {
    "app_title": ("Negative Review Guard — прогноз негативных отзывов",
                  "Negative Review Guard — prognozarea recenziilor negative",
                  "Negative Review Guard — negative review prediction"),
    "app_subtitle": ("Данные → ML → Прогноз → Объяснение → AI-агент → Действие",
                     "Date → ML → Predicție → Explicație → Agent AI → Acțiune",
                     "Data → ML → Prediction → Explanation → AI agent → Action"),
    "language": ("Язык", "Limba", "Language"),
    "page_dashboard": ("📊 Дашборд", "📊 Dashboard", "📊 Dashboard"),
    "page_models": ("🧠 Модели и XAI", "🧠 Modele și XAI", "🧠 Models & XAI"),
    "page_predict": ("🔮 Прогноз и действия", "🔮 Predicție și acțiuni", "🔮 Predict & act"),
    "page_approvals": ("✅ Согласование (HITL)", "✅ Aprobare (HITL)", "✅ Approvals (HITL)"),
    "page_monitoring": ("📈 Мониторинг", "📈 Monitorizare", "📈 Monitoring"),
    "kpi_games": ("Игр в базе", "Jocuri în bază", "Games in DB"),
    "kpi_negative": ("Доля негативных (<{thr})", "Ponderea negative (<{thr})", "Negative share (<{thr})"),
    "kpi_avg_score": ("Средняя оценка", "Scor mediu", "Average score"),
    "kpi_sales_gap": ("Медиана продаж: хорошие vs негативные", "Mediana vânzărilor: bune vs negative",
                      "Median sales: good vs negative"),
    "score_dist": ("Распределение оценок критиков", "Distribuția scorurilor criticilor",
                   "Critic score distribution"),
    "neg_by": ("Доля негативных отзывов по признаку", "Ponderea recenziilor negative după", "Negative rate by"),
    "sales_vs_score": ("Продажи vs оценка (цена негативного отзыва)",
                       "Vânzări vs scor (costul unei recenzii negative)",
                       "Sales vs score (the cost of a negative review)"),
    "data_quality": ("Качество данных", "Calitatea datelor", "Data quality"),
    "model_comparison": ("Сравнение моделей", "Compararea modelelor", "Model comparison"),
    "selected_model": ("Выбранная модель", "Modelul selectat", "Selected model"),
    "selection_rule": ("Выбор по PR-AUC на 5-fold CV (train 2004–2007); тест — релизы 2008 года (time-based split).",
                       "Selecție după PR-AUC pe 5-fold CV (train 2004–2007); test — lansările din 2008.",
                       "Selected by 5-fold CV PR-AUC (train 2004–2007); test = 2008 releases (time-based split)."),
    "global_importance": ("Глобальная важность признаков (mean |SHAP|)",
                          "Importanța globală a caracteristicilor (mean |SHAP|)",
                          "Global feature importance (mean |SHAP|)"),
    "mode_existing": ("Игра из базы", "Joc din bază", "Game from DB"),
    "mode_new": ("Новая игра (до релиза)", "Joc nou (înainte de lansare)", "New game (pre-release)"),
    "search": ("Поиск по названию", "Căutare după titlu", "Search by title"),
    "title": ("Название", "Titlu", "Title"),
    "console": ("Платформа", "Platformă", "Console"),
    "year": ("Год релиза", "Anul lansării", "Release year"),
    "rating": ("Возрастной рейтинг (ESRB)", "Rating de vârstă (ESRB)", "Age rating (ESRB)"),
    "genres": ("Жанры", "Genuri", "Genres"),
    "publisher": ("Издатель", "Editor", "Publisher"),
    "max_players": ("Макс. игроков", "Nr. maxim de jucători", "Max players"),
    "main_story_hours": ("Плановая длина сюжета, ч", "Durata planificată a poveștii, ore",
                         "Planned main story length, h"),
    "n_platforms": ("Число платформ релиза", "Număr de platforme", "Number of release platforms"),
    "is_sequel": ("Сиквел / часть серии", "Continuare / parte a seriei", "Sequel / franchise entry"),
    "predict_btn": ("Спрогнозировать", "Prezice", "Predict"),
    "proba": ("Вероятность негативного отзыва", "Probabilitatea unei recenzii negative",
              "Probability of a negative review"),
    "risk": ("Уровень риска", "Nivel de risc", "Risk level"),
    "risk_low": ("низкий", "scăzut", "low"),
    "risk_medium": ("средний", "mediu", "medium"),
    "risk_high": ("высокий", "ridicat", "high"),
    "actual": ("Фактическая оценка", "Scor real", "Actual score"),
    "why": ("Почему модель так считает (SHAP)", "De ce crede modelul așa (SHAP)", "Why the model thinks so (SHAP)"),
    "shap_note": ("Красные факторы повышают риск, зелёные — снижают. Значения — вклад в оценку модели.",
                  "Factorii roșii cresc riscul, cei verzi îl reduc. Valorile = contribuția la scorul modelului.",
                  "Red factors increase the risk, green ones decrease it. Values = contribution to the model score."),
    "measures": ("Оценка мер предотвращения (what-if через модель)",
                 "Evaluarea măsurilor de prevenire (what-if prin model)",
                 "Preventive measures evaluation (what-if through the model)"),
    "run_agent": ("🤖 Запустить AI-агента", "🤖 Rulează agentul AI", "🤖 Run AI agent"),
    "agent_report": ("Отчёт AI-агента", "Raportul agentului AI", "AI agent report"),
    "agent_trace": ("Вызовы инструментов агента", "Apelurile instrumentelor agentului", "Agent tool calls"),
    "agent_mode_llm": ("LLM-агент (llama.cpp: {model}) — ответ на CPU занимает 1–3 мин",
                       "Agent LLM (llama.cpp: {model}) — pe CPU durează 1–3 min",
                       "LLM agent (llama.cpp: {model}) — takes 1–3 min on CPU"),
    "agent_mode_rule": ("Rule-based агент (LLM-модель недоступна)", "Agent rule-based (modelul LLM indisponibil)",
                        "Rule-based agent (LLM model unavailable)"),
    "created_actions": ("Созданные действия", "Acțiuni create", "Created actions"),
    "pending": ("Ожидают согласования", "Așteaptă aprobare", "Pending approval"),
    "approve": ("Одобрить", "Aprobă", "Approve"),
    "reject": ("Отклонить", "Respinge", "Reject"),
    "no_pending": ("Нет действий, ожидающих согласования.", "Nu există acțiuni în așteptare.",
                   "No actions waiting for approval."),
    "all_actions": ("Все действия", "Toate acțiunile", "All actions"),
    "drift": ("Data drift (PSI): обучающие данные vs новые запросы",
              "Data drift (PSI): date de antrenare vs cereri noi",
              "Data drift (PSI): training data vs new requests"),
    "drift_note": ("PSI < 0.1 — стабильно, 0.1–0.25 — умеренный дрейф, > 0.25 — значительный (переобучить).",
                   "PSI < 0.1 — stabil, 0.1–0.25 — drift moderat, > 0.25 — semnificativ (reantrenare).",
                   "PSI < 0.1 stable, 0.1–0.25 moderate drift, > 0.25 significant (retrain)."),
    "predictions_log": ("Журнал прогнозов", "Jurnalul predicțiilor", "Prediction log"),
    "not_enough": ("Недостаточно новых прогнозов для оценки дрейфа (нужно ≥ {n}).",
                   "Prea puține predicții noi pentru drift (necesar ≥ {n}).",
                   "Not enough new predictions to assess drift (need ≥ {n})."),
    "no_model": ("Модель не обучена. Выполните: python -m nrg.train",
                 "Modelul nu este antrenat. Rulați: python -m nrg.train",
                 "Model not trained. Run: python -m nrg.train"),
    "recommended_plan": ("Рекомендованный план", "Planul recomandat", "Recommended plan"),
    "plan_summary": ("Риск {p0:.0%} → {p1:.0%} · стоимость {cost:.0f} k$ · ожидаемая выгода {ben:.0f} k$",
                     "Risc {p0:.0%} → {p1:.0%} · cost {cost:.0f} k$ · beneficiu așteptat {ben:.0f} k$",
                     "Risk {p0:.0%} → {p1:.0%} · cost {cost:.0f} k$ · expected benefit {ben:.0f} k$"),
    "needs_approval": ("требует согласования", "necesită aprobare", "needs approval"),
    "auto": ("авто", "automat", "auto"),
    "status": ("Статус", "Stare", "Status"),
    "hours_unknown": ("длина неизвестна (подставлена медиана)", "durată necunoscută (mediană imputată)",
                      "length unknown (median imputed)"),
}

FEATURES = {
    "main_story_hours": ("Длина сюжета (ч)", "Durata poveștii (ore)", "Main story length (h)"),
    "release_year": ("Год релиза", "Anul lansării", "Release year"),
    "max_players": ("Макс. игроков", "Nr. maxim de jucători", "Max players"),
    "n_platforms": ("Число платформ", "Nr. platforme", "Number of platforms"),
    "n_genres": ("Число жанров", "Nr. genuri", "Number of genres"),
    "n_publishers": ("Число издателей", "Nr. editori", "Number of publishers"),
    "pub_prior_mean_score": ("Ср. оценка прошлых игр издателя", "Scorul mediu anterior al editorului",
                             "Publisher's past mean score"),
    "pub_prior_neg_rate": ("Доля провалов издателя в прошлом", "Rata eșecurilor anterioare ale editorului",
                           "Publisher's past flop rate"),
    "pub_prior_n_games": ("Опыт издателя (игр ранее)", "Experiența editorului (jocuri anterioare)",
                          "Publisher experience (past games)"),
    "handheld": ("Портативная консоль", "Consolă portabilă", "Handheld console"),
    "is_multiplayer": ("Мультиплеер", "Multiplayer", "Multiplayer"),
    "is_multiplatform": ("Мультиплатформа", "Multiplatformă", "Multiplatform"),
    "is_sequel": ("Сиквел", "Continuare", "Sequel"),
    "has_subtitle": ("Подзаголовок в названии", "Subtitlu în titlu", "Subtitle in title"),
    "publisher_unknown": ("Издатель неизвестен", "Editor necunoscut", "Publisher unknown"),
    "console": ("Платформа", "Platformă", "Console"),
    "rating": ("Рейтинг ESRB", "Rating ESRB", "ESRB rating"),
    "publisher_main": ("Издатель", "Editor", "Publisher"),
    "genre_action": ("Жанр: Action", "Gen: Action", "Genre: Action"),
    "genre_sports": ("Жанр: Sports", "Gen: Sports", "Genre: Sports"),
    "genre_strategy": ("Жанр: Strategy", "Gen: Strategy", "Genre: Strategy"),
    "genre_role_playing_rpg": ("Жанр: RPG", "Gen: RPG", "Genre: RPG"),
    "genre_racing_driving": ("Жанр: Racing", "Gen: Racing", "Genre: Racing"),
    "genre_simulation": ("Жанр: Simulation", "Gen: Simulation", "Genre: Simulation"),
    "genre_adventure": ("Жанр: Adventure", "Gen: Adventure", "Genre: Adventure"),
    "genre_educational": ("Жанр: Educational", "Gen: Educational", "Genre: Educational"),
}

MEASURES = {
    "EXTEND_CONTENT": (("Расширить контент (+50% сюжета)", "Добавить миссии/главы: модель связывает короткие игры с низкими оценками."),
                       ("Extinderea conținutului (+50% poveste)", "Misiuni/capitole noi: modelul asociază jocurile scurte cu scoruri mici."),
                       ("Extend content (+50% story)", "Add missions/chapters: the model links short games to low scores.")),
    "ADD_MULTIPLAYER": (("Добавить кооператив/мультиплеер (до 4 игроков)", "Многопользовательский режим повышает ценность для критиков."),
                        ("Adăugarea co-op/multiplayer (până la 4 jucători)", "Modul multiplayer crește valoarea percepută de critici."),
                        ("Add co-op / multiplayer (up to 4 players)", "Multiplayer raises perceived value for critics.")),
    "MULTIPLATFORM_PORT": (("Выпуск ещё на одной платформе", "Порт расширяет аудиторию и обычно означает более зрелый продукт."),
                           ("Lansare pe încă o platformă", "Portarea extinde publicul; de obicei indică un produs mai matur."),
                           ("Release on one more platform", "A port widens the audience; usually signals a more mature product.")),
    "DELAY_RELEASE_POLISH": (("Перенос релиза на 3 мес. для полировки", "Дорогая мера, всегда требует согласования менеджера."),
                             ("Amânarea lansării cu 3 luni pentru finisare", "Măsură costisitoare, necesită mereu aprobarea managerului."),
                             ("Delay release by 3 months for polish", "Expensive measure, always needs manager approval.")),
    "EXTENDED_QA": (("Расширенное QA-тестирование", "Доп. цикл тестирования багов и производительности."),
                    ("Testare QA extinsă", "Un ciclu suplimentar de testare a bug-urilor și performanței."),
                    ("Extended QA testing", "An extra bug & performance testing cycle.")),
    "PLAYTEST_FOCUS_GROUP": (("Плейтест с фокус-группой", "Проверка геймплея на целевой аудитории до релиза."),
                             ("Playtest cu focus-grup", "Verificarea gameplay-ului pe publicul țintă înainte de lansare."),
                             ("Focus-group playtest", "Validate gameplay with the target audience before launch.")),
    "CRITIC_PREVIEW_PR": (("Превью для критиков и управление ожиданиями", "Ранний доступ для прессы, корректный маркетинг."),
                          ("Preview pentru critici și gestionarea așteptărilor", "Acces timpuriu pentru presă, marketing corect."),
                          ("Critic previews & expectation management", "Early press access, honest marketing.")),
    "MONITOR_ONLY": (("Только мониторинг", "Риск низкий — вмешательство не окупается."),
                     ("Doar monitorizare", "Risc scăzut — intervenția nu se justifică."),
                     ("Monitor only", "Low risk — intervention does not pay off.")),
}


def t(key: str, lang: str = "ru", **kw) -> str:
    s = STR.get(key, (key, key, key))[_IDX.get(lang, 0)]
    return s.format(**kw) if kw else s


def feature_label(feature: str, lang: str = "ru") -> str:
    return FEATURES.get(feature, (feature,) * 3)[_IDX.get(lang, 0)]


def measure_name(code: str, lang: str = "ru") -> str:
    return MEASURES.get(code, ((code, ""),) * 3)[_IDX.get(lang, 0)][0]


def measure_desc(code: str, lang: str = "ru") -> str:
    return MEASURES.get(code, ((code, ""),) * 3)[_IDX.get(lang, 0)][1]


def format_value(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}"
    return str(v)


def factor_sentence(f: dict, lang: str = "ru") -> str:
    arrow = "↑" if f["shap"] > 0 else "↓"
    word = {"ru": ("повышает риск", "снижает риск"), "ro": ("crește riscul", "reduce riscul"),
            "en": ("increases risk", "decreases risk")}[lang][0 if f["shap"] > 0 else 1]
    return f"{feature_label(f['feature'], lang)} = {format_value(f['value'])} {arrow} {word} ({f['shap']:+.3f})"
