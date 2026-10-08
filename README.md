# PredictiveGuard AI

Асинхронный FastAPI-сервис для оценки риска отказа оборудования: AI4I 2020 → EDA и эксперименты в MLflow → Model Registry → `POST /process`.

## Требования и установка

Python 3.12, uv 0.11.7, Docker с Docker Compose.

```bash
uv sync --frozen
uv run pre-commit install
cp .env.example .env
```

## Обучение и запуск

Сначала поднимите инфраструктуру. MLflow UI: http://127.0.0.1:5050.

```bash
docker compose up -d --build mlflow app-db
uv run jupyter lab notebooks/ai4i_research.ipynb
```

Откройте [ноутбук EDA и обучения](notebooks/ai4i_research.ipynb), выберите Python из `.venv` и выполните все ячейки. При запуске Jupyter через `uv run` этот Python используется по умолчанию. Ноутбук явно обучает модели и назначает `champion`. По завершении запустите API:

```bash
docker compose up -d --build app
```

Swagger: http://127.0.0.1:8000/docs. Приложение один раз разрешает alias в номер версии и загружает `models:/PredictiveGuardFailureModel/<version>`. Все запросы используют эту модель и сохранённый порог. Если alias ещё не назначен, startup завершится ошибкой: сначала выполните обучение.

Для повторного запуска уже обученного проекта достаточно `docker compose up -d`. Обучение отделено профилем `training` и автоматически вместе с API не запускается. При необходимости тот же воспроизводимый workflow можно явно выполнить без Jupyter:

```bash
docker compose --profile training run --build --rm trainer
```

Эта команда создаёт новые Runs и версии и переназначает `champion`. Обычный перезапуск API этого не делает.

## Данные, эксперименты и выбор модели

[data/ai4i2020.csv](data/ai4i2020.csv) — точная копия предоставленного CSV: 10 000 строк, 339 отказов (3.39%). SHA256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`. При отсутствии или подмене файла обучение останавливается.

Цель — `Machine failure`. Используются пять датчиков и категориальный `Type`. Идентификаторы `UDI`, `Product ID` и колонки типов отказов `TWF/HDF/PWF/OSF/RNF` исключены из признаков. Последние раскрывают целевую метку и создают утечку.

Стратифицированное разбиение с seed=42: train/validation/test = 6000/2000/2000. EDA выполняется на train; preprocessing sklearn обучается только на train. Notebook сравнивает:

- DummyClassifier prior — baseline;
- Logistic Regression — balanced weights, scaling, C=1;
- Random Forest — 250 деревьев, depth=12, min leaf=2, balanced weights;
- CatBoost — 500 итераций, depth=6, learning rate=0.05, native Type, balanced weights.

Главная метрика — **validation Average Precision**, поскольку положительный класс редкий. Дополнительно: ROC-AUC, precision, recall, F1, accuracy, PR/ROC-кривые и confusion matrix. Порог каждой содержательной модели максимизирует F1 на validation. Champion выбирается по validation AP среди моделей, прошедших проверку: AP выше prior, recall ≥ 0.5, precision ≥ 0.2, предсказания совпадают после сохранения/загрузки. Это учебная проверка допуска, а не производственный SLA.

После фиксации выбора test оценивается один раз в отдельном Run. Test не используется для выбора модели или порога. В Registry регистрируются два прошедших проверку кандидата; лучшему назначается `champion`. Артефакт модели включает preprocessing, native sklearn/skops или CatBoost, код загрузчика и порог.

AI4I — синтетические данные. Результаты IID-разбиения не доказывают качество на реальном оборудовании; для этого нужны временная/групповая проверка, стоимость ошибок и мониторинг дрейфа.

## Что показать в MLflow

В Experiment `predictiveguard-ai4i` находятся EDA, четыре model-development Runs и отдельный финальный test Run. Выберите четыре модельных Runs в UI и нажмите Compare; сопоставьте `validation_average_precision`, F1, precision и recall.

В артефактах EDA сохранены графики, статистика, выводы и исходный CSV. В модельных Runs — диагностические графики и результат проверки допуска. Dataset Tracking связывает фактические выборки с контекстами `eda`, `training`, `validation`, `test`.

- **source** — URI исходного CSV; его точная копия дополнительно сохранена в EDA artifacts;
- **digest** — вычисленный MLflow отпечаток конкретной таблицы; у разных splits разные digest;
- **lineage** — `lineage/split_manifest.json`: SHA256 исходных байтов, seed, индексы UDI каждого split, переименование, исключённые признаки и preprocessing. Digest MLflow и SHA256 файла имеют разные назначения.

Metadata Runs, datasets и Registry хранятся в отдельной PostgreSQL `mlflow-db` (Backend Store). Модели, исходный CSV и графики — в отдельном volume `mlflow_artifacts` (Artifact Store), доступном через сервер MLflow. Перезапуск контейнеров сохраняет эти данные. `app-db` обслуживает проверку PostgreSQL в API. Для логов настроена ротация; контейнеры имеют healthcheck, API — лимиты CPU и памяти.

## Проверка API и смены alias

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/api/v1/version
curl http://127.0.0.1:8000/api/v1/health
curl -X POST http://127.0.0.1:8000/process \
  -H 'Content-Type: application/json' \
  -d '{"type":"L","air_temperature_k":298.1,"process_temperature_k":308.6,"rotational_speed_rpm":1551,"torque_nm":42.8,"tool_wear_min":0}'
```

Ответ содержит решение, вероятность отказа, имя модели, alias, фактически загруженную версию и Run ID. `type` обязателен: L/M/H. Значения датчиков проверяются по диапазонам, неизвестные поля и нечисловые значения отклоняются.

На защите в Models → `PredictiveGuardFailureModel` переназначьте alias `champion` на другую зарегистрированную версию. Запрос до перезапуска продолжает использовать прежнюю версию. Затем:

```bash
docker compose restart app
```

Повторный запрос должен показать новую версию без изменения inference-кода. Возврат alias и повторный restart демонстрируют rollback. Приложение продолжает обслуживать запросы уже загруженной моделью даже при временной недоступности MLflow; для нового startup сервер нужен.

Ограничение: смена alias сама по себе не обновляет работающие экземпляры, а restart создаёт окно недоступности и может оставить экземпляры на разных версиях. В production используйте управляемый rollout с закреплённой версией, readiness, проверкой качества и rollback.

`GET /healthz` — liveness; `GET /api/v1/health` проверяет PostgreSQL (503 при недоступности); `GET /api/v1/version` возвращает версию пакета. Startup завершается только после загрузки модели. HTTP-ответы содержат `X-Request-ID` и `X-Process-Time-Ms`, логи имеют JSON-формат.

## Локальный запуск API

При работающем MLflow и существующем champion:

```bash
uv run uvicorn predictiveguard.main:app --reload
```

`.env.example` задаёт адрес MLflow для запуска с хоста; Compose задаёт свои внутренние адреса. PostgreSQL Compose не публикует порт: для локальной проверки `/api/v1/health` нужен доступный PostgreSQL и соответствующие `POSTGRES_*`.

## Проверки качества и CI/CD

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run --all-files
```

Тесты проверяют разделение и происхождение данных, отсутствие утечек, выбор по validation, реальное сохранение/загрузку sklearn и CatBoost, Registry, смену alias, startup FastAPI и inference. Branch coverage сервисного кода должен быть ≥ 90%; полноценное обучение дополнительно проверяется исполнением ноутбука.

CI на push/PR в `main` выполняет lint, format, pytest и pre-commit. CD сохранён в варианте, согласованном с преподавателем: установка зависимостей, сверка релизного тега и публикация образа; отдельный шаг quality checks в CD отсутствует.

## Версионирование и публикация образа

В проекте используется семантическое версионирование: `MAJOR.MINOR.PATCH`.

Версия приложения берётся из `project.version` в `pyproject.toml`. Каждый push в `main` запускает отдельные CI и CD: CD публикует образ с тегами `sha-<первые 12 символов SHA коммита>` и `latest`. Так можно показать полный цикл после небольшого изменения на защите без создания релизного тега.

Для релиза увеличьте версию в `pyproject.toml`, обновите `uv.lock` командой `uv lock`, закоммитьте и отправьте изменения в `main`. Затем создайте новый тег с той же версией и отправьте его отдельно. Например, для версии `0.1.1`:

```bash
git tag v0.1.1
git push origin v0.1.1
```

Push тега `v*.*.*` запускает CD повторно. Он сверяет тег с версией проекта и публикует релизный образ в GitHub Container Registry:

```text
ghcr.io/<owner>/<repository>:v0.1.1
```

При push в `main` публикуются:

```text
ghcr.io/<owner>/<repository>:sha-<первые 12 символов SHA коммита>
ghcr.io/<owner>/<repository>:latest
```

Несовместимые изменения API увеличивают `MAJOR`, обратно совместимая функциональность увеличивает `MINOR`, исправления увеличивают `PATCH`.

## Структура

- [notebooks/ai4i_research.ipynb](notebooks/ai4i_research.ipynb) — EDA, обучение, сравнение, финальная оценка и Registry;
- [ml/train.py](ml/train.py) — общие шаги ноутбука и ручного trainer;
- [data/ai4i2020.csv](data/ai4i2020.csv) — исходные данные;
- `src/predictiveguard/inference.py` — MLflow loader, estimator и порог;
- `src/predictiveguard/model_service.py` — закрепление версии при startup и inference;
- `src/predictiveguard/api/routes.py` — асинхронные HTTP-эндпоинты;
- `docker-compose.yml` — MLflow, отдельные хранилища, API и ручной trainer.
