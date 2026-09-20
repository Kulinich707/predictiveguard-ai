# PredictiveGuard AI

## Настройка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
pip install -e .
pre-commit install
```

## Запуск локально

```bash
uvicorn predictiveguard.main:app --reload
```

## Эндпоинты

- `GET /healthz` — быстрая проверка работоспособности (liveness check) приложения.
- `GET /api/v1/version` — версия приложения.
- `GET /api/v1/health` — комплексная проверка работоспособности (health check) сторонних компонентов с указанием их версий и времени отклика.

## Тесты

```bash
pytest
```

## Линтер и форматер

```bash
ruff check .
ruff format --check .
```

## Docker Compose

```bash
docker compose up --build
```

## Версионирование и CD

В проекте используется семантическое версионирование: `MAJOR.MINOR.PATCH`.

Версия приложения определяется единственным источником истины: полем `project.version` в файле `pyproject.toml`.
Эндпоинт `GET /api/v1/version` считывает метаданные установленного пакета, сформированные на основе этого значения.

Тег релиза должен точно соответствовать версии проекта:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Процесс CD проверяет, что тег `v0.1.0` совпадает со значением `0.1.0` из `pyproject.toml`.
После этого выполняется сборка и отправка (push) Docker-образа в GitHub Container Registry:

```text
ghcr.io/<owner>/<repository>:v0.1.0
ghcr.io/<owner>/<repository>:latest
```

Семантическое версионирование используется, так как сервис имеет публичный API: несовместимые изменения API
увеличивают версию `MAJOR`, добавление функциональности с сохранением обратной совместимости увеличивает `MINOR`,
а исправления с сохранением обратной совместимости увеличивают `PATCH`.
