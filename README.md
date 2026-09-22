# PredictiveGuard AI

Асинхронный FastAPI-сервис с PostgreSQL, проверками качества, контейнеризацией и публикацией версионированных образов в GitHub Container Registry.

## Требования

- Python 3.12
- uv 0.11.7
- Docker с Docker Compose

## Установка

```bash
uv sync --frozen
uv run pre-commit install
```

## Запуск локально

```bash
cp .env.example .env
uv run uvicorn predictiveguard.main:app --reload
```

Swagger доступен по адресу `http://127.0.0.1:8000/docs`.

## Эндпоинты

- `GET /healthz` возвращает liveness приложения без обращения к внешним компонентам.
- `GET /api/v1/version` возвращает версию установленного пакета из `pyproject.toml`.
- `GET /api/v1/health` проверяет PostgreSQL и возвращает его статус, версию и время ответа. При недоступности зависимости возвращается HTTP 503 и статус `degraded`.

Каждый HTTP-ответ содержит `X-Request-ID` и `X-Process-Time-Ms`. Входящий `X-Request-ID` сохраняется, а при его отсутствии приложение создаёт UUID. Логи выводятся в JSON и содержат уровень, логгер, request ID, HTTP-метод, путь, статус и длительность запроса.

## Проверки качества

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run --all-files
```

Pytest измеряет branch coverage и завершает проверку с ошибкой при покрытии ниже 90%.

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

После запуска доступны приложение на порту 8000 и PostgreSQL во внутренней сети Compose. Для обоих сервисов настроены healthcheck, restart policy, лимиты ресурсов и ротация логов. Данные PostgreSQL сохраняются в именованном volume.

Проверка стека:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/api/v1/version
curl http://127.0.0.1:8000/api/v1/health
```

## CI

На каждый push и pull request в `main` GitHub Actions выполняет установку строго по `uv.lock`, lint, format-check, тесты с coverage и полный набор pre-commit hooks.

## Версионирование и публикация образа

В проекте используется семантическое версионирование: `MAJOR.MINOR.PATCH`.

Версия приложения берётся из `project.version` в `pyproject.toml`. Каждый push в `main` запускает CI и CD: после проверок CD публикует образ с тегами `sha-<первые 12 символов SHA коммита>` и `latest`. Так можно показать полный цикл после небольшого изменения на защите без создания релизного тега.

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

```text
src/predictiveguard/
├── api/routes.py
├── config.py
├── db.py
├── logging_config.py
├── main.py
├── schemas.py
└── version.py
```

Роуты отвечают за HTTP-контракты, `db.py` выполняет асинхронную проверку PostgreSQL с таймаутами, `schemas.py` описывает модели ответов, а `logging_config.py` настраивает структурированные логи.
