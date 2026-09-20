FROM python:3.12-slim

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY requirements/web.txt ./requirements/web.txt
RUN pip install --no-cache-dir -r requirements/web.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "predictiveguard.main:app", "--host", "0.0.0.0", "--port", "8000"]
