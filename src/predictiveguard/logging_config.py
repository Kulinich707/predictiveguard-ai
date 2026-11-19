import json
import logging
import os
import re
import sys
import traceback
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

from predictiveguard.telemetry import batch_id_context, task_id_context, uuid_text

request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
DSN_CREDENTIALS = re.compile(r"(://)[^/\s@]+@")


class JsonFormatter(logging.Formatter):
    def __init__(self, service="pg-api"):
        super().__init__()
        self.service = os.getenv("OTEL_SERVICE_NAME") or service
        self.env = os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT") or "local"
        self.secrets = [
            value
            for key, value in os.environ.items()
            if value and key.upper().endswith(("_PASSWORD", "_TOKEN", "_SECRET", "_API_KEY"))
        ]

    def format(self, record: logging.LogRecord) -> str:
        ctx = trace.get_current_span().get_span_context()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "service": self.service,
            "env": self.env,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": uuid_text(request_id_context.get()) or "-",
            "task_id": task_id_context.get(),
            "run_id": batch_id_context.get(),
            "trace_id": f"{ctx.trace_id:032x}" if ctx.is_valid else None,
            "span_id": f"{ctx.span_id:016x}" if ctx.is_valid else None,
        }
        for field in (
            "event",
            "method",
            "route",
            "status_code",
            "duration_ms",
            "component",
            "attempt",
            "delivery",
            "redelivered",
            "queue_wait_ms",
            "retry_seconds",
            "model_alias",
            "model_version",
            "run_id",
            "accepted",
            "retries",
            "worker",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info and record.exc_info[0]:
            exc_type, _, stack = record.exc_info
            frames = traceback.extract_tb(stack)
            payload["exception"] = {
                "type": exc_type.__name__,
                "stacktrace": [
                    f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in frames
                ],
            }
        for field in ("message", "request_id", "model_alias", "model_version"):
            value = payload.get(field)
            if isinstance(value, str):
                value = DSN_CREDENTIALS.sub(r"\1[redacted]@", value)
                for secret in self.secrets:
                    value = value.replace(secret, "[redacted]")
                payload[field] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str, service="pg-api") -> None:
    level = level.upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter(service))
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "faststream"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("aio_pika", "aiormq"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)


def bind_request_id(request_id: str) -> Token[str]:
    return request_id_context.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    request_id_context.reset(token)
