import os
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from uuid import UUID

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from predictiveguard.version import get_app_version

batch_id_context: ContextVar[str | None] = ContextVar("batch_id", default=None)
task_id_context: ContextVar[str | None] = ContextVar("task_id", default=None)
propagator = TraceContextTextMapPropagator()


def resource(service):
    return Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME") or service,
            "service.version": os.getenv("OTEL_SERVICE_VERSION") or get_app_version(),
            "deployment.environment.name": os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT") or "local",
            "service.instance.id": os.getenv("OTEL_SERVICE_INSTANCE_ID") or socket.gethostname(),
        }
    )


def otlp_enabled(signal):
    return os.getenv("OTEL_SDK_DISABLED", "").lower() != "true" and bool(
        os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


@lru_cache(maxsize=1)
def configure_tracing(service):
    if otlp_enabled("TRACES"):
        provider = TracerProvider(resource=resource(service))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)


def trace_headers():
    headers = {}
    propagator.inject(headers)
    return headers


def error(span, exc):
    safe_exception = RuntimeError(type(exc).__name__).with_traceback(exc.__traceback__)
    span.record_exception(safe_exception, attributes={"exception.type": type(exc).__name__})
    span.set_status(Status(StatusCode.ERROR, type(exc).__name__))


@contextmanager
def span(name, **kwargs):
    tracer = trace.get_tracer("predictiveguard")
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False, **kwargs
    ) as current:
        try:
            yield current
        except Exception as exc:
            error(current, exc)
            raise


def uuid_text(value):
    try:
        return str(UUID(str(value)))
    except ValueError:
        return None


@contextmanager
def task_context(value):
    token = task_id_context.set(uuid_text(value))
    try:
        yield
    finally:
        task_id_context.reset(token)


@contextmanager
def batch_context(value):
    token = batch_id_context.set(uuid_text(value))
    try:
        yield
    finally:
        batch_id_context.reset(token)
