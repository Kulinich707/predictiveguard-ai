import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from opentelemetry.trace import SpanKind, Status, StatusCode
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.exposition import choose_encoder

from predictiveguard.api.routes import router
from predictiveguard.api.tasks import router as tasks_router
from predictiveguard.config import settings
from predictiveguard.logging_config import bind_request_id, configure_logging, reset_request_id
from predictiveguard.metrics import api_duration, api_registry, api_requests, exemplar, queue_limit
from predictiveguard.queue import declare, make_broker
from predictiveguard.task_store import TaskStore
from predictiveguard.telemetry import configure_tracing, propagator, span, task_context
from predictiveguard.version import get_app_version

configure_logging(settings.log_level)
configure_tracing("pg-api")
queue_limit.labels(settings.task_queue).set(settings.queue_max_length)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    store = TaskStore(settings)
    broker = make_broker(settings)
    await store.start()
    try:
        async with broker:
            await declare(broker, settings)
            application.state.store = store
            application.state.broker = broker
            application.state.config = settings
            yield
    finally:
        logger.info("API shutdown started", extra={"event": "shutdown_started"})
        await store.close()
        logger.info("API shutdown completed", extra={"event": "shutdown_finished"})


app = FastAPI(title="PredictiveGuard AI", version=get_app_version(), lifespan=lifespan)
app.include_router(router)
app.include_router(tasks_router)


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    encode, content_type = choose_encoder(request.headers.get("accept", CONTENT_TYPE_LATEST))
    return Response(content=encode(api_registry), headers={"Content-Type": content_type})


@app.middleware("http")
async def log_request(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    if request.url.path == "/healthz":
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID") or str(uuid4())
        response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started) * 1000:.3f}"
        return response
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    token = bind_request_id(request_id)
    started = perf_counter()
    status_code = 500
    try:
        with span(
            request.method,
            context=propagator.extract(request.headers),
            kind=SpanKind.SERVER,
            attributes={"http.request.method": request.method},
        ) as current:
            try:
                response = await call_next(request)
                status_code = response.status_code
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started) * 1000:.3f}"
                trace_context = current.get_span_context()
                request_trace_id = (
                    f"{trace_context.trace_id:032x}" if trace_context.is_valid else None
                )
                task_trace_id = getattr(request.state, "task_trace_id", None)
                if trace_id := task_trace_id or request_trace_id:
                    response.headers["X-Trace-ID"] = trace_id
                if task_trace_id and request_trace_id and request_trace_id != task_trace_id:
                    response.headers["X-Request-Trace-ID"] = request_trace_id
                return response
            finally:
                route = getattr(request.scope.get("route"), "path", "unmatched")
                elapsed = perf_counter() - started
                current.update_name(f"{request.method} {route}")
                current.set_attributes(
                    {"http.route": route, "http.response.status_code": status_code}
                )
                if status_code >= 500:
                    current.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
                method = (
                    request.method
                    if request.method in {"GET", "POST", "HEAD", "OPTIONS"}
                    else "OTHER"
                )
                api_requests.labels(method, route, str(status_code)).inc()
                api_duration.labels(method, route).observe(elapsed, exemplar=exemplar())
                with task_context(getattr(request.state, "task_id", None)):
                    logger.log(
                        logging.DEBUG
                        if route in {"/healthz", "/api/v1/health"} and status_code < 500
                        else logging.INFO,
                        "HTTP request completed",
                        extra={
                            "method": method,
                            "route": route,
                            "status_code": status_code,
                            "duration_ms": round(elapsed * 1000, 3),
                        },
                    )
    finally:
        reset_request_id(token)
