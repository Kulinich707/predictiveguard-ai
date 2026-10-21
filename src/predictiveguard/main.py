import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from predictiveguard.api.routes import router
from predictiveguard.api.tasks import router as tasks_router
from predictiveguard.config import settings
from predictiveguard.logging_config import bind_request_id, configure_logging, reset_request_id
from predictiveguard.queue import declare, make_broker
from predictiveguard.task_store import TaskStore
from predictiveguard.version import get_app_version

configure_logging(settings.log_level)
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
        await store.close()


app = FastAPI(title="PredictiveGuard AI", version=get_app_version(), lifespan=lifespan)
app.include_router(router)
app.include_router(tasks_router)


@app.middleware("http")
async def log_request(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    token = bind_request_id(request_id)
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started) * 1000:.3f}"
        return response
    finally:
        duration_ms = (perf_counter() - started) * 1000
        logger.info(
            "HTTP request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )
        reset_request_id(token)
