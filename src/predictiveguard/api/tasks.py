import asyncio
import logging
import math
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Request, Response
from opentelemetry import trace
from opentelemetry.context import attach, detach

from predictiveguard.queue import Overloaded, publish
from predictiveguard.schemas import ProcessRequest
from predictiveguard.tasks import TaskMessage
from predictiveguard.telemetry import error, propagator, task_context, trace_headers

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/process", status_code=202, deprecated=True)
@router.post("/api/v1/tasks", status_code=202)
async def create_task(
    payload: ProcessRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[UUID | None, Header()] = None,
):
    task = TaskMessage(task_id=idempotency_key or uuid4(), features=payload)
    request.state.task_id = task.task_id
    with task_context(task.task_id):
        current = trace.get_current_span()
        current.set_attribute("task.id", str(task.task_id))
        try:
            result = await submit(task, request, response)
        except HTTPException as exc:
            if exc.status_code == 503:
                error(current, exc)
                logger.warning(
                    exc.detail,
                    extra={"event": "api_unavailable", "status_code": 503},
                )
            raise
        logger.info("Task submission accepted", extra={"event": "task_accepted"})
        return result


async def submit(task, request, response):
    store, broker, config = (
        request.app.state.store,
        request.app.state.broker,
        request.app.state.config,
    )
    retry = {
        "Retry-After": str(max(1, math.ceil(config.inference_delay_seconds))),
        "X-Task-ID": str(task.task_id),
    }

    try:
        async with asyncio.timeout(config.publish_timeout_seconds + 0.5):
            async with store.locked(task.task_id) as conn:
                row = await store.ensure(conn, task, "SUBMITTING", trace_context=trace_headers())
                task_trace = correlate_task_trace(request, row["trace_context"])
                if row["status"] not in ("SUBMITTING", "UNKNOWN"):
                    response.headers["Location"] = f"/api/v1/tasks/{task.task_id}"
                    return {"task_id": task.task_id, "status": row["status"]}
                try:
                    token = attach(task_trace)
                    try:
                        await publish(broker, config, task)
                    finally:
                        detach(token)
                except Overloaded:
                    if row["status"] == "SUBMITTING":
                        await conn.execute(
                            "DELETE FROM scoring_tasks WHERE task_id=$1",
                            task.task_id,
                        )
                    raise HTTPException(
                        503, "Queue is full; retry with same Idempotency-Key", headers=retry
                    ) from None
                except (TimeoutError, ConnectionError):
                    await store.state(conn, task.task_id, "UNKNOWN")
                    raise HTTPException(
                        503, "Confirmation uncertain; retry the same task_id", headers=retry
                    ) from None
                await store.state(conn, task.task_id, "PENDING")
    except TimeoutError:
        raise HTTPException(503, "Admission timeout; reuse X-Task-ID", headers=retry) from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    response.headers["Location"] = f"/api/v1/tasks/{task.task_id}"
    return {"task_id": task.task_id, "status": "PENDING"}


def correlate_task_trace(request, headers):
    task_trace = propagator.extract(headers or {})
    original = trace.get_current_span(task_trace).get_span_context()
    if original.is_valid:
        request.state.task_trace_id = f"{original.trace_id:032x}"
        current = trace.get_current_span()
        if current.get_span_context().trace_id != original.trace_id:
            current.add_link(original)
            current.set_attribute("task.trace_id", request.state.task_trace_id)
    return task_trace


@router.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: UUID, request: Request):
    request.state.task_id = task_id
    with task_context(task_id):
        row = await request.app.state.store.get(task_id)
    if row is None:
        raise HTTPException(404, "Unknown task")
    row.pop("payload")
    correlate_task_trace(request, row.pop("trace_context"))
    row.pop("deliveries")
    row.pop("dlq_confirmed")
    return row
