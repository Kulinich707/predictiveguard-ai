import asyncio
import math
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Request, Response

from predictiveguard.queue import Overloaded, publish
from predictiveguard.schemas import ProcessRequest
from predictiveguard.tasks import TaskMessage

router = APIRouter()


@router.post("/process", status_code=202, deprecated=True)
@router.post("/api/v1/tasks", status_code=202)
async def create_task(
    payload: ProcessRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[UUID | None, Header()] = None,
):
    task = TaskMessage(task_id=idempotency_key or uuid4(), features=payload)
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
                row = await store.ensure(conn, task, "SUBMITTING")
                if row["status"] not in ("SUBMITTING", "UNKNOWN"):
                    response.headers["Location"] = f"/api/v1/tasks/{task.task_id}"
                    return {"task_id": task.task_id, "status": row["status"]}
                try:
                    await publish(broker, config, task)
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


@router.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: UUID, request: Request):
    row = await request.app.state.store.get(task_id)
    if row is None:
        raise HTTPException(404, "Unknown task")
    row.pop("payload")
    return row
