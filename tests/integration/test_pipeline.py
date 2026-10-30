import asyncio
import base64
import json
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import aio_pika
import pytest
from fastapi import FastAPI
from faststream.rabbit import RabbitMessage
from httpx import ASGITransport, AsyncClient

from predictiveguard.api.tasks import router
from predictiveguard.queue import Overloaded, declare, make_broker, publish, publish_wait, topology
from predictiveguard.task_store import TaskStore
from predictiveguard.worker import Worker

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(real_config):
    value = TaskStore(real_config)
    await value.start()
    yield value
    await value.close()


async def test_real_overflow_and_api_rollback(real_config, store, task):
    async with make_broker(real_config) as broker:
        await declare(broker, real_config)
        await publish(broker, real_config, task)
        await publish(broker, real_config, task.model_copy(update={"task_id": uuid4()}))
        with pytest.raises(Overloaded):
            await publish(broker, real_config, task)
        app = FastAPI()
        app.include_router(router)
        app.state.store, app.state.broker, app.state.config = store, broker, real_config
        key = uuid4()
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/tasks",
                json=task.features.model_dump(),
                headers={"Idempotency-Key": str(key)},
            )
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "1"
        assert await store.get(key) is None


async def test_duplicate_is_atomic(real_config, store, task, prediction):
    model = Mock(predict=Mock(return_value=prediction))
    worker = Worker(real_config, model=model, store=store)
    async with worker.broker:
        await declare(worker.broker, real_config)
        messages = [AsyncMock(spec=RabbitMessage), AsyncMock(spec=RabbitMessage)]
        await asyncio.gather(*(worker.handle(task.wire(), msg) for msg in messages))
        await worker.handle(task.wire(), AsyncMock(spec=RabbitMessage))
    row = await store.get(task.task_id)
    assert row["status"] == "SUCCEEDED" and row["attempts"] == 1
    assert model.predict.call_count == 1
    assert row["result"] == prediction.model_dump()
    assert all(msg.ack.await_count == 1 for msg in messages)
    worker.executor.shutdown()


async def test_error_budget_and_invalid_to_real_dlq(real_config, store, task):
    model = Mock(predict=Mock(side_effect=RuntimeError("broken")))
    worker = Worker(real_config, model=model, store=store)
    async with worker.broker:
        await declare(worker.broker, real_config)
        await worker.handle(task.wire(), AsyncMock(spec=RabbitMessage))
        await worker.handle({"bad": "schema"}, AsyncMock(spec=RabbitMessage))
        row = await store.get(task.task_id)
        assert row["status"] == "FAILED" and row["attempts"] == 3
        queue = await worker.broker.declare_queue(topology(real_config)[1])
        await queue.declare()
        assert queue.declaration_result.message_count == 2
    worker.executor.shutdown()


async def test_source_unblocks_when_consumer_frees_capacity(real_config, task):
    async with make_broker(real_config) as broker:
        await declare(broker, real_config)
        await publish(broker, real_config, task)
        await publish(broker, real_config, task)
        pending = asyncio.create_task(publish_wait(broker, real_config, task))
        await asyncio.sleep(0.1)
        assert not pending.done()
        queue = await broker.declare_queue(topology(real_config)[0])
        message = await queue.get(timeout=2)
        await message.ack()
        await asyncio.wait_for(pending, 3)


async def test_api_accepted_lookup_idempotency_and_conflict(real_config, store, task):
    async with make_broker(real_config) as broker:
        await declare(broker, real_config)
        app = FastAPI()
        app.include_router(router)
        app.state.store, app.state.broker, app.state.config = store, broker, real_config
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            headers = {"Idempotency-Key": str(task.task_id)}
            first = await client.post(
                "/api/v1/tasks", json=task.features.model_dump(), headers=headers
            )
            assert first.status_code == 202
            second = await client.post(
                "/api/v1/tasks", json=task.features.model_dump(), headers=headers
            )
            assert second.json() == first.json()
            assert (await client.get(first.headers["Location"])).json()["status"] == "PENDING"
            changed = task.features.model_dump() | {"torque_nm": 99}
            assert (
                await client.post("/api/v1/tasks", json=changed, headers=headers)
            ).status_code == 409
            assert (await client.get(f"/api/v1/tasks/{uuid4()}")).status_code == 404


async def test_worker_shutdown_and_crash_recovery(real_config, store, task, prediction, tmp_path):
    import os
    import signal
    import socket
    import subprocess
    import sys

    import aiohttp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = real_config.model_copy(
        update={
            "inference_delay_seconds": 1.5,
            "worker_health_port": port,
        }
    )
    boot = """
import asyncio, json, sys
from unittest.mock import Mock
from predictiveguard.config import Settings
from predictiveguard.schemas import ProcessResponse
from predictiveguard.worker import Worker
config = Settings.model_validate(json.loads(sys.argv[1]))
result = ProcessResponse.model_validate(json.loads(sys.argv[2]))
worker = Worker(config, model=Mock(predict=Mock(return_value=result)))
asyncio.run(worker.app.run())
"""
    processes = []
    log = (tmp_path / "worker.log").open("w")

    def start():
        process = subprocess.Popen(
            [sys.executable, "-c", boot, config.model_dump_json(), prediction.model_dump_json()],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
        )
        processes.append(process)
        return process

    async def wait_for(predicate, wait_seconds=30):
        async with asyncio.timeout(wait_seconds):
            for _ in range(int(wait_seconds / 0.05) + 1):
                if await predicate():
                    return
                await asyncio.sleep(0.05)

    async def is_running(task_id):
        row = await store.get(task_id)
        return row is not None and row["status"] == "RUNNING"

    async def is_done(task_id):
        row = await store.get(task_id)
        return row is not None and row["status"] == "SUCCEEDED"

    async def healthy():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/healthz") as response:
                    return response.status == 200
        except aiohttp.ClientError:
            return False

    try:
        process = start()
        await wait_for(healthy)
        async with make_broker(config) as broker:
            await publish(broker, config, task)
            await wait_for(lambda: is_running(task.task_id))
            assert await healthy()
            process.send_signal(signal.SIGTERM)
            await asyncio.wait_for(asyncio.to_thread(process.wait), 15)
            assert process.returncode == 0
            assert await is_done(task.task_id)
            assert (await store.get(task.task_id))["attempts"] == 1

            crashed_task = task.model_copy(update={"task_id": uuid4()})
            process = start()
            await wait_for(healthy)
            await publish(broker, config, crashed_task)
            await wait_for(lambda: is_running(crashed_task.task_id))
            process.kill()
            await asyncio.to_thread(process.wait)
            process = start()
            await wait_for(healthy)
            await wait_for(lambda: is_done(crashed_task.task_id))
            assert (await store.get(crashed_task.task_id))["attempts"] == 2
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                await asyncio.wait_for(asyncio.to_thread(process.wait), 15)
        log.close()


async def test_confirmation_timeout_is_recoverable(
    real_config, store, task, prediction, monkeypatch
):
    app = FastAPI()
    app.include_router(router)
    async with make_broker(real_config) as broker:
        await declare(broker, real_config)
        app.state.store, app.state.broker, app.state.config = store, broker, real_config

        async def accepted_without_confirmation(*args):
            await publish(*args)
            raise TimeoutError("lost confirmation")

        monkeypatch.setattr("predictiveguard.api.tasks.publish", accepted_without_confirmation)
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/tasks",
                json=task.features.model_dump(),
                headers={"Idempotency-Key": str(task.task_id)},
            )
            assert response.status_code == 503
            assert (await store.get(task.task_id))["status"] == "UNKNOWN"
            worker = Worker(real_config, Mock(predict=Mock(return_value=prediction)), store)
            await worker.handle(task.wire(), AsyncMock(spec=RabbitMessage))
            second = await client.post(
                "/api/v1/tasks",
                json=task.features.model_dump(),
                headers={"Idempotency-Key": str(task.task_id)},
            )
            assert second.status_code == 202 and second.json()["status"] == "SUCCEEDED"
            assert (await store.get(task.task_id))["attempts"] == 1
            worker.executor.shutdown()


async def test_actual_api_lifespan_without_loading_model(real_config, monkeypatch):
    from predictiveguard.main import app, lifespan

    monkeypatch.setattr("predictiveguard.main.settings", real_config)
    async with lifespan(app):
        assert app.state.store.pool is not None
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            assert (await client.get("/healthz")).status_code == 200


@pytest.mark.parametrize("concurrency", [1, 2])
async def test_real_prefetch_does_not_move_backlog_into_worker(
    real_config, store, task, prediction, concurrency
):
    config = real_config.model_copy(
        update={
            "inference_delay_seconds": 0.5,
            "queue_max_length": 10,
            "prefetch_count": concurrency,
            "worker_concurrency": concurrency,
        }
    )
    worker = Worker(config, Mock(predict=Mock(return_value=prediction)), store)
    async with worker.broker:
        await declare(worker.broker, config)
        await worker.broker.start()
        for _ in range(4):
            await publish(worker.broker, config, task.model_copy(update={"task_id": uuid4()}))
        await asyncio.sleep(0.1)
        queue = await worker.broker.declare_queue(topology(config)[0])
        await queue.declare()
        assert worker.active == concurrency
        assert queue.declaration_result.message_count == 4 - concurrency
    worker.executor.shutdown()


async def _check_invalid_transport_message(
    config, store, task, prediction, body, content_type, retry_dlq=False
):
    model = Mock(predict=Mock(return_value=prediction))
    worker = Worker(config, model, store)
    if retry_dlq:
        original_dead_letter = worker.dead_letter
        attempts = 0

        async def transient_dead_letter(raw, reason):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary DLQ outage")
            await original_dead_letter(raw, reason)

        worker.dead_letter = AsyncMock(side_effect=transient_dead_letter)
    try:
        async with worker.broker:
            await declare(worker.broker, config)
            await worker.broker.start()
            connection = await aio_pika.connect_robust(config.rabbit_url)
            async with connection:
                channel = await connection.channel(publisher_confirms=True)
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=body,
                        content_type=content_type,
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=config.task_queue,
                    mandatory=True,
                )
            await publish(worker.broker, config, task)
            for _ in range(500):
                row = await store.get(task.task_id)
                if row is not None and row["status"] == "SUCCEEDED":
                    break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("Invalid delivery blocked the following valid task")
            assert row["attempts"] == 1
            assert row["result"] == prediction.model_dump()
            assert model.predict.call_count == 1
            dlq = await worker.broker.declare_queue(topology(config)[1])
            letter = await dlq.get(timeout=3)
            envelope = json.loads(letter.body)
            assert envelope["original"]["encoding"] == "base64"
            assert base64.b64decode(envelope["original"]["body"]) == body
            assert envelope["reason"]
            await letter.ack()
            await dlq.declare()
            assert dlq.declaration_result.message_count == 0
            queue = await worker.broker.declare_queue(topology(config)[0])
            await queue.declare()
            assert queue.declaration_result.message_count == 0
            if retry_dlq:
                assert worker.dead_letter.await_count == 2
    finally:
        worker.executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "body,content_type",
    [
        (b'{"broken":', "application/json"),
        (b"\xffnot-json", "application/json; charset=utf-8"),
        (b"\xffnot-text", "text/plain"),
        (b"[" * 1100, "application/json"),
    ],
)
async def test_real_decode_error_goes_to_dlq_without_blocking_next_task(
    real_config, store, task, prediction, body, content_type
):
    await _check_invalid_transport_message(real_config, store, task, prediction, body, content_type)


async def test_real_decode_error_survives_temporary_dlq_failure(
    real_config, store, task, prediction
):
    await _check_invalid_transport_message(
        real_config, store, task, prediction, b'{"broken":', "application/json", retry_dlq=True
    )
