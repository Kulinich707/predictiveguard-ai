import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from faststream.rabbit import TestRabbitBroker

from predictiveguard.queue import Overloaded, publish_wait, topology
from predictiveguard.worker import Worker


class MemoryStore:
    def __init__(self):
        self.row = None
        self.start = AsyncMock()
        self.close = AsyncMock()

    @asynccontextmanager
    async def locked(self, task_id):
        yield None

    async def ensure(self, conn, task):
        if self.row is None:
            self.row = {"attempts": 0, "status": "PENDING", "error": None}
        return self.row

    async def attempt(self, conn, task_id):
        self.row["attempts"] += 1
        return self.row["attempts"]

    async def state(self, conn, task_id, state, error):
        self.row.update(status=state, error=error)

    async def success(self, conn, task, result, inference_seconds, delay_seconds):
        self.row.update(status="SUCCEEDED", result=result)


async def test_broker_handler_success_and_duplicate(config, task, prediction):
    store = MemoryStore()
    model = Mock(predict=Mock(return_value=prediction))
    worker = Worker(config, model, store)
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(task.wire(), queue=topology(config)[0])
        await broker.publish(task.wire(), queue=topology(config)[0])
    assert store.row["result"] == prediction
    model.predict.assert_called_once()
    worker.executor.shutdown()


@pytest.mark.parametrize("raw", [{"bad": 1}, [1, 2], "invalid json"])
async def test_broker_invalid_message_to_dlq(config, raw):
    worker = Worker(config, Mock(), MemoryStore())
    received = []

    @worker.broker.subscriber(topology(config)[1])
    async def dlq(body: dict):
        received.append(body)

    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(raw, queue=topology(config)[0])
    assert received[0]["original"] == raw
    worker.executor.shutdown()


async def test_broker_retry_budget(config, task):
    store = MemoryStore()
    model = Mock(predict=Mock(side_effect=ValueError("model failure")))
    worker = Worker(config, model, store)
    received = []

    @worker.broker.subscriber(topology(config)[1])
    async def dlq(body: dict):
        received.append(body)

    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(task.wire(), queue=topology(config)[0])
    assert model.predict.call_count == config.max_attempts
    assert store.row["status"] == "FAILED"
    assert len(received) == 1
    worker.executor.shutdown()


async def test_predict_does_not_block_loop(config, task, prediction):
    import time

    def predict(features):
        time.sleep(0.1)
        return prediction

    worker = Worker(config, Mock(predict=predict), MemoryStore())
    processing = asyncio.create_task(worker.handle(task.wire(), AsyncMock()))
    await asyncio.sleep(0.02)
    assert not processing.done()
    assert (await worker.health(None)).status == 200
    await processing
    worker.executor.shutdown()


async def test_dlq_failure_never_acks_original(config, task):
    worker = Worker(config, Mock(), MemoryStore())
    worker.dead_letter = AsyncMock(side_effect=TimeoutError())
    message = AsyncMock()
    await worker.handle({}, message)
    message.ack.assert_not_awaited()
    message.nack.assert_awaited_once_with(requeue=True)
    worker.executor.shutdown()


async def test_source_retries_same_task(config, task, monkeypatch):
    publisher = AsyncMock(side_effect=[Overloaded(), TimeoutError(), None])
    monkeypatch.setattr("predictiveguard.queue.publish", publisher)
    await publish_wait(None, config, task)
    assert publisher.await_count == 3
    assert all(call.args[2] is task for call in publisher.await_args_list)


async def test_worker_start_loads_once_and_closes_resources(config, prediction, monkeypatch):
    worker = Worker(config, Mock(predict=Mock(return_value=prediction)), MemoryStore())
    worker.broker.connect = AsyncMock()
    monkeypatch.setattr("predictiveguard.worker.declare", AsyncMock())
    runner = Mock(setup=AsyncMock(), cleanup=AsyncMock())
    monkeypatch.setattr("predictiveguard.worker.web.AppRunner", Mock(return_value=runner))
    site = Mock(start=AsyncMock())
    monkeypatch.setattr("predictiveguard.worker.web.TCPSite", Mock(return_value=site))
    await worker.start()
    assert worker.loaded
    worker.model.load.assert_called_once_with(config)
    await worker.close()
    worker.store.close.assert_awaited_once()
    runner.cleanup.assert_awaited_once()


async def test_infrastructure_failure_requeues(config, task):
    worker = Worker(config, Mock(), MemoryStore())
    worker.process = AsyncMock(side_effect=ConnectionError("db down"))
    message = AsyncMock()
    await worker.handle(task.wire(), message)
    message.ack.assert_not_awaited()
    message.nack.assert_awaited_once_with(requeue=True)
    worker.executor.shutdown()


async def test_cancel_keeps_unacked_for_redelivery(config, task):
    worker = Worker(config, Mock(), MemoryStore())
    worker.process = AsyncMock(side_effect=asyncio.CancelledError())
    message = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await worker.handle(task.wire(), message)
    message.ack.assert_not_awaited()
    worker.executor.shutdown()


async def test_payload_collision_is_dead_lettered(config, task):
    store = MemoryStore()
    store.ensure = AsyncMock(side_effect=ValueError("collision"))
    worker = Worker(config, Mock(), store)
    worker.dead_letter = AsyncMock()
    message = AsyncMock()
    await worker.handle(task.wire(), message)
    worker.dead_letter.assert_awaited_once()
    message.ack.assert_awaited_once()
    worker.executor.shutdown()


async def test_invalid_binary_is_serializable_in_dlq(config):
    worker = Worker(config, Mock(), MemoryStore())
    received = []

    @worker.broker.subscriber(topology(config)[1])
    async def dlq(body: dict):
        received.append(body)

    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(b"\xffnot-json", queue=topology(config)[0])
    assert received[0]["original"]["encoding"] == "base64"
    worker.executor.shutdown()


async def test_naive_timestamp_goes_to_dlq_before_database(config, task):
    worker = Worker(config, Mock(), MemoryStore())
    received = []

    @worker.broker.subscriber(topology(config)[1])
    async def dlq(body: dict):
        received.append(body)

    raw = task.wire() | {"created_at": "2026-10-19T10:00:00"}
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(raw, queue=topology(config)[0])
    assert worker.store.row is None
    assert received[0]["original"] == raw
    worker.executor.shutdown()
