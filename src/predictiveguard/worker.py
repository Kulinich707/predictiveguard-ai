import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

from aiohttp import web
from faststream import FastStream
from faststream.middlewares import AckPolicy
from faststream.rabbit import Channel, RabbitMessage
from pydantic import ValidationError

from predictiveguard.config import settings
from predictiveguard.model_service import ModelService
from predictiveguard.queue import declare, make_broker, topology
from predictiveguard.task_store import TaskStore
from predictiveguard.tasks import TaskMessage

logger = logging.getLogger(__name__)


async def decode_message(message: RabbitMessage, original_decoder):
    try:
        return await original_decoder(message)
    except (ValueError, RecursionError):
        return message.body


class Worker:
    def __init__(self, config, model=None, store=None):
        self.config = config
        self.model = model or ModelService()
        self.store = store or TaskStore(config)
        self.broker = make_broker(config)
        self.executor = ThreadPoolExecutor(max_workers=config.worker_concurrency)
        self.slots = asyncio.Semaphore(config.worker_concurrency)
        self.health_runner = None
        self.active = 0
        self.loaded = False
        self.broker.subscriber(
            topology(config)[0],
            ack_policy=AckPolicy.MANUAL,
            decoder=decode_message,
            channel=Channel(
                prefetch_count=config.prefetch_count,
                publisher_confirms=True,
                on_return_raises=True,
            ),
        )(self.handle)
        self.app = FastStream(self.broker)
        self.app.on_startup(self.start)
        self.app.after_shutdown(self.close)

    async def start(self):
        await self.store.start()
        await asyncio.get_running_loop().run_in_executor(
            self.executor,
            self.model.load,
            self.config,
        )
        self.loaded = True
        await self.broker.connect()
        await declare(self.broker, self.config)
        health = web.Application()
        health.router.add_get("/healthz", self.health)
        self.health_runner = web.AppRunner(health)
        await self.health_runner.setup()
        await web.TCPSite(self.health_runner, "0.0.0.0", self.config.worker_health_port).start()

    async def health(self, request):
        return web.json_response({"ready": self.loaded, "active": self.active})

    async def close(self):
        if self.health_runner:
            await self.health_runner.cleanup()
        await self.store.close()
        self.executor.shutdown(wait=True)

    async def dead_letter(self, raw, reason):
        if isinstance(raw, bytes):
            raw = {"encoding": "base64", "body": base64.b64encode(raw).decode("ascii")}
        await self.broker.publish(
            {"original": raw, "reason": reason},
            queue=topology(self.config)[1],
            persist=True,
            mandatory=True,
            timeout=self.config.publish_timeout_seconds,
        )

    async def handle(self, raw: Any, message: RabbitMessage):
        try:
            task = TaskMessage.model_validate(raw)
        except (ValidationError, TypeError, ValueError) as exc:
            try:
                await self.dead_letter(raw, str(exc))
                await message.ack()
            except Exception:
                await asyncio.sleep(self.config.retry_delay_seconds)
                await message.nack(requeue=True)
            return
        try:
            async with self.slots, self.store.locked(task.task_id) as conn:
                self.active += 1
                try:
                    await self.process(conn, task, raw)
                    await message.ack()
                finally:
                    self.active -= 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Infrastructure failure; keep delivery recoverable")
            await asyncio.sleep(self.config.retry_delay_seconds)
            await message.nack(requeue=True)

    async def process(self, conn, task, raw):
        try:
            row = await self.store.ensure(conn, task)
        except ValueError as exc:
            await self.dead_letter(raw, str(exc))
            return
        if row["status"] == "SUCCEEDED":
            return
        attempts = row["attempts"]
        error = row["error"] or "attempt budget exhausted (possibly interrupted worker)"
        while attempts < self.config.max_attempts:
            attempts = await self.store.attempt(conn, task.task_id)
            try:
                started = perf_counter()
                result = await asyncio.get_running_loop().run_in_executor(
                    self.executor,
                    self.model.predict,
                    task.features,
                )
                inference = perf_counter() - started
                delay_started = perf_counter()
                await asyncio.sleep(self.config.inference_delay_seconds)
                delay = perf_counter() - delay_started
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                await self.store.state(conn, task.task_id, "RUNNING", error)
                await asyncio.sleep(self.config.retry_delay_seconds)
                continue
            await self.store.success(conn, task, result, inference, delay)
            return
        await self.store.state(conn, task.task_id, "FAILED", error)
        await self.dead_letter(raw, error)


worker = Worker(settings)
app = worker.app


def main():
    logging.basicConfig(level=settings.log_level)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
