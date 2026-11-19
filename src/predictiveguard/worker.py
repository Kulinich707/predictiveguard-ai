import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from time import perf_counter, time_ns
from typing import Any

from aiohttp import web
from faststream import FastStream
from faststream.middlewares import AckPolicy
from faststream.rabbit import Channel, RabbitMessage
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from prometheus_client.exposition import choose_encoder
from pydantic import ValidationError

from predictiveguard.config import settings
from predictiveguard.logging_config import configure_logging
from predictiveguard.metrics import (
    exemplar,
    worker_active,
    worker_errors,
    worker_inference,
    worker_redelivered,
    worker_registry,
    worker_results,
    worker_task_latency,
)
from predictiveguard.model_service import ModelService
from predictiveguard.queue import declare, make_broker, topology
from predictiveguard.task_store import TaskStore
from predictiveguard.tasks import TaskMessage
from predictiveguard.telemetry import (
    batch_context,
    configure_tracing,
    error,
    propagator,
    resource,
    span,
    task_context,
    task_id_context,
    trace_headers,
    uuid_text,
)

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
        self.instance = resource("pg-worker").attributes["service.instance.id"]
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
        self.app.on_shutdown(self.begin_shutdown)
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
        health.router.add_get("/metrics", self.metrics)
        self.health_runner = web.AppRunner(health, access_log=None)
        await self.health_runner.setup()
        await web.TCPSite(self.health_runner, "0.0.0.0", self.config.worker_health_port).start()
        metrics_port = self.config.worker_metrics_port
        if metrics_port and metrics_port != self.config.worker_health_port:
            await web.TCPSite(self.health_runner, "0.0.0.0", metrics_port).start()

    async def health(self, request):
        return web.json_response({"ready": self.loaded, "active": self.active})

    async def metrics(self, request):
        encode, content_type = choose_encoder(request.headers.get("Accept", ""))
        return web.Response(body=encode(worker_registry), headers={"Content-Type": content_type})

    async def begin_shutdown(self):
        logger.info(
            "Worker shutdown started",
            extra={"event": "shutdown_started", "worker": self.instance},
        )

    async def close(self):
        if self.health_runner:
            await self.health_runner.cleanup()
        await self.store.close()
        self.executor.shutdown(wait=True)
        logger.info(
            "Worker shutdown completed",
            extra={"event": "shutdown_finished", "worker": self.instance},
        )

    async def dead_letter(self, raw, reason):
        if isinstance(raw, bytes):
            raw = {"encoding": "base64", "body": base64.b64encode(raw).decode("ascii")}
        with span("publish DLQ", kind=SpanKind.PRODUCER) as current:
            await self.broker.publish(
                {"original": raw, "reason": reason},
                queue=topology(self.config)[1],
                persist=True,
                correlation_id=task_id_context.get(),
                headers=trace_headers(),
                mandatory=True,
                timeout=self.config.publish_timeout_seconds,
            )
            current.set_status(Status(StatusCode.ERROR, "Task sent to DLQ"))
            logger.error("Task sent to DLQ", extra={"event": "task_dlq"})

    async def handle(self, raw: Any, message: RabbitMessage):
        received = time_ns()
        headers = message.headers if isinstance(message.headers, dict) else {}
        context = propagator.extract(headers)
        redelivered = getattr(message.raw_message, "redelivered", False) is True
        enqueued = headers.get("pg-enqueued-ns")
        message_age_ms = None
        if (
            isinstance(enqueued, str)
            and enqueued.isascii()
            and enqueued.isdecimal()
            and len(enqueued) <= 20
        ):
            sent = int(enqueued)
            if 0 < sent <= received:
                message_age_ms = (received - sent) / 1_000_000
                if not redelivered:
                    with span(
                        "queue.wait", context=context, start_time=sent, end_on_exit=False
                    ) as wait:
                        wait.end(end_time=received)
        fallback_id = raw.get("task_id") if isinstance(raw, dict) else None
        task_id = uuid_text(message.correlation_id) or fallback_id
        if redelivered:
            worker_redelivered.inc()
        with (
            task_context(task_id),
            batch_context(headers.get("pg-batch-id")),
            span(
                "process",
                context=context,
                kind=SpanKind.CONSUMER,
                attributes={
                    "messaging.system": "rabbitmq",
                    "messaging.destination.name": self.config.task_queue,
                    "messaging.rabbitmq.redelivered": redelivered,
                    "messaging.delivery.minimum_attempt": 2 if redelivered else 1,
                },
            ) as current,
        ):
            if batch_id := uuid_text(headers.get("pg-batch-id")):
                current.set_attribute("batch.id", batch_id)
            if bound_id := task_id_context.get():
                current.set_attribute("task.id", bound_id)
            if message_age_ms is not None:
                current.set_attribute("messaging.message_age_ms", message_age_ms)
                if not redelivered:
                    current.set_attribute("messaging.queue_wait_ms", message_age_ms)
            if redelivered:
                current.add_event("rabbitmq.redelivery")
            logger.info(
                "Task delivered",
                extra={
                    "event": "task_delivered",
                    "redelivered": redelivered,
                    "queue_wait_ms": None if redelivered else message_age_ms,
                    "worker": self.instance,
                },
            )
            try:
                await self.handle_delivery(raw, message, current)
            except Exception as exc:
                error(current, exc)
                worker_errors.labels("delivery").inc()
                logger.exception("Infrastructure failure; keep delivery recoverable")
                await asyncio.sleep(self.config.retry_delay_seconds)
                await message.nack(requeue=True)

    async def handle_delivery(self, raw, message, current):
        try:
            task = TaskMessage.model_validate(raw)
        except (ValidationError, TypeError, ValueError) as exc:
            error(current, exc)
            worker_errors.labels("validation").inc()
            logger.exception(
                "Invalid task delivery",
                extra={"event": "task_validation_failed"},
            )
            await self.dead_letter(raw, type(exc).__name__)
            worker_results.labels("dlq").inc()
            await message.ack()
            return
        with task_context(task.task_id):
            async with self.slots, self.store.locked(task.task_id) as conn:
                self.active += 1
                try:
                    with worker_active.track_inprogress():
                        await self.process(conn, task, raw)
                        await message.ack()
                finally:
                    self.active -= 1

    def predict(self, task, attempt):
        version = self.model.version
        with span(
            "inference",
            attributes={
                "model.alias": self.config.model_alias,
                "model.version": version if isinstance(version, str) else "unknown",
                "task.attempt": attempt,
            },
        ):
            if task.task_id == self.config.demo_poison_task_id:
                raise RuntimeError("Demonstration inference failure")
            return self.model.predict(task.features)

    async def process(self, conn, task, raw):
        current = trace.get_current_span()
        try:
            row = await self.store.ensure(conn, task, trace_context=trace_headers())
        except ValueError as exc:
            error(current, exc)
            worker_errors.labels("validation").inc()
            logger.exception("Task ID collision", extra={"event": "task_validation_failed"})
            await self.dead_letter(raw, type(exc).__name__)
            worker_results.labels("dlq").inc()
            return
        delivery = await self.store.delivery(conn, task.task_id)
        current.set_attribute("messaging.delivery.recorded_number", delivery)
        if delivery > 1:
            logger.info(
                "Task delivery recorded",
                extra={"event": "task_delivery_recorded", "delivery": delivery},
            )
        if row["status"] == "SUCCEEDED" or row["dlq_confirmed"]:
            logger.info("Duplicate delivery skipped", extra={"event": "task_duplicate"})
            return
        attempts = row["attempts"]
        reason = row["error"] or "attempt budget exhausted (possibly interrupted worker)"
        while attempts < self.config.max_attempts:
            attempts = await self.store.attempt(conn, task.task_id)
            try:
                started = perf_counter()
                result = await asyncio.get_running_loop().run_in_executor(
                    self.executor,
                    copy_context().run,
                    self.predict,
                    task,
                    attempts,
                )
                inference = perf_counter() - started
                delay_started = perf_counter()
                with span("inference.delay"):
                    await asyncio.sleep(self.config.inference_delay_seconds)
                delay = perf_counter() - delay_started
                worker_inference.observe(inference + delay, exemplar=exemplar())
            except Exception as exc:
                reason = type(exc).__name__
                error(current, exc)
                worker_inference.observe(perf_counter() - started, exemplar=exemplar())
                worker_errors.labels("inference").inc()
                logger.exception(
                    "Scoring attempt failed",
                    extra={"event": "task_attempt_failed", "attempt": attempts},
                )
                await self.store.state(conn, task.task_id, "RUNNING", reason)
                await asyncio.sleep(self.config.retry_delay_seconds)
                continue
            with span("db.write_result", attributes={"db.system.name": "postgresql"}):
                total_seconds = await self.store.success(conn, task, result, inference, delay)
                worker_task_latency.observe(total_seconds, exemplar=exemplar())
                logger.info(
                    "Task result stored", extra={"event": "task_succeeded", "attempt": attempts}
                )
            worker_results.labels("ok").inc()
            return
        current.set_status(Status(StatusCode.ERROR, "Attempt budget exhausted"))
        await self.dead_letter(raw, reason)
        await self.store.failed_to_dlq(conn, task.task_id, reason)
        worker_results.labels("dlq").inc()


worker = Worker(settings)
app = worker.app


def main():
    configure_logging(settings.log_level, "pg-worker")
    configure_tracing("pg-worker")
    try:
        asyncio.run(app.run())
    except Exception:
        logger.exception("Worker stopped after a fatal error", extra={"event": "worker_failed"})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
