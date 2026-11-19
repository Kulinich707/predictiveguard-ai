import asyncio
import logging
import random
from time import time_ns

from aio_pika.exceptions import DeliveryError
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from opentelemetry.trace import SpanKind

from predictiveguard.config import Settings
from predictiveguard.tasks import TaskMessage
from predictiveguard.telemetry import batch_id_context, span, trace_headers

logger = logging.getLogger(__name__)


class Overloaded(Exception):
    pass


def topology(config: Settings):
    arguments = {"x-overflow": "reject-publish"}
    if config.queue_max_length:
        arguments["x-max-length"] = config.queue_max_length
    return (
        RabbitQueue(config.task_queue, durable=True, arguments=arguments),
        RabbitQueue(config.task_queue + ".dlq", durable=True),
    )


def make_broker(config: Settings) -> RabbitBroker:
    return RabbitBroker(
        config.rabbit_url,
        logger=logging.getLogger("faststream"),
        log_level=logging.DEBUG,
        default_channel=Channel(
            prefetch_count=config.prefetch_count,
            publisher_confirms=True,
            on_return_raises=True,
        ),
        graceful_timeout=config.shutdown_timeout_seconds,
    )


async def declare(broker, config):
    for queue in topology(config):
        await broker.declare_queue(queue)


async def publish(broker, config: Settings, task: TaskMessage):
    with span(
        "publish",
        kind=SpanKind.PRODUCER,
        attributes={
            "messaging.system": "rabbitmq",
            "messaging.destination.name": config.task_queue,
            "task.id": str(task.task_id),
        },
    ):
        headers = trace_headers() | {"pg-enqueued-ns": str(time_ns())}
        if batch_id := batch_id_context.get():
            headers["pg-batch-id"] = batch_id
        try:
            await broker.publish(
                task.wire(),
                queue=topology(config)[0],
                persist=True,
                message_id=str(task.task_id),
                correlation_id=str(task.task_id),
                headers=headers,
                mandatory=True,
                timeout=config.publish_timeout_seconds,
            )
        except DeliveryError as exc:
            logger.warning("RabbitMQ rejected publication", extra={"event": "broker_nack"})
            raise Overloaded("Broker rejected publication") from exc
        logger.info("Task published", extra={"event": "task_published"})


async def publish_wait(broker, config, task, on_retry=None, *, metrics=None):
    attempt = 0
    while True:
        try:
            await publish(broker, config, task)
            return
        except (Overloaded, TimeoutError, ConnectionError) as exc:
            attempt += 1
            if on_retry:
                on_retry()
            if metrics and isinstance(exc, Overloaded):
                metrics.reject()
            ceiling = min(config.retry_delay_seconds * 2 ** min(attempt, 4), 1.0)
            delay = random.uniform(ceiling / 2, ceiling)
            fields = {"attempt": attempt, "retry_seconds": delay}
            with span("source.backpressure", attributes=fields) as waiting:
                waiting.add_event(type(exc).__name__)
                logger.warning(
                    "Batch source is backing off",
                    extra={"event": "source_backpressure", **fields},
                )
                started = asyncio.get_running_loop().time()
                try:
                    await asyncio.sleep(delay)
                finally:
                    if metrics:
                        metrics.waited(asyncio.get_running_loop().time() - started)
