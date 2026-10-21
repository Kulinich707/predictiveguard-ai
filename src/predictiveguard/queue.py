import asyncio
import random

from aio_pika.exceptions import DeliveryError
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue

from predictiveguard.config import Settings
from predictiveguard.tasks import TaskMessage


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
    try:
        await broker.publish(
            task.wire(),
            queue=topology(config)[0],
            persist=True,
            message_id=str(task.task_id),
            mandatory=True,
            timeout=config.publish_timeout_seconds,
        )
    except DeliveryError as exc:
        raise Overloaded("Broker rejected publication") from exc


async def publish_wait(broker, config, task, on_retry=None):
    attempt = 0
    while True:
        try:
            await publish(broker, config, task)
            return
        except (Overloaded, TimeoutError, ConnectionError):
            attempt += 1
            if on_retry:
                on_retry()
            ceiling = min(config.retry_delay_seconds * 2 ** min(attempt, 4), 1.0)
            await asyncio.sleep(random.uniform(ceiling / 2, ceiling))
