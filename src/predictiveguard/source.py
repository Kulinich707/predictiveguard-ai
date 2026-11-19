import argparse
import asyncio
import csv
import json
import logging
import signal
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4, uuid5

import psutil

from predictiveguard.config import settings
from predictiveguard.logging_config import configure_logging
from predictiveguard.metrics import BatchMetrics
from predictiveguard.queue import declare, make_broker, publish_wait
from predictiveguard.schemas import ProcessRequest
from predictiveguard.tasks import TaskMessage
from predictiveguard.telemetry import batch_id_context, configure_tracing, span, task_context

logger = logging.getLogger(__name__)

COLUMNS = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}


def rows(path, repeats=1):
    for _ in range(repeats):
        with Path(path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                yield ProcessRequest.model_validate({v: row[k] for k, v in COLUMNS.items()})


async def run(path, repeats, limit, manifest, run_id):
    accepted = retries = 0
    started = perf_counter()

    def retry():
        nonlocal retries
        retries += 1

    output = await asyncio.to_thread(Path(manifest).open, "w") if manifest else None
    metrics = BatchMetrics()
    baseline = asyncio.create_task(asyncio.to_thread(metrics.flush)) if metrics.provider else None
    token = batch_id_context.set(str(run_id))
    logger.info("Batch started", extra={"event": "batch_started"})
    try:
        async with make_broker(settings) as broker:
            await declare(broker, settings)
            for index, features in enumerate(rows(path, repeats)):
                if limit and index >= limit:
                    break
                task = TaskMessage(task_id=uuid5(run_id, str(index)), features=features)
                with (
                    task_context(task.task_id),
                    span(
                        "source.task",
                        attributes={"task.id": str(task.task_id), "batch.id": str(run_id)},
                    ),
                ):
                    if metrics.provider:
                        await publish_wait(broker, settings, task, retry, metrics=metrics)
                    else:
                        await publish_wait(broker, settings, task, retry)
                    metrics.accepted()
                    accepted += 1
                if output:
                    await asyncio.to_thread(
                        output.write,
                        json.dumps(
                            {
                                "task_id": str(task.task_id),
                                "elapsed": perf_counter() - started,
                                "retries": retries,
                                "rss_bytes": psutil.Process().memory_info().rss,
                            }
                        )
                        + "\n",
                    )
                    await asyncio.to_thread(output.flush)
    except asyncio.CancelledError:
        logger.info("Batch shutdown started", extra={"event": "shutdown_started"})
        raise
    except Exception:
        logger.exception("Batch failed", extra={"event": "batch_failed", "accepted": accepted})
        raise
    finally:
        if output:
            await asyncio.to_thread(output.close)
        if baseline:
            await baseline
            await asyncio.to_thread(metrics.close)
        batch_id_context.reset(token)
    logger.info(
        "Batch completed",
        extra={
            "event": "batch_completed",
            "accepted": accepted,
            "retries": retries,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "run_id": str(run_id),
        },
    )


async def run_until_stopped(args):
    loop = asyncio.get_running_loop()
    batch = asyncio.create_task(run(args.csv, args.repeats, args.limit, args.manifest, args.run_id))
    loop.add_signal_handler(signal.SIGTERM, batch.cancel)
    try:
        await batch
    except asyncio.CancelledError:
        logger.info(
            "Batch shutdown completed",
            extra={"event": "shutdown_finished", "run_id": str(args.run_id)},
        )
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/ai4i2020.csv")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest", help="Optional output manifest for the HW4 experiment")
    parser.add_argument("--run-id", type=UUID, default=uuid4())
    args = parser.parse_args()
    configure_logging(settings.log_level, "pg-source")
    configure_tracing("pg-source")
    try:
        asyncio.run(run_until_stopped(args))
    except Exception:
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
