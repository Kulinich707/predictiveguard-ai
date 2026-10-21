import argparse
import asyncio
import csv
import json
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4, uuid5

import psutil

from predictiveguard.config import settings
from predictiveguard.queue import declare, make_broker, publish_wait
from predictiveguard.schemas import ProcessRequest
from predictiveguard.tasks import TaskMessage

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

    output = await asyncio.to_thread(Path(manifest).open, "w")
    try:
        async with make_broker(settings) as broker:
            await declare(broker, settings)
            for index, features in enumerate(rows(path, repeats)):
                if limit and index >= limit:
                    break
                task = TaskMessage(task_id=uuid5(run_id, str(index)), features=features)
                await publish_wait(broker, settings, task, retry)
                accepted += 1
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
    finally:
        await asyncio.to_thread(output.close)
    print(
        json.dumps(
            {
                "accepted": accepted,
                "retries": retries,
                "seconds": perf_counter() - started,
                "run_id": str(run_id),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/ai4i2020.csv")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest", default="accepted.jsonl")
    parser.add_argument("--run-id", type=UUID, default=uuid4())
    args = parser.parse_args()
    asyncio.run(run(args.csv, args.repeats, args.limit, args.manifest, args.run_id))


if __name__ == "__main__":
    main()
