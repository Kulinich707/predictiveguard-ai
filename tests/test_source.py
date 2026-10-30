import json
from contextlib import asynccontextmanager
from uuid import uuid4

from predictiveguard import source


def test_streaming_csv_excludes_labels_and_reopens_each_repeat(tmp_path):
    path = tmp_path / "sensors.csv"
    path.write_text(
        "Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],"
        "Torque [Nm],Tool wear [min],Machine failure\nL,300,310,1500,40,100,1\n"
    )
    iterator = source.rows(path, repeats=2)
    first = next(iterator)
    assert "failure" not in first.model_dump()
    assert next(iterator) == first
    assert list(iterator) == []


async def test_manifest_contains_only_confirmed_tasks_and_stable_ids(tmp_path, config, monkeypatch):
    path = tmp_path / "sensors.csv"
    path.write_text(
        "Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],"
        "Torque [Nm],Tool wear [min]\nL,300,310,1500,40,100\nM,301,311,1600,41,101\n"
    )

    @asynccontextmanager
    async def broker():
        yield None

    async def declare(*args):
        pass

    confirmed = []

    async def publish(broker, config, task, retry):
        retry()
        confirmed.append(task)

    monkeypatch.setattr(source, "make_broker", lambda config: broker())
    monkeypatch.setattr(source, "declare", declare)
    monkeypatch.setattr(source, "publish_wait", publish)
    monkeypatch.setattr(source, "settings", config)
    run_id = uuid4()
    manifest = tmp_path / "accepted.jsonl"
    await source.run(path, 2, 1, manifest, run_id)
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(entries) == len(confirmed) == 1
    assert entries[0]["retries"] == 1
    assert entries[0]["task_id"] == str(confirmed[0].task_id)
    assert entries[0]["rss_bytes"] > 0
    await source.run(path, 1, 0, manifest, run_id)
    assert confirmed[1].task_id == confirmed[0].task_id
    assert len(manifest.read_text().splitlines()) == 2
