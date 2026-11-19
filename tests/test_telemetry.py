import json
import logging
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, Mock

from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from predictiveguard.logging_config import JsonFormatter, bind_request_id, reset_request_id
from predictiveguard.metrics import BatchMetrics, worker_registry
from predictiveguard.queue import publish
from predictiveguard.telemetry import batch_context, span, task_context
from predictiveguard.worker import Worker


async def test_trace_survives_amqp_headers_and_executor(config, task, prediction, monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("predictiveguard.telemetry.trace.get_tracer", provider.get_tracer)

    broker = AsyncMock()
    store = Mock(
        locked=Mock(return_value=MagicMock()),
        ensure=AsyncMock(
            return_value={"status": "PENDING", "attempts": 0, "error": None, "dlq_confirmed": False}
        ),
        delivery=AsyncMock(return_value=1),
        attempt=AsyncMock(return_value=1),
        success=AsyncMock(return_value=0.6),
    )
    store.locked.return_value.__aenter__ = AsyncMock()
    store.locked.return_value.__aexit__ = AsyncMock()
    formatter = JsonFormatter()

    def predict(features):
        record = logging.LogRecord("inference", logging.INFO, "", 0, "running", (), None)
        logged.append(json.loads(formatter.format(record)))
        return prediction

    logged = []
    worker = Worker(config, Mock(predict=predict, version="9"), store)
    run_id = "40000000-0000-4000-8000-000000006000"
    metric = "pg_worker_task_latency_seconds_count"
    latency_count = worker_registry.get_sample_value(metric)
    try:
        with batch_context(run_id), task_context(task.task_id), span("POST /api/v1/tasks"):
            await publish(broker, config, task)
        published = broker.publish.await_args.kwargs
        message = AsyncMock(
            headers=published["headers"],
            correlation_id=published["correlation_id"],
            raw_message=Mock(redelivered=False),
        )
        await worker.handle(task.wire(), message)
        message.ack.assert_awaited_once()
        store.ensure.return_value["status"] = "SUCCEEDED"
        store.delivery.return_value = 2
        message.raw_message.redelivered = True
        await worker.handle(task.wire(), message)
        assert message.ack.await_count == 2
        assert worker_registry.get_sample_value(metric) == latency_count + 1
    finally:
        worker.executor.shutdown()
        provider.shutdown()

    finished = exporter.get_finished_spans()
    redelivered = finished[-1]
    spans = {value.name: value for value in finished[:-1]}
    root = spans["POST /api/v1/tasks"]
    producer = spans["publish"]
    consumer = spans["process"]
    inference = spans["inference"]
    assert len({value.context.trace_id for value in spans.values()}) == 1
    assert producer.parent.span_id == root.context.span_id
    assert consumer.parent.span_id == producer.context.span_id
    assert inference.parent.span_id == consumer.context.span_id
    assert spans["db.write_result"].parent.span_id == consumer.context.span_id
    assert spans["queue.wait"].end_time >= spans["queue.wait"].start_time
    assert redelivered.name == "process"
    assert redelivered.parent.span_id == producer.context.span_id
    assert redelivered.context.trace_id == root.context.trace_id
    assert redelivered.attributes["messaging.delivery.recorded_number"] == 2
    assert "messaging.queue_wait_ms" not in redelivered.attributes
    assert redelivered.events[0].name == "rabbitmq.redelivery"
    assert inference.attributes["model.version"] == "9"
    assert logged[0]["run_id"] == run_id
    assert consumer.attributes["batch.id"] == run_id
    assert logged[0]["task_id"] == str(task.task_id)
    assert logged[0]["trace_id"] == f"{root.context.trace_id:032x}"
    assert logged[0]["span_id"] == f"{inference.context.span_id:016x}"
    assert consumer.status.status_code != StatusCode.ERROR


def test_json_formatter_redacts_dsn_and_exception_messages(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "password-demo-000")
    try:
        raise ValueError("sensitive sensor record")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "test",
            logging.ERROR,
            "",
            0,
            "postgresql://admin:password-demo-000@database:5432 password-demo-000",
            (),
            sys.exc_info(),
        )
    token = bind_request_id("password-demo-000")
    try:
        line = JsonFormatter().format(record)
    finally:
        reset_request_id(token)
    assert "password-demo-000" not in line
    assert "sensitive sensor record" not in line
    assert json.loads(line)["exception"]["type"] == "ValueError"
    assert json.loads(line)["exception"]["stacktrace"]
    assert json.loads(line)["request_id"] == "-"


def test_json_handler_enforces_log_level_for_uvicorn():
    probe = """
import logging
import logging.config
from uvicorn.config import LOGGING_CONFIG
from predictiveguard.logging_config import configure_logging

logging.config.dictConfig(LOGGING_CONFIG)
configure_logging("ERROR")
logger = logging.getLogger("uvicorn.error")
logger.info("discarded")
logger.error("kept")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    )
    (line,) = result.stdout.splitlines()
    assert json.loads(line)["message"] == "kept"
    assert not result.stderr


def test_source_pushes_cumulative_counters_and_memory(monkeypatch):
    reader = InMemoryMetricReader()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setattr("predictiveguard.metrics.OTLPMetricExporter", Mock())
    monkeypatch.setattr(
        "predictiveguard.metrics.PeriodicExportingMetricReader", Mock(return_value=reader)
    )

    metrics = BatchMetrics()

    def points():
        data = reader.get_metrics_data().resource_metrics[0]
        assert data.resource.attributes["service.instance.id"] == "batch"
        return {m.name: m.data.data_points[0].value for m in data.scope_metrics[0].metrics}

    try:
        metrics.flush()
        assert points()["pg_source_published"] == 0
        metrics.accepted()
        metrics.accepted()
        metrics.reject()
        metrics.waited(0.25)
        final = points()
        assert final["pg_source_published"] == 2
        assert final["pg_source_rejected"] == 1
        assert final["pg_source_wait_seconds"] == 0.25
        assert final["pg_source_resident_memory_bytes"] > 0
    finally:
        metrics.close()
