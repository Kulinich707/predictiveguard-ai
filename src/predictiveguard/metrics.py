import os

import psutil
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector

from predictiveguard.config import settings
from predictiveguard.telemetry import otlp_enabled, resource


def latency_buckets(baseline):
    return sorted(
        {
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            *(
                baseline * factor
                for factor in (
                    0.25,
                    0.5,
                    0.8,
                    0.95,
                    1,
                    1.05,
                    1.2,
                    1.5,
                    2,
                    4,
                )
            ),
        }
    )


api_registry = CollectorRegistry()
worker_registry = CollectorRegistry()
ProcessCollector(registry=api_registry)
ProcessCollector(registry=worker_registry)
api_requests = Counter(
    "pg_api_requests",
    "HTTP responses",
    ["method", "route", "status"],
    registry=api_registry,
)
for status in ("202", "503"):
    api_requests.labels("POST", "/api/v1/tasks", status)

api_duration = Histogram(
    "pg_api_duration_seconds",
    "HTTP response latency",
    ["method", "route"],
    buckets=latency_buckets(settings.publish_timeout_seconds),
    registry=api_registry,
)
worker_results = Counter(
    "pg_worker_tasks", "Terminal worker outcomes", ["result"], registry=worker_registry
)
worker_errors = Counter(
    "pg_worker_attempt_failures",
    "Failed worker attempts",
    ["stage"],
    registry=worker_registry,
)
worker_inference = Histogram(
    "pg_worker_inference_seconds",
    "Prediction and configured scoring delay",
    buckets=latency_buckets(max(settings.inference_delay_seconds, 0.025)),
    registry=worker_registry,
)
worker_task_latency = Histogram(
    "pg_worker_task_latency_seconds",
    "Time from task creation to storing a successful result",
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 300),
    registry=worker_registry,
)
worker_active = Gauge("pg_worker_in_progress", "Tasks being handled", registry=worker_registry)
worker_redelivered = Counter(
    "pg_worker_redeliveries", "RabbitMQ deliveries marked redelivered", registry=worker_registry
)
queue_limit = Gauge(
    "pg_queue_limit", "Configured limit of ready messages", ["queue"], registry=api_registry
)
for outcome in ("ok", "error", "dlq"):
    worker_results.labels(outcome)
for stage in ("delivery", "inference", "validation"):
    worker_errors.labels(stage)


def exemplar():
    ctx = trace.get_current_span().get_span_context()
    return {"trace_id": format(ctx.trace_id, "032x")} if ctx.trace_flags.sampled else None


class BatchMetrics:
    def __init__(self):
        self.provider = None
        if not otlp_enabled("METRICS"):
            return
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "5000")),
        )
        identity = resource("pg-source").attributes["service.name"]
        self.provider = MeterProvider(
            resource=Resource({"service.name": identity, "service.instance.id": "batch"}),
            metric_readers=[reader],
            shutdown_on_exit=False,
        )
        meter = self.provider.get_meter("predictiveguard.source")
        self.published = meter.create_counter("pg_source_published")
        self.rejected = meter.create_counter("pg_source_rejected")
        self.wait = meter.create_counter("pg_source_wait_seconds")
        process = psutil.Process()
        meter.create_observable_gauge(
            "pg_source_resident_memory_bytes",
            callbacks=[lambda _: [Observation(process.memory_info().rss)]],
        )
        self.published.add(0)
        self.rejected.add(0)
        self.wait.add(0)

    def reject(self):
        if self.provider:
            self.rejected.add(1)

    def waited(self, seconds):
        if self.provider:
            self.wait.add(seconds)

    def accepted(self):
        if self.provider:
            self.published.add(1)

    def flush(self):
        if self.provider:
            self.provider.force_flush(timeout_millis=1500)

    def close(self):
        if self.provider:
            self.flush()
            self.provider.shutdown(timeout_millis=1500)
