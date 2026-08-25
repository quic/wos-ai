"""
Metrics — Prometheus counters/histograms shared by server.py (HTTP layer) and session_manager.py (inference layer). 
"""

from prometheus_client import Counter, Histogram, REGISTRY
from utils.logger import get_logger

logger = get_logger(__name__)


def _counter(name: str, doc: str, labels):
    """Return existing Counter if already registered (handles uvicorn --reload)."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, doc, labels)


def _histogram(name: str, doc: str, labels):
    """Return existing Histogram if already registered (handles uvicorn --reload)."""
    # prometheus_client stores histograms under <name>_bucket
    bucket_name = name + "_bucket"
    if bucket_name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[bucket_name]
    return Histogram(name, doc, labels)


HTTP_REQUESTS_TOTAL = _counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "path", "status"],
)
HTTP_REQUEST_LATENCY_SECONDS = _histogram(
    "http_request_latency_seconds", "HTTP request latency in seconds",
    ["method", "path"],
)
MODEL_INFERENCE_TOTAL = _counter(
    "model_inference_total", "Inference calls per model",
    ["model_id", "backend", "operation", "status"],
)
MODEL_TOKENS_TOTAL = _counter(
    "model_tokens_total", "Tokens processed per model",
    ["model_id", "direction"],  # direction: prompt | completion
)
MODEL_LOAD_EVENTS_TOTAL = _counter(
    "model_load_events_total", "Model lifecycle events",
    ["model_id", "event"],  # event: loaded | load_failed | unloaded | unloaded_idle
)


def safe_inc(counter: Counter, label_values, amount: float = 1) -> None:
    """Increment a Counter without ever letting a metrics failure break the caller."""
    try:
        counter.labels(*label_values).inc(amount)
    except Exception as exc:
        logger.warning(f"[metrics] failed to increment {counter}: {exc}")


def safe_observe(histogram: Histogram, label_values, value: float) -> None:
    """Record a Histogram observation without ever letting a metrics failure break the caller."""
    try:
        histogram.labels(*label_values).observe(value)
    except Exception as exc:
        logger.warning(f"[metrics] failed to observe {histogram}: {exc}")

