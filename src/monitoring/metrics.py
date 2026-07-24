# src/monitoring/metrics.py

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "ocr_requests_total",
    "Total OCR prediction requests"
)

REQUEST_LATENCY = Histogram(
    "ocr_request_latency_seconds",
    "OCR request latency"
)