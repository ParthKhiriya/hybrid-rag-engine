from prometheus_client import Counter, Histogram

REQUESTS = Counter("rag_requests_total", "Total RAG API requests", ["endpoint", "status"])
RETRIEVAL_LATENCY = Histogram("rag_retrieval_seconds", "Hybrid retrieval latency")
GENERATION_LATENCY = Histogram("rag_generation_seconds", "LLM generation latency", ["provider"])
END_TO_END_LATENCY = Histogram("rag_end_to_end_seconds", "End-to-end query latency")


def configure_tracing(app, enabled: bool, service_name: str) -> None:
    if not enabled:
        return
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
