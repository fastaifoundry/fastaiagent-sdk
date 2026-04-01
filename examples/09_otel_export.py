"""Example 09: Export traces to an OTel collector (Jaeger, Datadog, etc.).

Shows how to add an OTLP exporter for sending traces
to external observability tools.
Requires: pip install fastaiagent[otel-export]
"""

from fastaiagent.trace import add_exporter, trace_context

# Add an OTLP HTTP exporter
# from fastaiagent.trace.export import create_otlp_exporter
# exporter = create_otlp_exporter(endpoint="http://localhost:4318/v1/traces")
# add_exporter(exporter)

if __name__ == "__main__":
    print("OTel Export example")
    print("=" * 40)
    print()
    print("1. Install: pip install fastaiagent[otel-export]")
    print("2. Start a collector (e.g., Jaeger):")
    print("   docker run -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one")
    print("3. Add to your code:")
    print("   from fastaiagent.trace.export import create_otlp_exporter")
    print("   from fastaiagent.trace import add_exporter")
    print("   add_exporter(create_otlp_exporter('http://localhost:4318/v1/traces'))")
    print()

    # Traces are also stored locally regardless of export
    with trace_context("example-operation") as span:
        span.set_attribute("example.key", "value")
        print("Created a trace span (stored locally)")
