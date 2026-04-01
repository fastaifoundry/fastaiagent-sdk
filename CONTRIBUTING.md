# Contributing to FastAIAgent SDK

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/fastaifoundry/fastaiagent-sdk.git
   cd fastaiagent-sdk
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[dev,all]"
   ```

3. Run the test suite:
   ```bash
   pytest tests/ -v
   ```

## Code Quality

We use the following tools to maintain code quality:

- **ruff** for linting and formatting
- **mypy** for type checking (strict mode)
- **pytest** for testing

Before submitting a PR, ensure:
```bash
ruff check .
ruff format --check .
mypy fastaiagent/ --ignore-missing-imports
pytest tests/ -v
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for any new functionality
3. Ensure all tests pass and code quality checks succeed
4. Update documentation if your change affects public APIs
5. Update `CHANGELOG.md` if the change is user-facing
6. Submit a PR with a clear description of changes

## Reporting Issues

Use [GitHub Issues](https://github.com/fastaifoundry/fastaiagent-sdk/issues) to report bugs or request features. We have templates for:

- Bug reports
- Feature requests
- New integration requests
- Documentation improvements

## How to Add a New Built-in Scorer

Scorers live in `fastaiagent/eval/builtins.py`. To add a new one:

1. Create a class that extends `Scorer`:

   ```python
   # fastaiagent/eval/builtins.py
   class ResponseLength(Scorer):
       """Check that response length is within bounds."""

       name = "response_length"

       def __init__(self, min_len: int = 1, max_len: int = 5000):
           self.min_len = min_len
           self.max_len = max_len

       def score(self, input: str, output: str, expected: str | None = None, **kwargs) -> ScorerResult:
           length = len(output)
           passed = self.min_len <= length <= self.max_len
           return ScorerResult(
               score=1.0 if passed else 0.0,
               passed=passed,
               reason=f"Length {length} {'within' if passed else 'outside'} [{self.min_len}, {self.max_len}]",
           )
   ```

2. Register it in the `BUILTIN_SCORERS` dict at the bottom of the file:

   ```python
   BUILTIN_SCORERS["response_length"] = ResponseLength
   ```

3. Add tests in `tests/test_eval.py`
4. Update the scorer list in `docs/evaluation/index.md`

## How to Add a New Tool Type

Tools live in `fastaiagent/tool/`. To add a new tool type:

1. Create a new file (e.g., `fastaiagent/tool/graphql.py`)
2. Extend the `Tool` base class from `fastaiagent/tool/base.py`:

   ```python
   from fastaiagent.tool.base import Tool, ToolResult

   class GraphQLTool(Tool):
       def __init__(self, name: str, url: str, query: str, **kwargs):
           self.url = url
           self.query = query
           super().__init__(name=name, **kwargs)

       async def aexecute(self, arguments: dict[str, Any]) -> ToolResult:
           # Implementation here
           ...

       def _tool_type(self) -> str:
           return "graphql"

       def _config_dict(self) -> dict[str, Any]:
           return {"url": self.url, "query": self.query}

       @classmethod
       def _from_dict(cls, data: dict[str, Any]) -> "GraphQLTool":
           config = data.get("config", {})
           return cls(name=data["name"], url=config["url"], query=config["query"])
   ```

3. Register it in `fastaiagent/tool/base.py` `Tool.from_dict()` dispatch
4. Export it from `fastaiagent/tool/__init__.py`
5. Add tests in `tests/test_tool.py`
6. Add documentation in `docs/tools/`

## How to Add a Framework Integration

Integrations live in `fastaiagent/integrations/`. To add auto-tracing for a new framework:

1. Create a new file (e.g., `fastaiagent/integrations/dspy.py`)
2. Implement `enable()` and `disable()` functions:

   ```python
   _original_fn = None

   def enable() -> None:
       """Enable auto-tracing for DSPy."""
       global _original_fn
       import dspy
       from fastaiagent.trace.otel import get_tracer

       tracer = get_tracer("fastaiagent.integrations.dspy")
       _original_fn = dspy.Module.forward

       def traced_forward(self, *args, **kwargs):
           with tracer.start_as_current_span(f"dspy.{type(self).__name__}"):
               return _original_fn(self, *args, **kwargs)

       dspy.Module.forward = traced_forward

   def disable() -> None:
       """Disable auto-tracing for DSPy."""
       global _original_fn
       if _original_fn is not None:
           import dspy
           dspy.Module.forward = _original_fn
           _original_fn = None
   ```

3. Add an optional dependency in `pyproject.toml` if needed
4. Add tests in `tests/test_integrations.py`
5. Add documentation in `docs/integrations/index.md`

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
