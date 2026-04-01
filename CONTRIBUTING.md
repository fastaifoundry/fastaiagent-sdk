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
- **mypy** for type checking
- **pytest** for testing

Before submitting a PR, ensure:
```bash
ruff check .
ruff format --check .
mypy fastaiagent/
pytest tests/ -v
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for any new functionality
3. Ensure all tests pass and code quality checks succeed
4. Submit a PR with a clear description of changes

## Reporting Issues

Use [GitHub Issues](https://github.com/fastaifoundry/fastaiagent-sdk/issues) to report bugs or request features.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
