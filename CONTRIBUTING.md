# Contributing to slicer-mcp

Thank you for your interest in contributing to the MCP Slicer Bridge!

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended)
- 3D Slicer with WebServer extension (for integration tests)

### Installation

```bash
# Clone the repository
git clone https://github.com/brainbloodbarrier/3dslicer-claude-bridge.git
cd 3dslicer-claude-bridge

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all unit tests (no Slicer required)
uv run pytest -v

# Run with coverage (aim for >85% branch coverage)
uv run pytest --cov=slicer_mcp --cov-report=html

# Run integration tests (requires Slicer on localhost:2016)
uv run pytest -v -m integration
```

### Code Quality

We use Black for formatting, Ruff for linting, and Mypy for static analysis:

```bash
# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Run all pre-commit hooks (includes mypy)
uv run pre-commit run --all-files
```

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Write tests** for new functionality (unit tests should mock HTTP calls, not require Slicer)
3. **Ensure all tests pass**: `uv run pytest`
4. **Format and lint**: `uv run pre-commit run --all-files`
5. **Submit PR** with a clear description of changes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Development Guidelines
Please refer directly to the `CLAUDE.md` file in the root of the project. It is the single source of truth for architectural invariants, error handling paradigms, tool building instructions, and validation requirements.

## Questions?

- Open a [GitHub issue](https://github.com/brainbloodbarrier/3dslicer-claude-bridge/issues)
