# Contributing to slicer-mcp

Thank you for your interest in contributing to the MCP Slicer Bridge! This document provides guidelines for contributing.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
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

# Run with coverage
uv run pytest --cov=slicer_mcp --cov-report=html

# Run integration tests (requires Slicer on localhost:2016)
uv run pytest -v -m integration
```

### Code Quality

We use Black for formatting and Ruff for linting:

```bash
# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Code Style

- **Line length**: 100 characters
- **Formatting**: Black
- **Linting**: Ruff (rules: E, F, W, I, N, UP)
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style for all public APIs

### Example

```python
def validate_node_id(node_id: str) -> str:
    """Validate MRML node ID format.

    Args:
        node_id: The node ID to validate.

    Returns:
        The validated node_id.

    Raises:
        ValidationError: If node_id format is invalid.
    """
    if not MRML_ID_PATTERN.match(node_id):
        raise ValidationError(f"Invalid node ID: {node_id}")
    return node_id
```

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Write tests** for new functionality
3. **Ensure all tests pass**: `uv run pytest`
4. **Format and lint**: `uv run pre-commit run --all-files`
5. **Update documentation** if adding new features
6. **Submit PR** with a clear description of changes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example: `feat: Add support for custom view layouts`

## Security

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
2. Email the maintainers directly
3. Include steps to reproduce
4. Allow time for a fix before disclosure

See [SECURITY.md](SECURITY.md) for our security policy.

## Architecture

Before making significant changes, review:

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [SPECIFICATION.md](SPECIFICATION.md) - API reference
- [CLAUDE.md](CLAUDE.md) - Development patterns

## Adding New Tools

1. Implement in `src/slicer_mcp/tools.py`
2. Register in `src/slicer_mcp/server.py` with `@mcp.tool()`
3. Add tests in `tests/test_tools.py`
4. Update documentation

## Adding New Resources

1. Implement in `src/slicer_mcp/resources.py`
2. Register in `src/slicer_mcp/server.py` with `@mcp.resource()`
3. Add tests in `tests/test_mcp_protocol.py`
4. Update documentation

## Questions?

- Open a [GitHub issue](https://github.com/brainbloodbarrier/3dslicer-claude-bridge/issues)
- Check existing documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
