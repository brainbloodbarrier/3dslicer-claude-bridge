# Multi-stage build for slicer-mcp
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-editable

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ ./src/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV SLICER_URL=http://host.docker.internal:2016
ENV SLICER_TIMEOUT=30

# Expose no ports (MCP uses stdio)
# The container communicates via stdin/stdout

# Health check (optional - for container orchestration)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import slicer_mcp; print('ok')" || exit 1

# Run the MCP server
ENTRYPOINT ["python", "-m", "slicer_mcp"]
