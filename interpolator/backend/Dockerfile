#########################################
# STAGE 1 — Builder
#########################################
FROM python:3.12-slim AS builder

# Avoid Python writing pyc files, ensure clean logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working dir in container
WORKDIR /app

# Install build tools and dependencies required for Python scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the backend project metadata
COPY pyproject.toml .

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy the backend source code properly
COPY fivedreg ./fivedreg

# Install backend in editable mode (in builder stage)
RUN pip install -e .

#########################################
# STAGE 2 — Runtime image
#########################################
FROM python:3.12-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy the installed package from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only backend source code (for uvicorn reload / artifacts)
COPY fivedreg ./fivedreg

# Create directory for model artifacts and give to non-root user
RUN mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app

# Switch to secure user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# HEALTHCHECK: confirm API is reachable
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "fivedreg.main:app", "--host", "0.0.0.0", "--port", "8000"]

