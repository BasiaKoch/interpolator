# Utility Scripts

Collection of helper scripts for development, testing, and deployment.

## Available Scripts

### Development

- **`setup-dev.sh`** - Complete development environment setup
  ```bash
  ./scripts/setup-dev.sh
  ```
  Sets up backend virtual environment, installs all dependencies, sets up frontend, and configures Jupyter kernel.

### Testing

- **`test-pipeline.sh`** - Run complete test suite with coverage
  ```bash
  ./scripts/test-pipeline.sh
  ```
  Runs pytest with coverage reports and code quality checks.

- **`run-benchmark.sh`** - Run performance benchmarks
  ```bash
  ./scripts/run-benchmark.sh
  ```
  Executes performance tests and saves results to JSON.

- **`profile-detailed.sh`** - Detailed profiling (memory + CPU)
  ```bash
  ./scripts/profile-detailed.sh
  ```
  Runs line-by-line memory profiling and cProfile analysis.

### Documentation

- **`build-docs.sh`** - Build Sphinx documentation
  ```bash
  ./scripts/build-docs.sh
  ```
  Generates HTML documentation in `backend/docs/build/html/`.

### Docker Management

- **`docker-logs.sh`** - View Docker container logs
  ```bash
  ./scripts/docker-logs.sh
  ```
  Interactive menu to view backend, frontend, or all logs.

- **`docker-cleanup.sh`** - Clean up Docker resources
  ```bash
  ./scripts/docker-cleanup.sh
  ```
  Stops containers, removes volumes, and cleans up unused resources.

## Quick Reference

```bash
# First time setup
./scripts/setup-dev.sh

# Start development servers
./backend/start_server.sh          # Backend API
cd frontend && npm run dev          # Frontend

# Or use Docker
./launch.sh                         # Both in Docker

# Run tests
./scripts/test-pipeline.sh

# Build documentation
./scripts/build-docs.sh

# View logs
./scripts/docker-logs.sh

# Benchmark performance
./scripts/run-benchmark.sh
```

## Script Locations

All scripts are in the `scripts/` directory at the project root for easy access and organization.
