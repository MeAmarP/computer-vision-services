#!/bin/bash
set -e

export PYTHONPATH="$(pwd)/pytorch:${PYTHONPATH}"

# append header with timestamp
echo "## $(date)" >> test-report.md
pytest --cov=pytorch --cov-report=term "$@" 2>&1 | tee -a test-report.md
