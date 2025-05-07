#!/bin/bash
# lint.sh
set -e

echo "Running isort..."
isort .

echo "Running black..."
black .

echo "Running flake8..."
flake8 .

echo "Running mypy..."
mypy src/services/shared/models