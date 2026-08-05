#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Virtual environment not found: $VENV_PATH"
  echo "Create it first with: python3 -m venv .venv"
  return 1 2>/dev/null || exit 1
fi

source "$VENV_PATH/bin/activate"
echo "Activated WAFT virtual environment: $VENV_PATH"
