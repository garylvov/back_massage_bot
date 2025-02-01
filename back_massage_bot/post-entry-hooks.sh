#!/usr/bin/env bash
# If you modify me, your next docker build may take long!
# Check if we're in the right environment, if not activate it
if [[ "$CONDA_DEFAULT_ENV" != "back_massage_bot" ]]; then
    conda activate back_massage_bot
fi

echo "Activated Conda."
