#!/bin/bash
# Sync MinText code to TPU VM
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TPU_HOST="${TPU_HOST:-tpu-v6e-8}"
TPU_DIR="${TPU_DIR:-~/mintext}"

rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  --exclude='.pytest_cache' \
  --exclude='.git' \
  --exclude='.claude' \
  --exclude='.omc' \
  --exclude='.venv' \
  --exclude='docs' \
  --exclude='tensorboard' \
  "$PROJECT_DIR/" "$TPU_HOST:$TPU_DIR/"

echo "Synced to $TPU_HOST:$TPU_DIR"
