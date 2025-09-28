#!/bin/bash
set -e

# Configuration
REPO_URL="https://github.com/peter/parkiet.git"  # Update with actual repo URL
PROJECT_DIR="$HOME/parkiet"

PATH="$HOME/.local/bin:$PATH"

export ACCELERATOR_TYPE="v5p-16"
export RUNTIME_VERSION="v2-alpha-tpuv5"


# Check if repository exists, clone if not
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Repository not found at $PROJECT_DIR"
    echo "Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
else
    echo "Repository found at $PROJECT_DIR"
fi

# Navigate to project directory
cd "$PROJECT_DIR"

# Pull latest changes
echo "Pulling latest changes..."
git pull

# Sync dependencies
echo "Syncing dependencies..."
uv sync --extra tpu
source .venv/bin/activate

# Check if .env file exists and export WANDB_API_KEY
if [ -f ".env" ]; then
    export WANDB_API_KEY=$(cat .env | grep WANDB_API_KEY | cut -d '=' -f 2)
    echo "WANDB_API_KEY loaded from .env"
else
    echo "Warning: .env file not found. WANDB_API_KEY may not be set."
fi

# Start training in the background, log stdout & stderr
echo "Starting training..."
nohup python3 src/parkiet/jax/train_distributed.py > train.log 2>&1 &
TRAIN_PID=$!

echo "Training started in the background."
echo "Logs: $PROJECT_DIR/train.log"
echo "PID: $TRAIN_PID"