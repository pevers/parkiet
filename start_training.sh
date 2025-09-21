#!/bin/bash
set -e

cd /home/peter/parkiet
git pull
uv sync
source .venv/bin/activate

# Start training in the background, log stdout & stderr
export WANDB_API_KEY=$(cat .env | grep WANDB_API_KEY | cut -d '=' -f 2)
nohup python3 src/parkiet/jax/train_distributed.py > train.log 2>&1 &
echo "Training started in the background."
echo "Logs: /home/peter/parkiet/train.log"
echo "PID: $!"