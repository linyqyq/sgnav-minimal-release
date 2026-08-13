#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python python/eval_harbor_checkpoints.py \
  --episodes 1 \
  --max-steps 1000 \
  --task task1 \
  --layout layout_a \
  --distractor none \
  --method sgnav \
  --checkpoint checkpoints/sgnav_task1_magenta_ball.pt \
  --unity-file unity/Build/USV_Harbor.x86_64 \
  --time-scale 20 \
  --output results/demo_task1_eval.csv

