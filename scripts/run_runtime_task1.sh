#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python python/experiments/runtime/profile_sgnav_runtime.py \
  --task task1 \
  --layout layout_a \
  --distractor none \
  --frames 1000 \
  --warmup-frames 50 \
  --checkpoint checkpoints/sgnav_task1_magenta_ball.pt \
  --unity-file unity/Build/USV_Harbor.x86_64 \
  --time-scale 20 \
  --output-dir results/runtime/sgnav_runtime_task1

