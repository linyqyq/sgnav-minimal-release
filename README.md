# SGNav Minimal Public Release

This repository contains a compact release of SGNav for semantic-guided autonomous surface vehicle navigation. To keep the release lightweight, it includes one representative Unity harbour build, one Task-1 checkpoint, evaluation code, and runtime profiling scripts.

The full experimental workspace used for the paper contains additional scenes, checkpoints, and ablations. Those assets are not required for the minimal demo.

## Contents

- `python/`: SGNav policy, semantic target localizer, Unity ML-Agents wrapper, and runtime profiling code.
- `checkpoints/sgnav_task1_magenta_ball.pt`: trained SGNav checkpoint for the representative Task-1 setting.
- `unity/Build/`: pre-built Linux Unity harbour executable. The Unity Editor project is not included. For GitHub releases, the build can be distributed as a separate release asset instead of being committed to git.
- `scripts/`: convenience commands for evaluation and runtime profiling.
- `results/paper_tables/`: example runtime summaries produced from the profiling scripts.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate usv-clip
```

The semantic perception module uses GroundingDINO through `transformers` and CLIP through `open_clip_torch`. The first run may download model weights from Hugging Face/OpenCLIP.

## Quick Demo

Run one Task-1 evaluation episode:

```bash
bash scripts/run_demo_task1.sh
```

The script uses:

```text
unity/Build/USV_Harbor.x86_64
checkpoints/sgnav_task1_magenta_ball.pt
```

If the Unity build is distributed separately, download `unity_harbor_build_linux.zip` from the GitHub Release page and unzip it into the repository root before running the demo:

```bash
unzip unity_harbor_build_linux.zip
chmod +x unity/Build/USV_Harbor.x86_64
```

If you run on a remote machine without a display, install and use `xvfb-run` or run with `--no-graphics`. Camera observations may be unreliable in Unity's null graphics mode, so a real or virtual display is recommended for visual perception experiments.

## Runtime Profiling

Run component-level profiling:

```bash
bash scripts/run_runtime_task1.sh
```

Outputs are written under:

```text
results/runtime/sgnav_runtime_task1/
```

The main files are:

- `runtime_frames.csv`: per-frame timings.
- `runtime_summary.csv`: amortized per-control-frame runtime.
- `runtime_refresh_summary.csv`: semantic-refresh-only runtime.

## Notes

This release is intended for review-time reproducibility and demonstration. It provides a representative scene and checkpoint rather than the full set of paper assets.

Unity build platform: Linux x86_64.
