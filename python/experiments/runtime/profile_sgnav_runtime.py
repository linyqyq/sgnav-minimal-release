import argparse
import csv
import json
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.dirname(PYTHON_ROOT)
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from envs.unity_ma_env import UnitySingleAgentEnv
from eval_harbor_checkpoints import (
    harbor_distractor_id,
    harbor_layout_id,
    load_model,
    normalize_distractor_name,
    normalize_layout_name,
    normalize_name,
    resolve_input_path,
)
from models.target_localizer_harbor import (
    HarborTargetLocalizer,
    TARGET_STATE_DIM,
    build_harbor_task_config,
)
from train_ppo_dino_harbor_editor import (
    LocalizerPolicyAdapter,
    build_policy_vector,
    harbor_task_id,
    hwc_to_chw,
)


DEFAULT_CHECKPOINTS = {
    "task1": "checkpoints/sgnav_task1_magenta_ball.pt",
    "magenta_ball": "checkpoints/sgnav_task1_magenta_ball.pt",
    "task3": "checkpoints/sgnav_task1_magenta_ball.pt",
    "yellow_tugboat_marker": "checkpoints/sgnav_task1_magenta_ball.pt",
}

FALLBACK_CHECKPOINTS = {
    "task1": [
        "checkpoints/sgnav_task1_magenta_ball.pt",
    ],
    "task3": [
        "checkpoints/sgnav_task1_magenta_ball.pt",
    ],
}


def resolve_output_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def default_checkpoint(task):
    task_key = normalize_name(task)
    if task_key in DEFAULT_CHECKPOINTS:
        path = resolve_input_path(DEFAULT_CHECKPOINTS[task_key])
        if os.path.exists(path):
            return path

    fallback_key = "task3" if task_key in ("task3", "yellow_tugboat_marker") else "task1"
    for rel_path in FALLBACK_CHECKPOINTS[fallback_key]:
        path = resolve_input_path(rel_path)
        if os.path.exists(path):
            return path
    return resolve_input_path(DEFAULT_CHECKPOINTS.get(task_key, FALLBACK_CHECKPOINTS[fallback_key][0]))


def now_ms():
    return time.perf_counter() * 1000.0


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_component(rows, component):
    values = [float(r[f"{component}_ms"]) for r in rows if int(r["is_warmup"]) == 0]
    return {
        "component": component,
        "frames": len(values),
        "mean_ms": float(np.mean(values)) if values else float("nan"),
        "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
    }


def summarize_component_filtered(rows, component, row_filter):
    values = [
        float(r[f"{component}_ms"])
        for r in rows
        if int(r["is_warmup"]) == 0 and row_filter(r)
    ]
    return {
        "component": component,
        "frames": len(values),
        "mean_ms": float(np.mean(values)) if values else float("nan"),
        "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
    }


def write_outputs(output_dir, rows, config):
    os.makedirs(output_dir, exist_ok=True)
    frames_csv = os.path.join(output_dir, "runtime_frames.csv")
    fieldnames = [
        "task",
        "frame",
        "is_warmup",
        "localizer_refreshed",
        "localizer_visible",
        "dino_visible_before_stage2",
        "build_image_ms",
        "dino_ms",
        "dino_target_ms",
        "dino_distractor_ms",
        "clip_ms",
        "filter_state_ms",
        "localizer_ms",
        "tensor_ms",
        "policy_ms",
        "env_step_ms",
        "total_ms",
        "reward",
        "done",
    ]
    with open(frames_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    components = [
        "build_image",
        "dino",
        "clip",
        "filter_state",
        "localizer",
        "tensor",
        "policy",
        "env_step",
        "total",
    ]
    summary_rows = [summarize_component(rows, name) for name in components]
    summary_csv = os.path.join(output_dir, "runtime_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    refresh_components = ["dino", "clip", "filter_state", "localizer", "policy", "total"]
    refresh_rows = [
        summarize_component_filtered(
            rows,
            name,
            row_filter=lambda row: int(row["localizer_refreshed"]) == 1,
        )
        for name in refresh_components
    ]
    refresh_summary_csv = os.path.join(output_dir, "runtime_refresh_summary.csv")
    with open(refresh_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(refresh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(refresh_rows)

    summary_json = os.path.join(output_dir, "runtime_summary.json")
    payload = {"config": config, "summary": summary_rows, "refresh_summary": refresh_rows}
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    print(f"[Saved] {frames_csv}")
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {refresh_summary_csv}")
    print(f"[Saved] {summary_json}")


def main():
    parser = argparse.ArgumentParser(description="Profile SGNav runtime by component in Unity.")
    parser.add_argument("--task", default="task1")
    parser.add_argument("--layout", default=os.environ.get("HARBOR_LAYOUT", "layout_a"))
    parser.add_argument("--distractor", default=os.environ.get("HARBOR_DISTRACTOR", "none"))
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--behavior", default=os.environ.get("UNITY_BEHAVIOR_NAME", "USV?team=0"))
    parser.add_argument("--unity-file", default=os.environ.get("UNITY_FILE", ""))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("UNITY_BASE_PORT", "5004")))
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("UNITY_WORKER_ID", "0")))
    parser.add_argument("--time-scale", type=float, default=float(os.environ.get("UNITY_TIME_SCALE", "20.0")))
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--refresh-every-steps", type=int, default=5)
    parser.add_argument("--cache-visible-steps", type=int, default=10)
    parser.add_argument("--quiet-env-step", action="store_true", default=True)
    args = parser.parse_args()

    task_label = normalize_name(args.task)
    layout_label = normalize_layout_name(args.layout)
    distractor_label = normalize_distractor_name(args.distractor)
    checkpoint = resolve_input_path(args.checkpoint) if args.checkpoint.strip() else default_checkpoint(args.task)
    output_dir = resolve_output_path(args.output_dir or f"results/runtime/sgnav_runtime_{task_label}")
    unity_file = args.unity_file.strip() or None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("[Runtime Config]")
    print("  task:", task_label, "task_id:", harbor_task_id(args.task))
    print("  layout:", layout_label, "layout_id:", harbor_layout_id(args.layout))
    print("  distractor:", distractor_label, "distractor_id:", harbor_distractor_id(args.distractor))
    print("  checkpoint:", checkpoint)
    print("  device:", device)
    print("  frames:", args.frames, "warmup:", args.warmup_frames)
    print("  output_dir:", output_dir)

    env = UnitySingleAgentEnv(
        file_name=unity_file,
        behavior_name=args.behavior,
        no_graphics=args.no_graphics,
        worker_id=args.worker_id,
        base_port=args.base_port,
        time_scale=args.time_scale,
        env_params={
            "harbor_task_id": float(harbor_task_id(args.task)),
            "harbor_layout_id": float(harbor_layout_id(args.layout)),
            "harbor_distractor_id": float(harbor_distractor_id(args.distractor)),
            "use_oracle_target_observation": 0.0,
        },
    )

    rows = []
    try:
        obs = env.reset()
        raw_vec_dim = int(obs.vector.shape[0])
        action_dim = int(env.action_size)
        localizer_cfg = build_harbor_task_config(task_name=args.task, device=device)
        localizer = HarborTargetLocalizer(localizer_cfg)
        localizer_adapter = LocalizerPolicyAdapter(
            localizer=localizer,
            refresh_every_steps=args.refresh_every_steps,
            cache_visible_steps=args.cache_visible_steps,
        )
        model = load_model(
            checkpoint,
            action_dim=action_dim,
            vec_dim=raw_vec_dim + TARGET_STATE_DIM,
            device=device,
            use_clip=False,
        )

        total_frames = int(args.frames) + int(args.warmup_frames)
        for frame in range(total_frames):
            total_t0 = now_ms()

            t0 = now_ms()
            img_chw = hwc_to_chw(obs.image)
            build_image_ms = now_ms() - t0

            t0 = now_ms()
            vec_np, localizer_result, localizer_refreshed = build_policy_vector(
                obs,
                localizer_adapter=localizer_adapter,
            )
            localizer_ms = now_ms() - t0
            profile = localizer.last_profile if bool(localizer_refreshed) else {}

            t0 = now_ms()
            img_t = torch.from_numpy(img_chw).unsqueeze(0)
            vec_t = torch.from_numpy(vec_np).unsqueeze(0)
            tensor_ms = now_ms() - t0

            t0 = now_ms()
            with torch.no_grad():
                action_t, _, _ = model.act(img_t, vec_t, deterministic=True)
            if device == "cuda":
                torch.cuda.synchronize()
            action = action_t.squeeze(0).cpu().numpy()
            policy_ms = now_ms() - t0

            t0 = now_ms()
            if args.quiet_env_step:
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with redirect_stdout(devnull):
                        obs, reward, done, _ = env.step(action)
            else:
                obs, reward, done, _ = env.step(action)
            env_step_ms = now_ms() - t0
            total_ms = now_ms() - total_t0

            if done:
                obs = env.reset()
                localizer_adapter.reset()

            rows.append(
                {
                    "task": task_label,
                    "frame": frame - int(args.warmup_frames),
                    "is_warmup": int(frame < int(args.warmup_frames)),
                    "localizer_refreshed": int(bool(localizer_refreshed)),
                    "localizer_visible": int(bool(getattr(localizer_result, "visible", False))),
                    "dino_visible_before_stage2": int(bool(getattr(localizer_result, "dino_visible_before_stage2", False))),
                    "build_image_ms": build_image_ms,
                    "dino_ms": float(profile.get("dino_ms", 0.0)),
                    "dino_target_ms": float(profile.get("dino_target_ms", 0.0)),
                    "dino_distractor_ms": float(profile.get("dino_distractor_ms", 0.0)),
                    "clip_ms": float(profile.get("clip_ms", 0.0)),
                    "filter_state_ms": float(profile.get("filter_state_ms", localizer_ms if bool(localizer_refreshed) else 0.0)),
                    "localizer_ms": localizer_ms,
                    "tensor_ms": tensor_ms,
                    "policy_ms": policy_ms,
                    "env_step_ms": env_step_ms,
                    "total_ms": total_ms,
                    "reward": float(reward),
                    "done": int(bool(done)),
                }
            )

            if (frame + 1) % 50 == 0 or frame + 1 == total_frames:
                measured = max(0, frame + 1 - int(args.warmup_frames))
                print(f"[Progress] task={task_label} measured_frames={measured}/{args.frames}")

        config = {
            "task": task_label,
            "layout": layout_label,
            "distractor": distractor_label,
            "checkpoint": checkpoint,
            "frames": int(args.frames),
            "warmup_frames": int(args.warmup_frames),
            "time_scale": float(args.time_scale),
            "no_graphics": bool(args.no_graphics),
            "refresh_every_steps": int(args.refresh_every_steps),
            "cache_visible_steps": int(args.cache_visible_steps),
        }
        write_outputs(output_dir, rows, config)
    finally:
        env.close()


if __name__ == "__main__":
    main()
