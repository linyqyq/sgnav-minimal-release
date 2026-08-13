import argparse
import csv
import json
import math
import os
import shutil
import time
from contextlib import redirect_stdout

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional visualization dependency
    Image = None
    ImageDraw = None
    ImageFont = None

from envs.unity_ma_env import UnitySingleAgentEnv
from models.actor_critic import ActorCritic
from models.target_localizer_harbor import (
    HarborTargetLocalizer,
    TARGET_STATE_DIM,
    build_harbor_task_config,
)
from train_ppo_dino_harbor_editor import (
    LocalizerPolicyAdapter,
    UNITY_SUMMARY_PATH,
    _read_unity_summary_rows,
    build_policy_vector,
    harbor_task_id,
    hwc_to_chw,
    unity_summary_path_for_file,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

TASK_TARGET_XZ = {
    "magenta_ball": (-55.0, -150.0),
    "task1": (-55.0, -150.0),
    "green_buoy": (-50.0, -170.0),
    "task2": (-50.0, -170.0),
    "yellow_tugboat": (-65.0, -135.0),
    "tugboat": (-65.0, -135.0),
    "vessel03": (-65.0, -135.0),
    "vessl03": (-65.0, -135.0),
    "yellow_tugboat_marker": (-65.0, -135.0),
    "tugboat_marker": (-65.0, -135.0),
    "task3_marker": (-65.0, -135.0),
    "task3": (-65.0, -135.0),
}
USV_START_XZ = (-70.0, -150.0)

LAYOUT_START_XZ = {
    "layout_a": (-70.0, -150.0),
    "layout_b": (-70.0, -150.0),
    "layout_c": (-65.0, -135.0),
    "layout_d": (-60.0, -125.0),
}

LAYOUT_TARGET_XZ = {
    "layout_a": {
        "task1": (-55.0, -150.0),
        "magenta_ball": (-55.0, -150.0),
        "task2": (-50.0, -160.0),
        "green_buoy": (-50.0, -160.0),
        "task3": (-65.0, -135.0),
        "yellow_tugboat": (-65.0, -135.0),
        "yellow_tugboat_marker": (-65.0, -135.0),
    },
    "layout_b": {
        "task1": (-30.0, -170.0),
        "magenta_ball": (-30.0, -170.0),
        "task2": (-30.0, -170.0),
        "green_buoy": (-30.0, -170.0),
        "task3": (-30.0, -170.0),
        "yellow_tugboat": (-30.0, -170.0),
        "yellow_tugboat_marker": (-30.0, -170.0),
    },
    "layout_c": {
        "task1": (-30.0, -150.0),
        "magenta_ball": (-30.0, -150.0),
        "task2": (-30.0, -150.0),
        "green_buoy": (-30.0, -150.0),
        "task3": (-30.0, -150.0),
        "yellow_tugboat": (-30.0, -150.0),
        "yellow_tugboat_marker": (-30.0, -150.0),
    },
    "layout_d": {
        "task1": (-30.0, -160.0),
        "magenta_ball": (-30.0, -160.0),
        "task2": (-30.0, -160.0),
        "green_buoy": (-30.0, -160.0),
        "task3": (-30.0, -160.0),
        "yellow_tugboat": (-30.0, -160.0),
        "yellow_tugboat_marker": (-30.0, -160.0),
    },
}

HARBOR_DISTRACTOR_IDS = {
    "none": 0,
    "off": 0,
    "0": 0,
    "colour": 1,
    "color": 1,
    "1": 1,
    "shape": 2,
    "2": 2,
    "semantic": 3,
    "3": 3,
}


def resolve_input_path(path):
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(path),
        os.path.join(SCRIPT_DIR, path),
        os.path.join(REPO_ROOT, path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def resolve_output_path(path):
    if os.path.isabs(path):
        return path
    if path.startswith(f"python{os.sep}") or path == "python":
        return os.path.join(REPO_ROOT, path)
    return os.path.abspath(path)


def _summary_count(summary_path):
    rows = _read_unity_summary_rows(summary_path)
    if rows is None:
        return 0
    return len(rows)


def _poll_next_summary(summary_path, last_count, timeout_s=3.0):
    deadline = time.time() + timeout_s
    local_last_count = max(0, int(last_count))
    while time.time() < deadline:
        rows = _read_unity_summary_rows(summary_path)
        if rows is None:
            time.sleep(0.05)
            continue
        if len(rows) < local_last_count:
            local_last_count = 0
        if len(rows) > local_last_count:
            return rows[local_last_count], local_last_count + 1
        time.sleep(0.05)
    return None, local_last_count


def normalize_name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_distractor_name(value):
    key = normalize_name(value)
    if key == "color":
        key = "colour"
    if key in ("0", "off"):
        key = "none"
    return key


def harbor_distractor_id(value):
    key = normalize_distractor_name(value)
    if key not in HARBOR_DISTRACTOR_IDS:
        raise ValueError(f"Unknown distractor '{value}'. Use none, colour, shape, or semantic.")
    return HARBOR_DISTRACTOR_IDS[key]


def task_target_xz(task_name):
    return TASK_TARGET_XZ.get(normalize_name(task_name), TASK_TARGET_XZ["magenta_ball"])


def normalize_layout_name(layout_name):
    key = normalize_name(layout_name)
    mapping = {
        "1": "layout_a",
        "a": "layout_a",
        "layouta": "layout_a",
        "layout_a": "layout_a",
        "2": "layout_b",
        "b": "layout_b",
        "layoutb": "layout_b",
        "layout_b": "layout_b",
        "3": "layout_c",
        "c": "layout_c",
        "layoutc": "layout_c",
        "layout_c": "layout_c",
        "4": "layout_d",
        "d": "layout_d",
        "layoutd": "layout_d",
        "layout_d": "layout_d",
    }
    return mapping.get(key, "layout_a")


def harbor_layout_id(layout_name):
    return {
        "layout_a": 1,
        "layout_b": 2,
        "layout_c": 3,
        "layout_d": 4,
    }[normalize_layout_name(layout_name)]


def layout_start_xz(layout_name):
    return LAYOUT_START_XZ[normalize_layout_name(layout_name)]


def layout_target_xz(layout_name, task_name):
    layout_label = normalize_layout_name(layout_name)
    task_label = normalize_name(task_name)
    return LAYOUT_TARGET_XZ[layout_label].get(task_label, task_target_xz(task_name))


def checkpoint_slug(path):
    base = os.path.basename(path)
    if base.endswith(".pt"):
        base = base[:-3]
    return normalize_name(base)


def unity_episode_log_path(summary_path, unity_episode):
    if unity_episode is None or int(unity_episode) < 0:
        return None
    return os.path.join(os.path.dirname(summary_path), f"ep{int(unity_episode)}_log.csv")


def read_trajectory_xz(path):
    points = []
    if not path or not os.path.exists(path):
        return points
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append((float(row["X"]), float(row["Z"])))
            except Exception:
                continue
    return points


def path_length(points):
    total = 0.0
    for (x0, z0), (x1, z1) in zip(points[:-1], points[1:]):
        total += math.hypot(x1 - x0, z1 - z0)
    return total


def straight_line_distance(start_xz, target_xz):
    return math.hypot(target_xz[0] - start_xz[0], target_xz[1] - start_xz[1])


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def image_to_uint8(image_hwc_float):
    img = np.asarray(image_hwc_float)
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    return img


def localizer_result_to_dict(result):
    if result is None:
        return {
            "visible": False,
            "visible_source": "",
            "direction": "",
            "center_x": np.nan,
            "center_y": np.nan,
            "center_x_norm": np.nan,
            "center_y_norm": np.nan,
            "box": None,
            "dino_visible_before_stage2": False,
            "dino_score": np.nan,
            "dino_second_score": np.nan,
            "dino_score_margin": np.nan,
            "stage2_checked": False,
            "stage2_pass": "",
            "stage2_margin": np.nan,
            "stage2_target_score": np.nan,
            "stage2_negative_score": np.nan,
            "fallback_visible": False,
            "cache_age_steps": 0,
        }
    return {
        "visible": bool(result.visible),
        "visible_source": getattr(result, "visible_source", ""),
        "direction": getattr(result, "direction", ""),
        "center_x": float(getattr(result, "center_xy", (np.nan, np.nan))[0]),
        "center_y": float(getattr(result, "center_xy", (np.nan, np.nan))[1]),
        "center_x_norm": float(getattr(result, "center_norm_xy", (np.nan, np.nan))[0]),
        "center_y_norm": float(getattr(result, "center_norm_xy", (np.nan, np.nan))[1]),
        "box": None if getattr(result, "box", None) is None else [float(v) for v in result.box],
        "dino_visible_before_stage2": bool(getattr(result, "dino_visible_before_stage2", False)),
        "dino_score": float(getattr(result, "dino_score", np.nan)),
        "dino_second_score": float(getattr(result, "dino_second_score", np.nan)),
        "dino_score_margin": float(getattr(result, "dino_score_margin", np.nan)),
        "stage2_checked": bool(getattr(result, "stage2_checked", False)),
        "stage2_pass": getattr(result, "stage2_pass", ""),
        "stage2_margin": float(getattr(result, "stage2_margin", np.nan)),
        "stage2_target_score": float(getattr(result, "stage2_target_score", np.nan)),
        "stage2_negative_score": float(getattr(result, "stage2_negative_score", np.nan)),
        "fallback_visible": bool(getattr(result, "fallback_visible", False)),
        "cache_age_steps": int(getattr(result, "cache_age_steps", 0)),
    }


def save_grounding_overlay(path, image_hwc_float, result, title):
    if Image is None:
        return False
    img_u8 = image_to_uint8(image_hwc_float)
    pil = Image.fromarray(img_u8)
    draw = ImageDraw.Draw(pil)

    if result is not None and getattr(result, "box", None) is not None:
        x1, y1, x2, y2 = [float(v) for v in result.box]
        color = (40, 230, 80) if getattr(result, "visible", False) else (240, 190, 40)
        for offset in range(2):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

    caption = title
    if result is not None:
        caption += (
            f" | visible={bool(getattr(result, 'visible', False))}"
            f" source={getattr(result, 'visible_source', '')}"
            f" score={float(getattr(result, 'dino_score', 0.0)):.3f}"
        )
    draw.rectangle([0, 0, pil.size[0], 18], fill=(0, 0, 0))
    draw.text((4, 3), caption[:140], fill=(255, 255, 255))

    ensure_dir(os.path.dirname(path))
    pil.save(path)
    return True


class EvalRecorder:
    def __init__(
        self,
        run_dir,
        method,
        task,
        layout,
        distractor,
        checkpoint_name,
        start_xz=None,
        target_xz=None,
        use_clip=False,
        save_trajectories=False,
        save_grounding_frames=False,
        save_failure_cases=False,
        grounding_every=25,
    ):
        self.run_dir = run_dir
        self.method = method
        self.task = task
        self.layout = layout
        self.distractor = distractor
        self.checkpoint_name = checkpoint_name
        self.start_xz = start_xz if start_xz is not None else USV_START_XZ
        self.target_xz = target_xz if target_xz is not None else task_target_xz(task)
        self.use_clip = bool(use_clip)
        self.save_trajectories = save_trajectories
        self.save_grounding_frames = save_grounding_frames
        self.save_failure_cases = save_failure_cases
        self.grounding_every = max(1, int(grounding_every))
        self.episode_rows = []
        self.grounding_rows = []
        self.last_grounding_frame = {}

        if self.run_dir:
            ensure_dir(self.run_dir)
            ensure_dir(self.episode_log_dir)
            if self.save_grounding_frames:
                ensure_dir(self.grounding_dir)
            if self.save_failure_cases:
                ensure_dir(self.failure_dir)

    @property
    def episode_log_dir(self):
        return os.path.join(self.run_dir, "episode_logs")

    @property
    def grounding_dir(self):
        return os.path.join(self.run_dir, "grounding_frames")

    @property
    def failure_dir(self):
        return os.path.join(self.run_dir, "failure_cases")

    def maybe_record_grounding(self, episode, step_idx, obs_image, result):
        if not self.run_dir or result is None:
            return

        info = localizer_result_to_dict(result)
        row = {
            "method": self.method,
            "task": self.task,
            "layout": self.layout,
            "distractor": self.distractor,
            "checkpoint": self.checkpoint_name,
            "use_clip": int(self.use_clip),
            "episode": episode,
            "step": step_idx,
            **{k: v for k, v in info.items() if k != "box"},
            "box_json": json.dumps(info["box"]),
        }
        self.grounding_rows.append(row)

        if self.save_grounding_frames and step_idx % self.grounding_every == 0:
            frame_path = os.path.join(
                self.grounding_dir,
                f"ep{episode:03d}_step{step_idx:04d}.png",
            )
            if save_grounding_overlay(
                frame_path,
                obs_image,
                result,
                title=f"{self.method} {self.task} ep{episode:03d} step{step_idx:04d}",
            ):
                self.last_grounding_frame[episode] = frame_path

    def finalize_episode(
        self,
        episode,
        success,
        reason,
        unity_episode,
        unity_reward,
        python_reward,
        steps,
        source_episode_log,
        localizer_seen,
        localizer_visible,
        localizer_dino_visible,
    ):
        dest_episode_log = ""
        points = []
        if self.run_dir and source_episode_log and os.path.exists(source_episode_log):
            dest_episode_log = os.path.join(self.episode_log_dir, f"ep{episode:03d}_log.csv")
            if self.save_trajectories:
                shutil.copy2(source_episode_log, dest_episode_log)
            else:
                dest_episode_log = source_episode_log
            points = read_trajectory_xz(dest_episode_log)
        elif source_episode_log:
            points = read_trajectory_xz(source_episode_log)

        direct_dist = straight_line_distance(self.start_xz, self.target_xz)
        traj_len = path_length(points)
        path_eff = direct_dist / traj_len if traj_len > 1e-6 else np.nan

        visible_ratio = localizer_visible / max(localizer_seen, 1)
        dino_visible_ratio = localizer_dino_visible / max(localizer_seen, 1)
        row = {
            "method": self.method,
            "task": self.task,
            "layout": self.layout,
            "distractor": self.distractor,
            "checkpoint": self.checkpoint_name,
            "use_clip": int(self.use_clip),
            "episode": episode,
            "success": int(success),
            "termination_reason": reason,
            "unity_episode": unity_episode,
            "unity_reward": unity_reward,
            "python_reward": python_reward,
            "steps": steps,
            "ct_steps": steps,
            "path_length": traj_len,
            "straight_line_distance": direct_dist,
            "path_efficiency": path_eff,
            "localizer_visible_ratio": visible_ratio,
            "localizer_dino_visible_ratio": dino_visible_ratio,
            "episode_log": dest_episode_log,
        }
        self.episode_rows.append(row)

        if self.run_dir and self.save_failure_cases and not success:
            failure_json = os.path.join(self.failure_dir, f"ep{episode:03d}_failure.json")
            failure_payload = dict(row)
            failure_payload["last_grounding_frame"] = self.last_grounding_frame.get(episode, "")
            with open(failure_json, "w", encoding="utf-8") as f:
                json.dump(failure_payload, f, indent=2, allow_nan=True)
            last_frame = self.last_grounding_frame.get(episode)
            if last_frame and os.path.exists(last_frame):
                shutil.copy2(last_frame, os.path.join(self.failure_dir, f"ep{episode:03d}_last_grounding.png"))

        return row

    def save(self):
        if not self.run_dir:
            return

        episode_csv = os.path.join(self.run_dir, "episodes.csv")
        if self.episode_rows:
            with open(episode_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.episode_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.episode_rows)

        grounding_csv = os.path.join(self.run_dir, "grounding.csv")
        if self.grounding_rows:
            with open(grounding_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.grounding_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.grounding_rows)

        success_rows = [r for r in self.episode_rows if int(r["success"]) == 1]
        metrics = {
            "method": self.method,
            "task": self.task,
            "layout": self.layout,
            "distractor": self.distractor,
            "checkpoint": self.checkpoint_name,
            "start_xz": list(self.start_xz),
            "target_xz": list(self.target_xz),
            "use_clip": bool(self.use_clip),
            "episodes": len(self.episode_rows),
            "SR": float(np.mean([r["success"] for r in self.episode_rows])) if self.episode_rows else np.nan,
            "CT_steps_success_mean": float(np.mean([r["ct_steps"] for r in success_rows])) if success_rows else np.nan,
            "PE_success_mean": float(np.nanmean([r["path_efficiency"] for r in success_rows])) if success_rows else np.nan,
            "localizer_visible_ratio_mean": float(np.mean([r["localizer_visible_ratio"] for r in self.episode_rows])) if self.episode_rows else np.nan,
            "localizer_dino_visible_ratio_mean": float(np.mean([r["localizer_dino_visible_ratio"] for r in self.episode_rows])) if self.episode_rows else np.nan,
            "termination_reasons": {},
        }
        for row in self.episode_rows:
            reason = row["termination_reason"]
            metrics["termination_reasons"][reason] = metrics["termination_reasons"].get(reason, 0) + 1

        with open(os.path.join(self.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, allow_nan=True)


def load_model(path, action_dim, vec_dim, device, use_clip=False):
    model = ActorCritic(
        action_dim=action_dim,
        vec_dim=vec_dim,
        use_clip=use_clip,
        clip_model="ViT-B-32",
        clip_pretrained="openai",
        freeze_clip=True,
        proj_dim=128,
        trunk_dim=256,
        device=device,
    )
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def run_episode(
    env,
    model,
    localizer_adapter,
    max_steps,
    last_summary_count,
    summary_path,
    quiet_env_step=True,
    recorder=None,
    episode_index=0,
):
    obs = env.reset()
    if localizer_adapter is not None:
        localizer_adapter.reset()

    total_reward = 0.0
    localizer_seen = 0
    localizer_visible = 0
    localizer_dino_visible = 0

    for step_idx in range(max_steps):
        img_chw = hwc_to_chw(obs.image)
        vec_np, localizer_result, _ = build_policy_vector(
            obs,
            localizer_adapter=localizer_adapter,
        )

        if localizer_result is not None:
            localizer_seen += 1
            localizer_visible += int(bool(localizer_result.visible))
            localizer_dino_visible += int(bool(localizer_result.dino_visible_before_stage2))
            if recorder is not None:
                recorder.maybe_record_grounding(
                    episode=episode_index,
                    step_idx=step_idx,
                    obs_image=obs.image,
                    result=localizer_result,
                )

        img_t = torch.from_numpy(img_chw).unsqueeze(0)
        vec_t = torch.from_numpy(vec_np).unsqueeze(0)

        with torch.no_grad():
            action_t, _, _ = model.act(img_t, vec_t, deterministic=True)
        action = action_t.squeeze(0).cpu().numpy()

        if quiet_env_step:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stdout(devnull):
                    obs, reward, done, _ = env.step(action)
        else:
            obs, reward, done, _ = env.step(action)

        total_reward += float(reward)
        if done:
            row, last_summary_count = _poll_next_summary(summary_path, last_summary_count, timeout_s=1.0)
            if row is None:
                # ML-Agents MaxStep timeouts are backfilled by USVAgent.OnEpisodeBegin().
                obs = env.reset()
                if localizer_adapter is not None:
                    localizer_adapter.reset()
                row, last_summary_count = _poll_next_summary(summary_path, last_summary_count, timeout_s=3.0)

            return {
                "episode_reward": total_reward,
                "steps": step_idx + 1,
                "summary": row,
                "last_summary_count": last_summary_count,
                "truncated_by_python": False,
                "localizer_seen": localizer_seen,
                "localizer_visible": localizer_visible,
                "localizer_dino_visible": localizer_dino_visible,
            }

    # If Python reaches max_steps before Unity terminates the episode, reset once
    # so USVAgent.OnEpisodeBegin can backfill timeout_auto when applicable.
    obs = env.reset()
    if localizer_adapter is not None:
        localizer_adapter.reset()
    row, last_summary_count = _poll_next_summary(summary_path, last_summary_count, timeout_s=3.0)

    return {
        "episode_reward": total_reward,
        "steps": max_steps,
        "summary": row,
        "last_summary_count": last_summary_count,
        "truncated_by_python": True,
        "localizer_seen": localizer_seen,
        "localizer_visible": localizer_visible,
        "localizer_dino_visible": localizer_dino_visible,
    }


def default_checkpoints():
    ckpt_dir = os.path.join(
        SCRIPT_DIR,
        "checkpoints",
        "usv_harbor_dino_sgq_task1_magenta_ball",
    )
    return [
        os.path.join(ckpt_dir, "ckpt_iter_001000.pt"),
        os.path.join(ckpt_dir, "ckpt_iter_001500.pt"),
        os.path.join(ckpt_dir, "ckpt_iter_002000.pt"),
        os.path.join(ckpt_dir, "ckpt_iter_006000.pt"),
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate SGQ harbor checkpoints with Unity summary success.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--task", default=os.environ.get("HARBOR_TASK", "magenta_ball"))
    parser.add_argument("--layout", default=os.environ.get("HARBOR_LAYOUT", "layout_a"))
    parser.add_argument(
        "--distractor",
        default=os.environ.get("HARBOR_DISTRACTOR", "none"),
        help="Exp3 distractor condition: none, colour/color, shape, or semantic.",
    )
    parser.add_argument("--behavior", default=os.environ.get("UNITY_BEHAVIOR_NAME", "USV?team=0"))
    parser.add_argument("--unity-file", default=os.environ.get("UNITY_FILE", ""))
    parser.add_argument(
        "--no-graphics",
        action="store_true",
        default=os.environ.get("UNITY_NO_GRAPHICS", "0").lower() in ("1", "true", "yes"),
    )
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("UNITY_BASE_PORT", "5004")))
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("UNITY_WORKER_ID", "0")))
    parser.add_argument("--time-scale", type=float, default=float(os.environ.get("UNITY_TIME_SCALE", "1.0")))
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "eval_sgq_task1_checkpoints.csv"))
    parser.add_argument("--checkpoint", action="append", dest="checkpoints")
    parser.add_argument("--no-localizer", action="store_true")
    parser.add_argument(
        "--method",
        default=os.environ.get("HARBOR_EVAL_METHOD", ""),
        help="Method label for organized experiment records, e.g. sgnav, no_target_ppo, vision_only_ppo, oracle_ppo.",
    )
    parser.add_argument(
        "--use-clip",
        choices=("auto", "0", "1"),
        default=os.environ.get("HARBOR_EVAL_USE_CLIP", "auto"),
        help="Use ActorCritic CLIP image branch. auto enables it for --method vision_only_ppo only.",
    )
    parser.add_argument(
        "--save-run-dir",
        default=os.environ.get("HARBOR_EVAL_SAVE_RUN_DIR", ""),
        help="Root directory for recorder outputs. Defaults to results/exp1/{method}/{task}.",
    )
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--save-grounding-frames", action="store_true")
    parser.add_argument("--save-failure-cases", action="store_true")
    parser.add_argument(
        "--grounding-frame-every",
        type=int,
        default=int(os.environ.get("HARBOR_EVAL_GROUNDING_FRAME_EVERY", "25")),
        help="Save one grounding overlay every N policy steps when --save-grounding-frames is enabled.",
    )
    parser.add_argument(
        "--oracle-target-observation",
        choices=("auto", "0", "1"),
        default="auto",
        help="Use Unity oracle target slots. auto enables it only for --no-localizer baseline eval.",
    )
    parser.add_argument("--verbose-env-step", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unity_file = args.unity_file.strip() or None
    summary_path = unity_summary_path_for_file(unity_file)
    task_id = harbor_task_id(args.task)
    layout_label = normalize_layout_name(args.layout)
    layout_id = harbor_layout_id(args.layout)
    distractor_label = normalize_distractor_name(args.distractor)
    distractor_id = harbor_distractor_id(args.distractor)
    start_xz = layout_start_xz(args.layout)
    target_xz = layout_target_xz(args.layout, args.task)
    if args.oracle_target_observation == "auto":
        use_oracle_target_observation = bool(args.no_localizer)
    else:
        use_oracle_target_observation = args.oracle_target_observation == "1"
    task_label = normalize_name(args.task)
    if args.method.strip():
        method_label = normalize_name(args.method)
    elif args.no_localizer and use_oracle_target_observation:
        method_label = "oracle_ppo"
    elif args.no_localizer:
        method_label = "no_target_ppo"
    else:
        method_label = "sgnav"
    if args.use_clip == "auto":
        use_clip = method_label in ("vision_only_ppo", "visiononly_ppo", "vision_only")
    else:
        use_clip = args.use_clip == "1"

    save_run_root = args.save_run_dir.strip()
    if save_run_root:
        save_run_root = resolve_output_path(save_run_root)
    elif args.save_trajectories or args.save_grounding_frames or args.save_failure_cases:
        save_run_root = os.path.join(REPO_ROOT, "results", "exp1", method_label, task_label)
    else:
        save_run_root = ""

    env = UnitySingleAgentEnv(
        file_name=unity_file,
        behavior_name=args.behavior,
        no_graphics=args.no_graphics,
        worker_id=args.worker_id,
        time_scale=args.time_scale,
        base_port=args.base_port,
        env_params={
            "harbor_task_id": float(task_id),
            "harbor_layout_id": float(layout_id),
            "harbor_distractor_id": float(distractor_id),
            "use_oracle_target_observation": 1.0 if use_oracle_target_observation else 0.0,
        },
    )

    obs = env.reset()
    raw_vec_dim = int(obs.vector.shape[0])
    action_dim = int(env.action_size)

    if args.no_localizer:
        vec_dim = raw_vec_dim
        localizer_adapter = None
    else:
        vec_dim = raw_vec_dim + TARGET_STATE_DIM
        localizer_cfg = build_harbor_task_config(task_name=args.task, device=device)
        localizer = HarborTargetLocalizer(localizer_cfg)
        localizer_adapter = LocalizerPolicyAdapter(
            localizer=localizer,
            refresh_every_steps=5,
            cache_visible_steps=10,
        )

    checkpoints = args.checkpoints or default_checkpoints()
    output_path = resolve_output_path(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("[Eval Config]")
    print("  method:", method_label)
    print("  task:", task_label, "task_id:", task_id)
    print("  layout:", layout_label, "layout_id:", layout_id)
    print("  distractor:", distractor_label, "distractor_id:", distractor_id)
    print("  no_localizer:", bool(args.no_localizer))
    print("  oracle_target_observation:", bool(use_oracle_target_observation))
    print("  use_clip:", bool(use_clip))
    print("  save_run_root:", save_run_root or "(disabled)")
    if method_label in ("vision_only_ppo", "visiononly_ppo", "vision_only"):
        print("  vision_only_note: localizer should be disabled and oracle target observation should be 0.")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "task",
            "layout",
            "distractor",
            "use_clip",
            "checkpoint",
            "episode",
            "success",
            "termination_reason",
            "unity_episode",
            "unity_reward",
            "python_reward",
            "steps",
            "ct_steps",
            "path_length",
            "path_efficiency",
            "localizer_visible_ratio",
            "localizer_dino_visible_ratio",
            "run_dir",
            "episode_log",
        ])

        try:
            for ckpt in checkpoints:
                ckpt_path = resolve_input_path(ckpt)
                ckpt_name = checkpoint_slug(ckpt_path)
                ckpt_run_dir = os.path.join(save_run_root, ckpt_name) if save_run_root else ""
                recorder = EvalRecorder(
                    run_dir=ckpt_run_dir,
                    method=method_label,
                    task=task_label,
                    layout=layout_label,
                    distractor=distractor_label,
                    checkpoint_name=os.path.basename(ckpt_path),
                    start_xz=start_xz,
                    target_xz=target_xz,
                    use_clip=use_clip,
                    save_trajectories=args.save_trajectories,
                    save_grounding_frames=args.save_grounding_frames and not args.no_localizer,
                    save_failure_cases=args.save_failure_cases,
                    grounding_every=args.grounding_frame_every,
                )
                model = load_model(
                    ckpt_path,
                    action_dim=action_dim,
                    vec_dim=vec_dim,
                    device=device,
                    use_clip=use_clip,
                )
                last_summary_count = _summary_count(summary_path)
                successes = 0
                reasons = {}

                print(f"[Eval] checkpoint={ckpt_path}")
                if ckpt_run_dir:
                    print(f"  recorder_dir={ckpt_run_dir}")
                for ep in range(1, args.episodes + 1):
                    result = run_episode(
                        env=env,
                        model=model,
                        localizer_adapter=localizer_adapter,
                        max_steps=args.max_steps,
                        last_summary_count=last_summary_count,
                        summary_path=summary_path,
                        quiet_env_step=not args.verbose_env_step,
                        recorder=recorder,
                        episode_index=ep,
                    )
                    last_summary_count = int(result["last_summary_count"])
                    summary = result["summary"]
                    if summary is None:
                        success = 0
                        reason = "timeout_python" if result.get("truncated_by_python", False) else "missing_summary"
                        unity_episode = -1
                        unity_reward = np.nan
                    else:
                        success = int(summary["unity_success"] == 1)
                        reason = summary["reason"]
                        unity_episode = summary["episode_idx"]
                        unity_reward = summary["cumulative_reward"]

                    successes += success
                    reasons[reason] = reasons.get(reason, 0) + 1
                    visible_ratio = result["localizer_visible"] / max(result["localizer_seen"], 1)
                    dino_visible_ratio = result["localizer_dino_visible"] / max(result["localizer_seen"], 1)
                    source_episode_log = unity_episode_log_path(summary_path, unity_episode)
                    rec_row = recorder.finalize_episode(
                        episode=ep,
                        success=success,
                        reason=reason,
                        unity_episode=unity_episode,
                        unity_reward=unity_reward,
                        python_reward=result["episode_reward"],
                        steps=result["steps"],
                        source_episode_log=source_episode_log,
                        localizer_seen=result["localizer_seen"],
                        localizer_visible=result["localizer_visible"],
                        localizer_dino_visible=result["localizer_dino_visible"],
                    )

                    writer.writerow([
                        method_label,
                        task_label,
                        layout_label,
                        distractor_label,
                        int(use_clip),
                        os.path.basename(ckpt_path),
                        ep,
                        success,
                        reason,
                        unity_episode,
                        unity_reward,
                        result["episode_reward"],
                        result["steps"],
                        rec_row["ct_steps"],
                        rec_row["path_length"],
                        rec_row["path_efficiency"],
                        visible_ratio,
                        dino_visible_ratio,
                        ckpt_run_dir,
                        rec_row["episode_log"],
                    ])
                    f.flush()

                    print(
                        f"  ep={ep:03d} success={success} reason={reason} "
                        f"running_success={successes / ep:.3f} "
                        f"visible={visible_ratio:.3f} dino_visible={dino_visible_ratio:.3f}"
                    )

                print(
                    f"[Eval Summary] {os.path.basename(ckpt_path)} "
                    f"success={successes}/{args.episodes} rate={successes / max(args.episodes, 1):.3f} "
                        f"reasons={reasons}"
                )
                recorder.save()
        finally:
            env.close()

    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
