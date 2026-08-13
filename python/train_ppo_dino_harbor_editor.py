import os
import time
import json
from collections import deque
import numpy as np
import torch

from mlagents_envs.exception import UnityWorkerInUseException

from envs.unity_ma_env import UnitySingleAgentEnv
from models.actor_critic import ActorCritic
from models.target_localizer_harbor import (
    HarborTargetLocalizer,
    TARGET_STATE_DIM,
    build_harbor_task_config,
)
from algo.rollout_buffer import RolloutBuffer
from algo.ppo import PPO

import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LOG_PATH = os.path.join(SCRIPT_DIR, "harbor_training_log.csv")
CLIP_DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, "harbor_clip_debug_log.csv")
LOCALIZER_DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, "harbor_target_localizer_train_log.csv")

TASK_LOG_NAMES = {
    "magenta_ball": "task1_magenta_ball",
    "task1": "task1_magenta_ball",
    "green_buoy": "task2_green_buoy",
    "task2": "task2_green_buoy",
    "yellow_tugboat": "task3_yellow_tugboat",
    "tugboat": "task3_yellow_tugboat",
    "vessel03": "task3_yellow_tugboat",
    "vessl03": "task3_yellow_tugboat",
    "yellow_tugboat_marker": "task3_yellow_tugboat_marker",
    "tugboat_marker": "task3_yellow_tugboat_marker",
    "task3_marker": "task3_yellow_tugboat_marker",
    "task3": "task3_yellow_tugboat_marker",
}


def normalize_task_key(task_name: str) -> str:
    return task_name.strip().lower().replace("-", "_")


def task_log_name(task_name: str) -> str:
    task_key = normalize_task_key(task_name)
    return TASK_LOG_NAMES.get(task_key, task_key)


def harbor_task_id(task_name: str) -> int:
    task_key = normalize_task_key(task_name)
    if task_key in ("green_buoy", "task2"):
        return 2
    if task_key in (
        "yellow_tugboat",
        "tugboat",
        "vessel03",
        "vessl03",
        "yellow_tugboat_marker",
        "tugboat_marker",
        "task3_marker",
        "task3",
    ):
        return 3
    return 1


def normalize_run_variant(variant_name: str) -> str:
    variant = normalize_task_key(variant_name or "sgq")
    return variant if variant else "sgq"

# CLIP_DEBUG_HEADER = [
#     "global_step",
#     "iter",
#     "reward",
#     "dist",
#     "heading_err",
#     "fwd_speed",
#     "yaw_rate",
#     "weights_mean",
#     "weights_std",
#     "weights_min",
#     "weights_max",
#     "fg_mean",
#     "bg_mean",
# ]

CLIP_DEBUG_HEADER = [
    "global_step",
    "iter",
    "reward",
    "dist",
    "heading_err",
    "fwd_speed",
    "yaw_rate",

    "target_weights_mean",
    "target_weights_std",
    "target_weights_min",
    "target_weights_max",
    "target_fg_mean",
    "target_bg_mean",

    "obstacle_weights_mean",
    "obstacle_weights_std",
    "obstacle_weights_min",
    "obstacle_weights_max",
    "obstacle_fg_mean",
    "obstacle_bg_mean",
]

LOCALIZER_DEBUG_HEADER = [
    "global_step",
    "iter",
    "reward",
    "visible",
    "dino_visible_before_stage2",
    "direction",
    "center_x_norm",
    "center_y_norm",
    "dino_score",
    "stage2_margin",
    "stage2_checked",
    "stage2_pass",
    "visible_source",
    "cache_age_steps",
    "fallback_visible",
    "fallback_magenta_ratio",
    "dino_magenta_ratio",
    "raw_vector_dim",
    "aug_vector_dim",
    "state_json",
    "box_json",
    "stage2_negative_prompt",
]

UNITY_SUMMARY_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "unity", "Assets", "EpisodeLogs", "summary.csv")
)


def unity_summary_path_for_file(unity_file):
    if not unity_file:
        return os.path.abspath(
            os.path.join(SCRIPT_DIR, "..", "unity", "Assets", "EpisodeLogs", "summary.csv")
        )

    build_dir = os.path.dirname(os.path.abspath(unity_file))
    build_name = os.path.basename(unity_file)
    if build_name.endswith(".x86_64"):
        build_name = build_name[:-len(".x86_64")]
    return os.path.join(build_dir, f"{build_name}_Data", "EpisodeLogs", "summary.csv")
LOG_HEADER = [
    "iter",
    "global_step",
    "logged_episode_count",
    "episode_reward",
    "success",
    "success_rate",
    "terminal_reward",
    "unity_episode",
    "unity_success",
    "termination_reason",
]
RESET_LOG_ON_START = os.environ.get("RESET_LOG_ON_START", "1").lower() not in ("0", "false", "no")


def init_training_log(path: str, reset: bool = True):
    """Initialize log file and optionally clear stale content from previous runs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if reset or not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADER)
            f.flush()
            os.fsync(f.fileno())
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)

    if first_row != LOG_HEADER:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADER)
            f.flush()
            os.fsync(f.fileno())


def append_training_log(
    path: str,
    it: int,
    global_step: int,
    logged_episode_count: int,
    episode_reward: float,
    success: int,
    success_rate: float,
    terminal_reward: float,
    unity_episode: int,
    unity_success: int,
    termination_reason: str,
):
    """Append one training row and force sync so file watchers see updates immediately."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                it,
                global_step,
                logged_episode_count,
                episode_reward,
                success,
                success_rate,
                terminal_reward,
                unity_episode,
                unity_success,
                termination_reason,
            ]
        )
        f.flush()
        os.fsync(f.fileno())


def init_clip_debug_log(path: str, reset: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if reset or not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CLIP_DEBUG_HEADER)
            f.flush()
            os.fsync(f.fileno())


def append_clip_debug_log(
    path: str,
    global_step: int,
    it: int,
    reward: float,
    dist: float,
    heading_err: float,
    fwd_speed: float,
    yaw_rate: float,
    clip_debug: dict,
):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            global_step,
            it,
            reward,
            dist,
            heading_err,
            fwd_speed,
            yaw_rate,
            clip_debug.get("target_weights_mean"),
            clip_debug.get("target_weights_std"),
            clip_debug.get("target_weights_min"),
            clip_debug.get("target_weights_max"),
            clip_debug.get("target_fg_mean"),
            clip_debug.get("target_bg_mean"),
            clip_debug.get("obstacle_weights_mean"),
            clip_debug.get("obstacle_weights_std"),
            clip_debug.get("obstacle_weights_min"),
            clip_debug.get("obstacle_weights_max"),
            clip_debug.get("obstacle_fg_mean"),
            clip_debug.get("obstacle_bg_mean"),
        ])
        f.flush()
        os.fsync(f.fileno())


def init_localizer_debug_log(path: str, reset: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if reset or not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOCALIZER_DEBUG_HEADER)
            f.flush()
            os.fsync(f.fileno())


def append_localizer_debug_log(
    path: str,
    global_step: int,
    it: int,
    reward: float,
    raw_vector_dim: int,
    aug_vector_dim: int,
    result,
):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            global_step,
            it,
            reward,
            result.visible,
            result.dino_visible_before_stage2,
            result.direction,
            result.center_norm_xy[0],
            result.center_norm_xy[1],
            result.dino_score,
            result.stage2_margin,
            result.stage2_checked,
            result.stage2_pass,
            getattr(result, "visible_source", ""),
            getattr(result, "cache_age_steps", 0),
            getattr(result, "fallback_visible", False),
            getattr(result, "fallback_magenta_ratio", 0.0),
            getattr(result, "dino_magenta_ratio", 0.0),
            raw_vector_dim,
            aug_vector_dim,
            json.dumps(np.round(result.state, 6).tolist()),
            json.dumps(None if result.box is None else [round(v, 2) for v in result.box]),
            result.stage2_negative_prompt,
        ])
        f.flush()
        os.fsync(f.fileno())


def _read_unity_summary_rows(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = [row for row in csv.reader(f) if row]
    except Exception:
        return None

    if len(rows) <= 1:
        return []

    parsed = []
    for r in rows[1:]:
        if len(r) < 6:
            continue
        try:
            episode_idx = int(r[0].strip())
            cumulative_reward = float(r[1].strip())
            steps = int(r[3].strip())
        except Exception:
            continue

        success_raw = r[4].strip().lower()
        if success_raw in ("true", "1"):
            unity_success = 1
        elif success_raw in ("false", "0"):
            unity_success = 0
        else:
            unity_success = -1

        reason = r[5].strip() if r[5].strip() else "unknown"
        parsed.append({
            "episode_idx": episode_idx,
            "cumulative_reward": cumulative_reward,
            "steps": steps,
            "unity_success": unity_success,
            "reason": reason,
        })

    return parsed


def poll_unity_episode_result(path: str, last_row_count: int, timeout_s: float = 2.0):
    """Poll Unity summary.csv and consume one new summary row after last_row_count."""
    t_end = time.time() + timeout_s
    local_last_count = max(0, last_row_count)

    while time.time() < t_end:
        rows = _read_unity_summary_rows(path)
        if rows is None:
            time.sleep(0.02)
            continue

        # Handle summary reset (Unity object recreated / file rewritten).
        if len(rows) < local_last_count:
            local_last_count = 0

        if len(rows) > local_last_count:
            row = rows[local_last_count]  # consume next unseen row
            return row, local_last_count + 1

        time.sleep(0.02)
    return None, local_last_count


def get_unity_summary_baseline(path: str):
    rows = _read_unity_summary_rows(path)
    if rows is None:
        return 0, None
    if len(rows) == 0:
        return 0, None
    return len(rows), rows[-1]


def hwc_to_chw(img_hwc: np.ndarray) -> np.ndarray:
    if img_hwc.dtype != np.float32:
        img_hwc = img_hwc.astype(np.float32)
    return np.transpose(img_hwc, (2, 0, 1))


def save_ckpt(path: str, model: ActorCritic, ppo: PPO, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optim": ppo.opt.state_dict() if hasattr(ppo, "opt") else None,
        "meta": meta,
    }
    torch.save(payload, path)


def resolve_input_path(path: str) -> str:
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


def load_ckpt(path: str, model: ActorCritic, ppo: PPO, device: str, load_optimizer: bool = True):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    if load_optimizer and payload.get("optim") is not None and hasattr(ppo, "opt"):
        ppo.opt.load_state_dict(payload["optim"])
    return payload.get("meta", {})


class LocalizerPolicyAdapter:
    def __init__(self, localizer=None, refresh_every_steps: int = 1, cache_visible_steps: int = 10):
        self.localizer = localizer
        self.refresh_every_steps = max(1, int(refresh_every_steps))
        self.cache_visible_steps = max(0, int(cache_visible_steps))
        self.visible_cache = None
        self.visible_cache_age = 0
        self.cached_result = None
        self.call_count = 0

    def reset(self):
        self.visible_cache = None
        self.visible_cache_age = 0
        self.cached_result = None
        self.call_count = 0

    def build_vector(self, obs):
        if self.localizer is None:
            return obs.vector.astype(np.float32), None, False

        should_refresh = (
            self.cached_result is None
            or self.call_count % self.refresh_every_steps == 0
        )
        refreshed_result = None
        if should_refresh:
            refreshed_result = self.localizer.localize(obs.image)
            if refreshed_result.visible:
                self.visible_cache = refreshed_result
                self.visible_cache_age = 0

        self.call_count += 1

        if refreshed_result is not None and refreshed_result.visible:
            result_for_policy = refreshed_result
        elif refreshed_result is not None:
            result_for_policy = refreshed_result
        elif self.visible_cache is not None and self.visible_cache_age < self.cache_visible_steps:
            self.visible_cache_age += 1
            result_for_policy = self.localizer.with_cache_age(
                self.visible_cache,
                age_steps=self.visible_cache_age,
                max_age_steps=self.cache_visible_steps,
            )
        else:
            self.visible_cache = None
            result_for_policy = self.localizer.zero_result(age_norm=1.0)

        self.cached_result = result_for_policy
        vec = self.localizer.augment_vector(obs.vector, result_for_policy)
        return vec.astype(np.float32), result_for_policy, should_refresh


def build_policy_vector(obs, localizer_adapter=None):
    if localizer_adapter is None:
        return obs.vector.astype(np.float32), None, False

    return localizer_adapter.build_vector(obs)


def evaluate(env, model, steps=200, success_dist=1.0, localizer_adapter=None):
    obs = env.reset()
    if localizer_adapter is not None:
        localizer_adapter.reset()

    total_reward = 0
    success = 0

    for _ in range(steps):
        img = torch.from_numpy(hwc_to_chw(obs.image)).unsqueeze(0)
        vec_np, _, _ = build_policy_vector(obs, localizer_adapter=localizer_adapter)
        vec = torch.from_numpy(vec_np).unsqueeze(0)

        with torch.no_grad():
            action_t, _, _ = model.act(img, vec, deterministic=True)

        action = action_t.squeeze(0).cpu().numpy()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # ===== success 判断 =====
        dist = np.linalg.norm(obs.vector[:2])

        if dist < success_dist:
            success = 1   # 👈 到达目标
            print(f"[SUCCESS] dist={dist:.3f}")
            break

        if done:
            break

    print(f"[Eval] return={total_reward:.3f}, success={success}")
    return total_reward, success   # 👈 返回两个

def main():
    global LOG_PATH, CLIP_DEBUG_LOG_PATH, LOCALIZER_DEBUG_LOG_PATH, UNITY_SUMMARY_PATH

    HARBOR_TASK = os.environ.get("HARBOR_TASK", "magenta_ball")
    task_key = normalize_task_key(HARBOR_TASK)
    task_id = harbor_task_id(HARBOR_TASK)
    task_name_for_files = task_log_name(HARBOR_TASK)
    run_variant = normalize_run_variant(os.environ.get("HARBOR_RUN_VARIANT", "sgq"))
    run_name_for_files = f"{run_variant}_{task_name_for_files}"

    LOG_PATH = os.environ.get(
        "HARBOR_TRAINING_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"harbor_training_log_{run_name_for_files}.csv"),
    )
    CLIP_DEBUG_LOG_PATH = os.environ.get(
        "HARBOR_CLIP_DEBUG_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"harbor_clip_debug_log_{run_name_for_files}.csv"),
    )
    LOCALIZER_DEBUG_LOG_PATH = os.environ.get(
        "HARBOR_LOCALIZER_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"harbor_target_localizer_train_log_{run_name_for_files}.csv"),
    )

    init_training_log(LOG_PATH, reset=RESET_LOG_ON_START)
    init_clip_debug_log(CLIP_DEBUG_LOG_PATH, reset=RESET_LOG_ON_START)
    init_localizer_debug_log(LOCALIZER_DEBUG_LOG_PATH, reset=RESET_LOG_ON_START)
    start_time = time.time()
    # ========= USER CONFIG =========
    # Build 模式：设置 UNITY_FILE=/path/to/USV.x86_64
    # Editor 模式：UNITY_FILE 留空（Editor 不能真正 headless）
    UNITY_FILE = os.environ.get("UNITY_FILE", "").strip() or None
    UNITY_SUMMARY_PATH = unity_summary_path_for_file(UNITY_FILE)
    BEHAVIOR = os.environ.get("UNITY_BEHAVIOR_NAME", "USV?team=0")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Build 模式一般不需要纠结 port，给个 worker_id 即可（多进程训练时才要不同）
    WORKER_ID = 0
    MAX_WORKER_RETRIES = 20
    # BASE_PORT = 5005   # build 模式通常用默认即可；如果冲突再改
    BASE_PORT = 5004
    NO_GRAPHICS = os.environ.get("UNITY_NO_GRAPHICS", "0").lower() in ("1", "true", "yes")

    TIME_SCALE = float(os.environ.get("UNITY_TIME_SCALE", "1.0"))
    HORIZON = int(os.environ.get("HARBOR_HORIZON", "128"))
    GAMMA = 0.99
    LAM = 0.95

    # LR = 3e-4
    # PPO_EPOCHS = 4
    # MINIBATCH = 256
    # USE_CLIP = False
    # SUCCESS_REWARD_THRESHOLD = 50.0  # terminal reward >= threshold => success (target hit)

    LR = float(os.environ.get("PPO_LR", os.environ.get("LR", "1e-4")))
    PPO_EPOCHS = 4
    MINIBATCH = 256
    PPO_ENT_COEF = float(os.environ.get("PPO_ENT_COEF", "0.001"))
    PPO_TARGET_KL = float(os.environ.get("PPO_TARGET_KL", "0.03"))

    # ===== FIRST-STEP PPO CONNECTION CONFIG =====
    USE_TARGET_LOCALIZER = os.environ.get("HARBOR_USE_TARGET_LOCALIZER", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    default_oracle_target_observation = "0" if USE_TARGET_LOCALIZER else "1"
    USE_ORACLE_TARGET_OBSERVATION = os.environ.get(
        "HARBOR_USE_ORACLE_TARGET_OBSERVATION",
        default_oracle_target_observation,
    ).lower() in ("1", "true", "yes")
    LOCALIZER_LOG_EVERY_STEPS = int(os.environ.get("LOCALIZER_LOG_EVERY_STEPS", "50"))
    task3_keys = (
        "yellow_tugboat", "tugboat", "vessel03", "vessl03", "task3",
        "yellow_tugboat_marker", "tugboat_marker", "task3_marker",
    )
    default_refresh_every_steps = "20" if task_key in task3_keys else "5"
    default_cache_visible_steps = "20" if task_key in task3_keys else "10"
    LOCALIZER_REFRESH_EVERY_STEPS = int(os.environ.get("LOCALIZER_REFRESH_EVERY_STEPS", default_refresh_every_steps))
    LOCALIZER_CACHE_VISIBLE_STEPS = int(os.environ.get("LOCALIZER_CACHE_VISIBLE_STEPS", default_cache_visible_steps))
    LOCALIZER_DINO_VISIBLE_SCORE = float(os.environ.get("LOCALIZER_DINO_VISIBLE_SCORE", "0.28"))
    LOCALIZER_STAGE2_GATE_MARGIN_THRESHOLD = float(os.environ.get("LOCALIZER_STAGE2_GATE_MARGIN_THRESHOLD", "0.0"))
    LOCALIZER_FALLBACK_MIN_MAGENTA_RATIO = float(os.environ.get("LOCALIZER_FALLBACK_MIN_MAGENTA_RATIO", "0.05"))

    # Keep the old ActorCritic CLIP branch off for the first connection smoke test.
    # This makes it easier to verify that PPO is learning from the 18 + 8 localizer vector.
    USE_CLIP = False
    CLIP_MODEL_NAME = "ViT-B-32"
    CLIP_PRETRAINED = "openai"

    FG_PROMPTS = [
        "red buoy",
        "green buoy",
        "black buoy",
        "blue target",
        "floating navigation marker",
        "colored buoy on water",
        "small floating object",
    ]

    BG_PROMPTS = [
        "ocean",
        "sea surface",
        "water background",
        "open water",
        "waves",
        "water reflection",
        "sky",
        "horizon",
    ]

    TEXT_GUIDANCE_LAMBDA = 0.5
    FREEZE_CLIP = True
    CLIP_PROJ_DIM = 128
    # =======================

    SUCCESS_REWARD_THRESHOLD = 50.0  # terminal reward >= threshold => success (target hit)

    SAVE_EVERY_ITERS = int(os.environ.get("SAVE_EVERY_ITERS", "50"))
    LOG_EVERY_ITERS = int(os.environ.get("LOG_EVERY_ITERS", "1"))
    MAX_ITERS = int(os.environ.get("HARBOR_MAX_ITERS", "2000"))
    RANDOM_VERIFY_STEPS = int(os.environ.get("HARBOR_RANDOM_VERIFY_STEPS", "0"))
    STEP_DEBUG_PRINT = os.environ.get("HARBOR_STEP_DEBUG_PRINT", "0").lower() in ("1", "true", "yes")
    BEST_WINDOW_EPISODES = int(os.environ.get("HARBOR_BEST_WINDOW_EPISODES", "50"))
    EARLY_STOP_ENABLED = os.environ.get("HARBOR_EARLY_STOP", "1").lower() not in ("0", "false", "no")
    EARLY_STOP_MIN_ITERS = int(os.environ.get("HARBOR_EARLY_STOP_MIN_ITERS", "1000"))
    EARLY_STOP_MIN_EPISODES = int(os.environ.get("HARBOR_EARLY_STOP_MIN_EPISODES", "50"))
    EARLY_STOP_SUCCESS_RATE = float(os.environ.get("HARBOR_EARLY_STOP_SUCCESS_RATE", "0.98"))
    BEST_RATE_EPS = float(os.environ.get("HARBOR_BEST_RATE_EPS", "1e-6"))
    CKPT_DIR = os.environ.get(
        "HARBOR_CKPT_DIR",
        f"checkpoints/usv_harbor_dino_{run_name_for_files}",
    )
    RESUME_CKPT = os.environ.get("HARBOR_RESUME_CKPT", "").strip()
    INIT_MODEL_CKPT = os.environ.get("HARBOR_INIT_MODEL_CKPT", "").strip()
    # ===============================

    # ---- Create env (build strongly recommended) ----
    env = None
    selected_worker_id = WORKER_ID

    # Editor mode requires worker_id == 0.
    if UNITY_FILE is None:
        selected_worker_id = 0
        try:
            env = UnitySingleAgentEnv(
                file_name=UNITY_FILE,
                behavior_name=BEHAVIOR,
                no_graphics=NO_GRAPHICS,
                worker_id=selected_worker_id,
                time_scale=TIME_SCALE,
                base_port=BASE_PORT,
                env_params={
                    "harbor_task_id": float(task_id),
                    "use_oracle_target_observation": 1.0 if USE_ORACLE_TARGET_OBSERVATION else 0.0,
                },
            )
        except UnityWorkerInUseException as e:
            raise RuntimeError(
                "Editor mode detected (UNITY_FILE=None). "
                "Unity Editor only supports worker_id=0. "
                "Port is busy: close previous python training process or restart Play mode, then retry."
            ) from e
    else:
        for _ in range(MAX_WORKER_RETRIES):
            try:
                env = UnitySingleAgentEnv(
                    file_name=UNITY_FILE,
                    behavior_name=BEHAVIOR,
                    no_graphics=NO_GRAPHICS,
                    worker_id=selected_worker_id,
                    time_scale=TIME_SCALE,
                    base_port=BASE_PORT,
                    env_params={
                        "harbor_task_id": float(task_id),
                        "use_oracle_target_observation": 1.0 if USE_ORACLE_TARGET_OBSERVATION else 0.0,
                    },
                )
                break
            except UnityWorkerInUseException:
                print(
                    f"[Port Busy] worker_id={selected_worker_id} in use, "
                    f"retry with worker_id={selected_worker_id + 1}"
                )
                selected_worker_id += 1

    if env is None:
        raise RuntimeError(
            f"Failed to create Unity env after {MAX_WORKER_RETRIES} retries "
            f"starting from worker_id={WORKER_ID}."
        )
    
    # Prime reset to infer shapes
    obs = env.reset()
    img_shape = obs.image.shape
    raw_vec_dim = obs.vector.shape[0]
    action_dim = env.action_size
    localizer = None
    localizer_adapter = None
    vec_dim = raw_vec_dim

    if USE_TARGET_LOCALIZER:
        localizer_cfg = build_harbor_task_config(
            task_name=HARBOR_TASK,
            device=DEVICE,
            dino_visible_score=LOCALIZER_DINO_VISIBLE_SCORE,
            stage2_gate_margin_threshold=LOCALIZER_STAGE2_GATE_MARGIN_THRESHOLD,
            fallback_min_magenta_ratio=LOCALIZER_FALLBACK_MIN_MAGENTA_RATIO,
        )
        localizer = HarborTargetLocalizer(localizer_cfg)
        localizer_adapter = LocalizerPolicyAdapter(
            localizer=localizer,
            refresh_every_steps=LOCALIZER_REFRESH_EVERY_STEPS,
            cache_visible_steps=LOCALIZER_CACHE_VISIBLE_STEPS,
        )
        vec_dim = raw_vec_dim + TARGET_STATE_DIM

    print("[Connected]")
    print("  behavior:", BEHAVIOR)
    print("  worker_id:", selected_worker_id, "base_port:", BASE_PORT)
    print("  image:", img_shape, "raw_vector_dim:", raw_vec_dim, "policy_vector_dim:", vec_dim, "action_dim:", action_dim)
    print("  device:", DEVICE)
    print("  use_target_localizer:", USE_TARGET_LOCALIZER)
    print("  use_oracle_target_observation:", USE_ORACLE_TARGET_OBSERVATION)
    print("  use_clip:", USE_CLIP)
    print("  harbor_task_arg:", HARBOR_TASK)
    print("  harbor_task_id:", task_id)
    print("  harbor_run_variant:", run_variant)
    print("  harbor_task_file_name:", task_name_for_files)
    print("  harbor_run_file_name:", run_name_for_files)
    print("  training_log:", LOG_PATH)
    print("  localizer_log:", LOCALIZER_DEBUG_LOG_PATH)
    print("  clip_debug_log:", CLIP_DEBUG_LOG_PATH)
    print("  checkpoint_dir:", os.path.join(SCRIPT_DIR, CKPT_DIR) if not os.path.isabs(CKPT_DIR) else CKPT_DIR)
    print("  resume_ckpt:", RESUME_CKPT or "(none)")
    print("  init_model_ckpt:", INIT_MODEL_CKPT or "(none)")
    print("  ppo_lr:", LR)
    print("  ppo_horizon:", HORIZON)
    print("  ppo_epochs:", PPO_EPOCHS)
    print("  ppo_minibatch:", MINIBATCH)
    print("  ppo_ent_coef:", PPO_ENT_COEF)
    print("  ppo_target_kl:", PPO_TARGET_KL)
    print("  max_iters:", MAX_ITERS)
    print("  step_debug_print:", STEP_DEBUG_PRINT)
    print("  best_window_episodes:", BEST_WINDOW_EPISODES)
    print("  early_stop_enabled:", EARLY_STOP_ENABLED)
    print("  early_stop_min_iters:", EARLY_STOP_MIN_ITERS)
    print("  early_stop_min_episodes:", EARLY_STOP_MIN_EPISODES)
    print("  early_stop_success_rate:", EARLY_STOP_SUCCESS_RATE)
    if USE_TARGET_LOCALIZER:
        print("  harbor_task:", localizer_cfg.task_name)
        print("  localizer_target_prompts:", list(localizer_cfg.target_prompts))
        print("  localizer_stage2_target_prompts:", list(localizer_cfg.stage2_target_prompts))
        print("  localizer_use_ball_geometry_filter:", localizer_cfg.use_ball_geometry_filter)
        print("  localizer_use_magenta_filter:", localizer_cfg.use_magenta_filter)
        print("  localizer_use_magenta_fallback:", localizer_cfg.use_magenta_fallback)
        print("  target_state_dim:", TARGET_STATE_DIM)
        print("  localizer_refresh_every_steps:", LOCALIZER_REFRESH_EVERY_STEPS)
        print("  localizer_cache_visible_steps:", LOCALIZER_CACHE_VISIBLE_STEPS)
        print("  localizer_dino_visible_score:", LOCALIZER_DINO_VISIBLE_SCORE)
        print("  localizer_stage2_gate_margin_threshold:", LOCALIZER_STAGE2_GATE_MARGIN_THRESHOLD)
        print("  localizer_fallback_min_magenta_ratio:", LOCALIZER_FALLBACK_MIN_MAGENTA_RATIO)
    print("  harbor_random_verify_steps:", RANDOM_VERIFY_STEPS)
    if USE_CLIP:
        print("  clip_model:", CLIP_MODEL_NAME)
        print("  clip_pretrained:", CLIP_PRETRAINED)
        print("  fg_prompts:", FG_PROMPTS)
        print("  bg_prompts:", BG_PROMPTS)
        print("  text_guidance_lambda:", TEXT_GUIDANCE_LAMBDA)
        print("  freeze_clip:", FREEZE_CLIP)
        print("  clip_proj_dim:", CLIP_PROJ_DIM)
    print("  unity_summary:", UNITY_SUMMARY_PATH)

    if RANDOM_VERIFY_STEPS > 0:
        print(f"[Random Verify] running {RANDOM_VERIFY_STEPS} random-action steps without PPO update")
        rng = np.random.default_rng(123)
        verify_visible = 0
        verify_refreshed = 0
        for verify_step in range(RANDOM_VERIFY_STEPS):
            vec, localizer_result, localizer_refreshed = build_policy_vector(
                obs,
                localizer_adapter=localizer_adapter,
            )
            action = np.array(
                [
                    rng.uniform(-0.6, 0.6),
                    rng.uniform(0.5, 1.5),
                ],
                dtype=np.float32,
            )
            next_obs, reward, done, info = env.step(action)

            if localizer_result is not None:
                verify_visible += int(localizer_result.visible)
                verify_refreshed += int(localizer_refreshed)
                append_localizer_debug_log(
                    LOCALIZER_DEBUG_LOG_PATH,
                    global_step=verify_step,
                    it=0,
                    reward=float(reward),
                    raw_vector_dim=int(obs.vector.shape[0]),
                    aug_vector_dim=int(vec.shape[0]),
                    result=localizer_result,
                )
                print(
                    "[Random Verify]",
                    {
                        "step": verify_step,
                        "reward": round(float(reward), 3),
                        "refreshed": localizer_refreshed,
                        "visible": localizer_result.visible,
                        "source": getattr(localizer_result, "visible_source", ""),
                        "cache_age": getattr(localizer_result, "cache_age_steps", 0),
                        "direction": localizer_result.direction,
                        "state": np.round(localizer_result.state, 3).tolist(),
                    },
                )
            obs = env.reset() if done else next_obs

        print(
            "[Random Verify Summary]",
            {
                "steps": RANDOM_VERIFY_STEPS,
                "refreshed": verify_refreshed,
                "visible": verify_visible,
                "visible_ratio": verify_visible / max(RANDOM_VERIFY_STEPS, 1),
                "log": LOCALIZER_DEBUG_LOG_PATH,
            },
        )
        env.close()
        print("[Closed] Unity env closed cleanly.")
        return

    model = ActorCritic(
        action_dim=action_dim,
        vec_dim=vec_dim,
        use_clip=USE_CLIP,
        clip_model=CLIP_MODEL_NAME,
        clip_pretrained=CLIP_PRETRAINED,
        freeze_clip=FREEZE_CLIP,
        proj_dim=CLIP_PROJ_DIM,
        trunk_dim=256,
        device=DEVICE,
    )

    ppo = PPO(
        model=model,
        lr=LR,
        epochs=PPO_EPOCHS,
        minibatch_size=MINIBATCH,
        ent_coef=PPO_ENT_COEF,
        target_kl=PPO_TARGET_KL,
        device=DEVICE,
    )

    resume_meta = {}
    if RESUME_CKPT and INIT_MODEL_CKPT:
        raise ValueError("Use only one of HARBOR_RESUME_CKPT or HARBOR_INIT_MODEL_CKPT.")

    if RESUME_CKPT:
        resume_path = resolve_input_path(RESUME_CKPT)
        resume_meta = load_ckpt(resume_path, model, ppo, DEVICE)
        print(
            "[Resume]",
            {
                "path": resume_path,
                "iter": resume_meta.get("iter", 0),
                "global_step": resume_meta.get("global_step", 0),
            },
        )
    elif INIT_MODEL_CKPT:
        init_path = resolve_input_path(INIT_MODEL_CKPT)
        init_meta = load_ckpt(init_path, model, ppo, DEVICE, load_optimizer=False)
        print(
            "[Init Model]",
            {
                "path": init_path,
                "source_iter": init_meta.get("iter", 0),
                "source_global_step": init_meta.get("global_step", 0),
            },
        )

    buffer = RolloutBuffer(
        horizon=HORIZON,
        img_shape=img_shape,
        vec_dim=vec_dim,
        action_dim=action_dim,
        device=DEVICE,
    )

    # ---- Training loop ----
    it = int(resume_meta.get("iter", 0) or 0)
    global_step = int(resume_meta.get("global_step", 0) or 0)
    ep_return, ep_len = 0.0, 0
    last_unity_row_count, last_unity_row = get_unity_summary_baseline(UNITY_SUMMARY_PATH)
    start_time = time.time()
    episodes_logged = 0
    successes_logged = 0
    recent_successes = deque(maxlen=max(1, BEST_WINDOW_EPISODES))
    best_recent_success_rate = -1.0
    best_ckpt_path = os.path.join(CKPT_DIR, "ckpt_best.pt")
    localizer_seen_count = 0
    localizer_visible_count = 0
    localizer_pre_visible_count = 0
    localizer_gate_checked_count = 0
    localizer_gate_pass_count = 0

    print("  unity_summary_baseline_rows:", last_unity_row_count)
    if last_unity_row is not None:
        print(
            "  unity_summary_last_seen:",
            {
                "episode": last_unity_row["episode_idx"],
                "reward": round(float(last_unity_row["cumulative_reward"]), 3),
                "steps": int(last_unity_row["steps"]),
                "success": int(last_unity_row["unity_success"]),
                "reason": last_unity_row["reason"],
            },
        )

    try:
        while True:
            it += 1
            buffer.t = 0

            # Collect rollout
            for _ in range(HORIZON):
                img_chw = hwc_to_chw(obs.image)
                vec, localizer_result, localizer_refreshed = build_policy_vector(
                    obs,
                    localizer_adapter=localizer_adapter,
                )

                img_t = torch.from_numpy(img_chw).unsqueeze(0)
                vec_t = torch.from_numpy(vec).unsqueeze(0)

                with torch.no_grad():
                    action_t, logp_t, v_t = model.act(img_t, vec_t)

                action = action_t.squeeze(0).cpu().numpy()

                logp = logp_t.squeeze(0).cpu().numpy()
                v = v_t.squeeze(0).cpu().numpy()

                next_obs, reward, done, info = env.step(action)

                # ===== DEBUG（正确版本）=====
                vec_debug = next_obs.vector
                rel = vec_debug[:2]
                dist = np.linalg.norm(rel)

                # print(f"[Step] reward={reward:.4f}")
                # print(f"[Step] action={action}")
                # print(f"[Step] vec_head={vec_debug[:8]}")
                # print(
                #     f"[Debug] dist={dist:.3f}, heading_err={vec_debug[2]:.3f}, "
                #     f"fwd_speed={vec_debug[3]:.3f}, yaw_rate={vec_debug[4]:.3f}"
                # )
                # if hasattr(model, "last_clip_debug"):
                #     print("[CLIP DEBUG]", model.last_clip_debug)

                if STEP_DEBUG_PRINT:
                    print(f"[Step] reward={reward:.4f}")
                    print(f"[Step] action={action}")
                    print(f"[Step] vec_head={vec_debug[:8]}")
                    print(
                        f"[Debug] dist={dist:.3f}, heading_err={vec_debug[2]:.3f}, "
                        f"fwd_speed={vec_debug[3]:.3f}, yaw_rate={vec_debug[4]:.3f}"
                    )

                if localizer_result is not None:
                    localizer_seen_count += 1
                    if localizer_result.visible:
                        localizer_visible_count += 1
                    if localizer_result.dino_visible_before_stage2:
                        localizer_pre_visible_count += 1
                    if localizer_result.stage2_checked:
                        localizer_gate_checked_count += 1
                    if localizer_result.stage2_pass:
                        localizer_gate_pass_count += 1

                    if global_step % LOCALIZER_LOG_EVERY_STEPS == 0:
                        print(
                            "[LOCALIZER]",
                            {
                                "visible": localizer_result.visible,
                                "source": getattr(localizer_result, "visible_source", ""),
                                "cache_age": getattr(localizer_result, "cache_age_steps", 0),
                                "pre_visible": localizer_result.dino_visible_before_stage2,
                                "refreshed": localizer_refreshed,
                                "direction": localizer_result.direction,
                                "state": np.round(localizer_result.state, 3).tolist(),
                                "dino_score": round(float(localizer_result.dino_score), 3),
                                "dino_magenta": round(float(getattr(localizer_result, "dino_magenta_ratio", 0.0)), 3),
                                "fallback": getattr(localizer_result, "fallback_visible", False),
                                "stage2_margin": round(float(localizer_result.stage2_margin), 3),
                                "raw_vec_dim": int(obs.vector.shape[0]),
                                "aug_vec_dim": int(vec.shape[0]),
                            },
                        )
                        append_localizer_debug_log(
                            LOCALIZER_DEBUG_LOG_PATH,
                            global_step=global_step,
                            it=it,
                            reward=float(reward),
                            raw_vector_dim=int(obs.vector.shape[0]),
                            aug_vector_dim=int(vec.shape[0]),
                            result=localizer_result,
                        )

                # ===== CLIP DEBUG（打印 + 保存）=====
                if hasattr(model, "last_clip_debug") and global_step % 10 == 0:
                    print("[CLIP DEBUG]", model.last_clip_debug)

                    append_clip_debug_log(
                        CLIP_DEBUG_LOG_PATH,
                        global_step=global_step,
                        it=it,
                        reward=float(reward),
                        dist=float(dist),
                        heading_err=float(vec_debug[2]),
                        fwd_speed=float(vec_debug[3]),
                        yaw_rate=float(vec_debug[4]),
                        clip_debug=model.last_clip_debug,
                    )

                buffer.add(
                    img_chw=img_chw,
                    vec=vec,
                    action=action,
                    logp=logp,
                    reward=np.array([reward], dtype=np.float32),
                    done=done,
                    value=v,
                )

                ep_return += reward
                ep_len += 1
                global_step += 1
                obs = next_obs

                # if done:
                #     print(f"[Episode] return={ep_return:.3f} len={ep_len}")
                #     obs = env.reset()
                #     ep_return, ep_len = 0.0, 0
                if done:
                    # ===== success 判断 =====
                    # Prefer Unity's own summary (ground truth), fallback to terminal reward threshold.
                    unity_row, new_row_count = poll_unity_episode_result(
                        UNITY_SUMMARY_PATH,
                        last_row_count=last_unity_row_count,
                        timeout_s=5.0,
                    )
                    last_unity_row_count = new_row_count

                    if unity_row is not None:
                        unity_ep = int(unity_row["episode_idx"])
                        unity_success = int(unity_row["unity_success"])
                        termination_reason = unity_row["reason"]
                        unity_reward = float(unity_row["cumulative_reward"])
                        unity_steps = int(unity_row["steps"])
                        success = unity_success if unity_success in (0, 1) else 0
                        episodes_logged += 1
                        successes_logged += success
                        recent_successes.append(success)
                        success_rate = successes_logged / max(episodes_logged, 1)
                        recent_success_rate = (
                            sum(recent_successes) / len(recent_successes)
                            if recent_successes
                            else 0.0
                        )

                        reward_delta = abs(float(ep_return) - unity_reward)
                        if reward_delta > 1.0:
                            print(
                                "[Summary Sync Warning]",
                                {
                                    "python_episode_reward": round(float(ep_return), 3),
                                    "unity_episode_reward": round(unity_reward, 3),
                                    "delta": round(reward_delta, 3),
                                    "python_ep_len": int(ep_len),
                                    "unity_steps": unity_steps,
                                    "unity_ep": unity_ep,
                                },
                            )

                        print(
                            f"[Episode] return={ep_return:.3f} len={ep_len} "
                            f"terminal_reward={reward:.3f} success={success} "
                            f"unity_ep={unity_ep} unity_success={unity_success} "
                            f"success_rate={success_rate:.3f} "
                            f"recent_success_rate={recent_success_rate:.3f} "
                            f"reason={termination_reason}"
                        )

                        # ===== 写 CSV（仅写入 Unity 已确认的 episode）=====
                        append_training_log(
                            LOG_PATH,
                            it,
                            global_step,
                            episodes_logged,
                            ep_return,
                            success,
                            success_rate,
                            reward,
                            unity_ep,
                            unity_success,
                            termination_reason,
                        )

                        if (
                            len(recent_successes) == recent_successes.maxlen
                            and recent_success_rate > best_recent_success_rate + BEST_RATE_EPS
                        ):
                            best_recent_success_rate = recent_success_rate
                            best_meta = {
                                "iter": it,
                                "global_step": global_step,
                                "time": time.time(),
                                "episodes_logged": episodes_logged,
                                "success_rate": success_rate,
                                "recent_success_rate": recent_success_rate,
                                "recent_window_episodes": len(recent_successes),
                                "reason": "best_recent_success_rate",
                            }
                            save_ckpt(best_ckpt_path, model, ppo, best_meta)
                            print(
                                f"[Best] saved {best_ckpt_path} "
                                f"recent_success_rate={recent_success_rate:.3f} "
                                f"episodes={episodes_logged} iter={it}"
                            )

                        if (
                            EARLY_STOP_ENABLED
                            and it >= EARLY_STOP_MIN_ITERS
                            and episodes_logged >= EARLY_STOP_MIN_EPISODES
                            and len(recent_successes) == recent_successes.maxlen
                            and recent_success_rate >= EARLY_STOP_SUCCESS_RATE
                        ):
                            print(
                                f"[Early Stop] recent_success_rate={recent_success_rate:.3f} "
                                f">= {EARLY_STOP_SUCCESS_RATE:.3f}, "
                                f"episodes={episodes_logged}, iter={it}"
                            )
                            raise StopIteration
                    else:
                        print(
                            f"[Episode] return={ep_return:.3f} len={ep_len} terminal_reward={reward:.3f} "
                            f"-> skipped logging: Unity summary row not available within timeout"
                        )

                    obs = env.reset()
                    if localizer_adapter is not None:
                        localizer_adapter.reset()
                    ep_return, ep_len = 0.0, 0

            # Bootstrap value
            img_chw = hwc_to_chw(obs.image)
            vec, _, _ = build_policy_vector(obs, localizer_adapter=localizer_adapter)
            img_t = torch.from_numpy(img_chw).unsqueeze(0)
            vec_t = torch.from_numpy(vec).unsqueeze(0)
            with torch.no_grad():
                _, v_last = model.forward(img_t, vec_t)
            last_value = float(v_last.squeeze(0).cpu().numpy())

            buffer.compute_gae(last_value=last_value, gamma=GAMMA, lam=LAM)
            img_b, vec_b, act_b, logp_b, ret_b, adv_b, _ = buffer.get_tensors()

            stats = ppo.update(img_b, vec_b, act_b, logp_b, ret_b, adv_b)

            if it % LOG_EVERY_ITERS == 0:
                fps = int(global_step / max(1e-6, (time.time() - start_time)))
                print(
                    f"[Iter {it:06d}] steps={global_step} fps={fps} "
                    f"actor_loss={stats['actor_loss']:.4f} "
                    f"critic_loss={stats['critic_loss']:.4f} "
                    f"entropy={stats['entropy']:.4f} "
                    f"approx_kl={stats.get('approx_kl', 0.0):.5f} "
                    f"epochs_used={stats.get('ppo_epochs_used', PPO_EPOCHS)} "
                    f"kl_stop={stats.get('kl_early_stop', False)}"
                )
                if localizer_seen_count > 0:
                    print(
                        f"[Iter {it:06d}] localizer "
                        f"seen={localizer_seen_count} "
                        f"visible={localizer_visible_count} "
                        f"pre_visible={localizer_pre_visible_count} "
                        f"gate_checked={localizer_gate_checked_count} "
                        f"gate_pass={localizer_gate_pass_count} "
                        f"visible_ratio={localizer_visible_count / localizer_seen_count:.3f}"
                    )

            # # ===== EVAL =====
            # if it % 20 == 0:
            #     eval_reward, success = evaluate(env, model, localizer=localizer)

            #     print(f"[EVAL] iter={it}, reward={eval_reward:.2f}, success={success}")

            #     with open(log_path, "a", newline="") as f:
            #         writer = csv.writer(f)
            #         writer.writerow([it, eval_reward, success])  # 记录 success

            if it % SAVE_EVERY_ITERS == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"ckpt_iter_{it:06d}.pt")
                meta = {"iter": it, "global_step": global_step, "time": time.time()}
                save_ckpt(ckpt_path, model, ppo, meta)
                print(f"[Saved] {ckpt_path}")

            if MAX_ITERS > 0 and it >= MAX_ITERS:
                print(f"[Done] reached HARBOR_MAX_ITERS={MAX_ITERS}")
                break

    except StopIteration:
        print("[Stopped] Early stopping condition reached.")

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving last checkpoint...")
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_last.pt")
        meta = {"iter": it, "global_step": global_step, "time": time.time()}
        save_ckpt(ckpt_path, model, ppo, meta)
        print(f"[Saved] {ckpt_path}")

    finally:
        env.close()
        print("[Closed] Unity env closed cleanly.")


if __name__ == "__main__":
    main()
