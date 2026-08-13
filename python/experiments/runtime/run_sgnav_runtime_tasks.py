import argparse
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PROFILE_SCRIPT = os.path.join(SCRIPT_DIR, "profile_sgnav_runtime.py")
SUMMARY_SCRIPT = os.path.join(SCRIPT_DIR, "summarize_sgnav_runtime.py")


def run(cmd):
    print("[Run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run SGNav runtime profiling for Task1 and Task3, then combine outputs.")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--layout", default=os.environ.get("HARBOR_LAYOUT", "layout_a"))
    parser.add_argument("--distractor", default=os.environ.get("HARBOR_DISTRACTOR", "none"))
    parser.add_argument("--unity-file", default=os.environ.get("UNITY_FILE", ""))
    parser.add_argument("--behavior", default=os.environ.get("UNITY_BEHAVIOR_NAME", "USV?team=0"))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("UNITY_BASE_PORT", "5004")))
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("UNITY_WORKER_ID", "0")))
    parser.add_argument("--time-scale", type=float, default=float(os.environ.get("UNITY_TIME_SCALE", "20.0")))
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--task1-checkpoint", default="")
    parser.add_argument("--task3-checkpoint", default="")
    args = parser.parse_args()

    common = [
        "--frames",
        str(args.frames),
        "--warmup-frames",
        str(args.warmup_frames),
        "--layout",
        args.layout,
        "--distractor",
        args.distractor,
        "--behavior",
        args.behavior,
        "--base-port",
        str(args.base_port),
        "--worker-id",
        str(args.worker_id),
        "--time-scale",
        str(args.time_scale),
    ]
    if args.unity_file.strip():
        common.extend(["--unity-file", args.unity_file])
    if args.no_graphics:
        common.append("--no-graphics")

    task1_out = os.path.join("results", "runtime", "sgnav_runtime_task1")
    task3_out = os.path.join("results", "runtime", "sgnav_runtime_task3")

    task1_cmd = [sys.executable, PROFILE_SCRIPT, "--task", "task1", "--output-dir", task1_out] + common
    if args.task1_checkpoint.strip():
        task1_cmd.extend(["--checkpoint", args.task1_checkpoint])
    run(task1_cmd)

    task3_cmd = [sys.executable, PROFILE_SCRIPT, "--task", "task3", "--output-dir", task3_out] + common
    if args.task3_checkpoint.strip():
        task3_cmd.extend(["--checkpoint", args.task3_checkpoint])
    run(task3_cmd)

    run(
        [
            sys.executable,
            SUMMARY_SCRIPT,
            os.path.join(task1_out, "runtime_frames.csv"),
            os.path.join(task3_out, "runtime_frames.csv"),
            "--output-dir",
            os.path.join("results", "runtime", "sgnav_runtime_combined"),
        ]
    )


if __name__ == "__main__":
    main()
