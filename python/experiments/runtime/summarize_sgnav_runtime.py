import argparse
import csv
import json
import os

import numpy as np


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(rows):
    preferred = [
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
    components = [name for name in preferred if f"{name}_ms" in rows[0]]
    out = []
    for task in sorted(set(row["task"] for row in rows)):
        task_rows = [row for row in rows if row["task"] == task and int(row["is_warmup"]) == 0]
        for component in components:
            values = [float(row[f"{component}_ms"]) for row in task_rows]
            out.append(
                {
                    "task": task,
                    "component": component,
                    "frames": len(values),
                    "mean_ms": float(np.mean(values)) if values else float("nan"),
                    "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median_ms": percentile(values, 50),
                    "p90_ms": percentile(values, 90),
                    "p95_ms": percentile(values, 95),
                }
            )

    pooled_rows = [row for row in rows if int(row["is_warmup"]) == 0]
    for component in components:
        values = [float(row[f"{component}_ms"]) for row in pooled_rows]
        out.append(
            {
                "task": "combined",
                "component": component,
                "frames": len(values),
                "mean_ms": float(np.mean(values)) if values else float("nan"),
                "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median_ms": percentile(values, 50),
                "p90_ms": percentile(values, 90),
                "p95_ms": percentile(values, 95),
            }
        )
    return out


def summarize_refreshed(rows):
    preferred = ["dino", "clip", "filter_state", "localizer", "policy", "total"]
    components = [name for name in preferred if f"{name}_ms" in rows[0]]
    out = []
    for task in sorted(set(row["task"] for row in rows)):
        task_rows = [
            row
            for row in rows
            if row["task"] == task and int(row["is_warmup"]) == 0 and int(row["localizer_refreshed"]) == 1
        ]
        for component in components:
            values = [float(row[f"{component}_ms"]) for row in task_rows]
            out.append(
                {
                    "task": task,
                    "component": component,
                    "frames": len(values),
                    "mean_ms": float(np.mean(values)) if values else float("nan"),
                    "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median_ms": percentile(values, 50),
                    "p90_ms": percentile(values, 90),
                    "p95_ms": percentile(values, 95),
                }
            )

    pooled_rows = [
        row
        for row in rows
        if int(row["is_warmup"]) == 0 and int(row["localizer_refreshed"]) == 1
    ]
    for component in components:
        values = [float(row[f"{component}_ms"]) for row in pooled_rows]
        out.append(
            {
                "task": "combined",
                "component": component,
                "frames": len(values),
                "mean_ms": float(np.mean(values)) if values else float("nan"),
                "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median_ms": percentile(values, 50),
                "p90_ms": percentile(values, 90),
                "p95_ms": percentile(values, 95),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Combine SGNav runtime frame CSVs into one summary table.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", default="results/runtime/sgnav_runtime_combined")
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(read_rows(path))

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = summarize(rows)

    csv_path = os.path.join(args.output_dir, "runtime_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = os.path.join(args.output_dir, "runtime_summary.json")
    refresh_rows = summarize_refreshed(rows)
    refresh_csv_path = os.path.join(args.output_dir, "runtime_refresh_summary.csv")
    with open(refresh_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(refresh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(refresh_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "refresh_summary": refresh_rows}, f, indent=2, allow_nan=True)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {refresh_csv_path}")
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
