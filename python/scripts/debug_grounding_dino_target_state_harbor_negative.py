import os
import sys
import csv
import math
import random
import socket
import matplotlib.patches as patches

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from envs.unity_ma_env import UnitySingleAgentEnv
from mlagents_envs.exception import UnityWorkerInUseException

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except Exception:
    AutoProcessor = None
    AutoModelForZeroShotObjectDetection = None

try:
    from models.clip_encoder_single import ClipVisionEncoder
except Exception:
    ClipVisionEncoder = None


def to_uint8_image(img):
    img_vis = img.copy()
    if img_vis.max() <= 1.0:
        img_vis = (img_vis * 255).clip(0, 255).astype(np.uint8)
    else:
        img_vis = img_vis.clip(0, 255).astype(np.uint8)
    return img_vis


def weights_to_map(weights):
    num_patches = weights.shape[1]
    grid_size = int(num_patches ** 0.5)
    return weights[0].reshape(grid_size, grid_size).detach().cpu().numpy()


def extract_target_state_from_weights(weights, img_h=84, img_w=84):
    num_patches = weights.shape[1]
    grid_size = int(num_patches ** 0.5)

    flat = weights[0].detach().cpu().numpy()
    sorted_idx = np.argsort(-flat)

    top1_idx = int(sorted_idx[0])
    top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else int(sorted_idx[0])

    top1_score = float(flat[top1_idx])
    top2_score = float(flat[top2_idx])
    score_margin = top1_score - top2_score

    py = top1_idx // grid_size
    px = top1_idx % grid_size

    cell_w = img_w / grid_size
    cell_h = img_h / grid_size

    best_x = (px + 0.5) * cell_w
    best_y = (py + 0.5) * cell_h

    dir_left = 0.0
    dir_center = 0.0
    dir_right = 0.0

    for idx, score in enumerate(flat):
        px_i = idx % grid_size
        x_center = (px_i + 0.5) * cell_w

        if x_center < img_w / 3:
            dir_left += float(score)
        elif x_center < 2 * img_w / 3:
            dir_center += float(score)
        else:
            dir_right += float(score)

    return {
        "visible": float(top1_score),
        "best_score": float(top1_score),
        "best_x": float(best_x),
        "best_y": float(best_y),
        "dir_left": float(dir_left),
        "dir_center": float(dir_center),
        "dir_right": float(dir_right),
        "score_margin": float(score_margin),
        "top1_idx": int(top1_idx),
        "top2_idx": int(top2_idx),
    }


def roi_mean_from_patch_map(w_map, x1, y1, x2, y2, img_h=84, img_w=84):
    gh, gw = w_map.shape

    px1 = int(x1 / img_w * gw)
    px2 = int(x2 / img_w * gw)
    py1 = int(y1 / img_h * gh)
    py2 = int(y2 / img_h * gh)

    px1 = max(0, min(gw - 1, px1))
    px2 = max(px1 + 1, min(gw, px2))
    py1 = max(0, min(gh - 1, py1))
    py2 = max(py1 + 1, min(gh, py2))

    roi = w_map[py1:py2, px1:px2]
    return float(roi.mean())


def target_state_hit_ball_roi(target_state, ball_roi):
    if ball_roi is None:
        return False
    x = target_state["best_x"]
    y = target_state["best_y"]
    x1, y1, x2, y2 = ball_roi
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def gt_direction_from_ball_roi(ball_roi, img_w=84):
    if ball_roi is None:
        return None

    x1, y1, x2, y2 = ball_roi
    cx = 0.5 * (x1 + x2)

    if cx < img_w / 3:
        return "left"
    elif cx < 2 * img_w / 3:
        return "center"
    else:
        return "right"


def predicted_direction_from_state(target_state):
    d = {
        "left": target_state["dir_left"],
        "center": target_state["dir_center"],
        "right": target_state["dir_right"],
    }
    return max(d, key=d.get)


def target_state_from_box(box, img_h, img_w, score=1.0, score_margin=0.0):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    dir_left = 0.0
    dir_center = 0.0
    dir_right = 0.0
    if cx < img_w / 3:
        dir_left = 1.0
    elif cx < 2 * img_w / 3:
        dir_center = 1.0
    else:
        dir_right = 1.0

    return {
        "visible": float(score),
        "best_score": float(score),
        "best_x": float(cx),
        "best_y": float(cy),
        "dir_left": float(dir_left),
        "dir_center": float(dir_center),
        "dir_right": float(dir_right),
        "score_margin": float(score_margin),
        "top1_idx": -1,
        "top2_idx": -1,
    }


def _binary_dilate(mask, iterations=1):
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        new_mask = np.zeros_like(out, dtype=np.uint8)

        for dy in range(3):
            for dx in range(3):
                new_mask = np.maximum(new_mask, padded[dy:dy + out.shape[0], dx:dx + out.shape[1]])

        out = new_mask
    return out


def _binary_erode(mask, iterations=1):
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        new_mask = np.ones_like(out, dtype=np.uint8)

        for dy in range(3):
            for dx in range(3):
                new_mask = np.minimum(new_mask, padded[dy:dy + out.shape[0], dx:dx + out.shape[1]])

        out = new_mask
    return out


def _binary_close(mask, iterations=1):
    return _binary_erode(_binary_dilate(mask, iterations), iterations)


def _connected_components(mask):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    def bfs(sy, sx):
        q = [(sy, sx)]
        visited[sy, sx] = True
        pixels = []

        while q:
            y, x = q.pop()
            pixels.append((y, x))
            for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        return pixels

    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                comp = bfs(y, x)
                components.append(comp)

    return components


def _crop_expand_roi(roi, img_h, img_w, pad=2):
    x1, y1, x2, y2 = roi
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w - 1, x2 + pad)
    y2 = min(img_h - 1, y2 + pad)
    return (x1, y1, x2, y2)


def _roi_center(roi):
    x1, y1, x2, y2 = roi
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _roi_distance(roi_a, roi_b):
    if roi_a is None or roi_b is None:
        return 1e9
    ax, ay = _roi_center(roi_a)
    bx, by = _roi_center(roi_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def find_ball_roi(img, hint_xy=None, prev_roi=None):
    img_u8 = to_uint8_image(img)

    r = img_u8[..., 0].astype(np.int16)
    g = img_u8[..., 1].astype(np.int16)
    b = img_u8[..., 2].astype(np.int16)
    h, w = r.shape

    # Harbor target is magenta/purple. The old red-only mask can lock onto
    # red buoys or port lights, which makes the debug "ground truth" wrong.
    mask = (
        (r > 90) &
        (b > 90) &
        (r > g + 22) &
        (b > g + 18)
    ).astype(np.uint8)

    mask = _binary_close(mask, iterations=1)
    mask = _binary_dilate(mask, iterations=1)
    components = _connected_components(mask)

    component_count = len(components)
    candidates = []

    for comp in components:
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        area = len(comp)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        if area < 3 or area > 2000:
            continue
        if bw < 2 or bh < 2:
            continue
        if bw > 60 or bh > 60:
            continue

        aspect = bw / (bh + 1e-6)
        squareness = abs(aspect - 1.0)
        fill_ratio = area / float(bw * bh + 1e-6)

        mean_r = float(np.mean([r[y, x] for y, x in comp]))
        mean_g = float(np.mean([g[y, x] for y, x in comp]))
        mean_b = float(np.mean([b[y, x] for y, x in comp]))
        magenta_strength = 0.5 * (mean_r + mean_b) - mean_g

        dist_hint = 0.0
        if hint_xy is not None:
            hx, hy = hint_xy
            dist_hint = ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5

        dist_prev = 0.0
        if prev_roi is not None:
            dist_prev = _roi_distance((x1, y1, x2, y2), prev_roi)

        edge_penalty = 0.0
        if x1 <= 1 or x2 >= w - 2:
            edge_penalty += 8.0
        if y1 <= 1 or y2 >= h - 2:
            edge_penalty += 8.0

        score = (
            magenta_strength * 0.12
            + fill_ratio * 10.0
            - squareness * 1.0
            - dist_hint * 0.10
            - dist_prev * 0.03
            - edge_penalty
        )

        candidates.append({
            "score": score,
            "roi": (x1, y1, x2, y2),
            "dist_prev": dist_prev,
        })

    if prev_roi is not None and len(candidates) > 0:
        filtered = [cand for cand in candidates if cand["dist_prev"] <= 35]
        if len(filtered) > 0:
            candidates = filtered

    candidate_count = len(candidates)

    if candidate_count == 0:
        if prev_roi is not None:
            return prev_roi, True, 0, component_count
        return None, False, 0, component_count

    candidates.sort(key=lambda d: d["score"], reverse=True)
    best = candidates[0]
    roi = _crop_expand_roi(best["roi"], h, w, pad=3)
    return roi, False, candidate_count, component_count


def box_magenta_ratio(image_hwc_float, box):
    img_u8 = to_uint8_image(image_hwc_float)
    img_h, img_w = img_u8.shape[:2]
    x1, y1, x2, y2 = box
    x1_i = int(max(0, min(img_w - 1, math.floor(x1))))
    y1_i = int(max(0, min(img_h - 1, math.floor(y1))))
    x2_i = int(max(x1_i + 1, min(img_w, math.ceil(x2))))
    y2_i = int(max(y1_i + 1, min(img_h, math.ceil(y2))))
    crop = img_u8[y1_i:y2_i, x1_i:x2_i]
    if crop.size == 0:
        return 0.0

    r = crop[..., 0].astype(np.int16)
    g = crop[..., 1].astype(np.int16)
    b = crop[..., 2].astype(np.int16)
    mask = (
        (r > 90) &
        (b > 90) &
        (r > g + 22) &
        (b > g + 18)
    )
    return float(mask.mean())


def find_green_roi(img, prev_roi=None):
    img_u8 = to_uint8_image(img)
    r = img_u8[..., 0].astype(np.int16)
    g = img_u8[..., 1].astype(np.int16)
    b = img_u8[..., 2].astype(np.int16)
    h, w = g.shape

    mask = (
        (g > 70) &
        (g > r + 22) &
        (g > b + 18)
    ).astype(np.uint8)

    mask = _binary_close(mask, iterations=1)
    mask = _binary_dilate(mask, iterations=1)
    components = _connected_components(mask)

    component_count = len(components)
    candidates = []

    for comp in components:
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        area = len(comp)

        if area < 4 or area > 2600:
            continue
        if bw < 2 or bh < 3:
            continue
        if bw > 80 or bh > 100:
            continue

        fill_ratio = area / float(bw * bh + 1e-6)
        tall_bonus = min(float(bh) / max(float(bw), 1.0), 5.0)
        mean_g = float(np.mean([g[y, x] for y, x in comp]))
        mean_r = float(np.mean([r[y, x] for y, x in comp]))
        mean_b = float(np.mean([b[y, x] for y, x in comp]))
        green_strength = mean_g - 0.5 * (mean_r + mean_b)

        dist_prev = 0.0
        if prev_roi is not None:
            dist_prev = _roi_distance((x1, y1, x2, y2), prev_roi)

        edge_bonus = 0.0
        if x1 <= 2 or x2 >= w - 3:
            edge_bonus += 2.0

        score = green_strength * 0.10 + fill_ratio * 6.0 + tall_bonus * 0.6 + edge_bonus - dist_prev * 0.02
        candidates.append({
            "score": score,
            "roi": (x1, y1, x2, y2),
            "dist_prev": dist_prev,
        })

    if prev_roi is not None and len(candidates) > 0:
        filtered = [cand for cand in candidates if cand["dist_prev"] <= 45]
        if len(filtered) > 0:
            candidates = filtered

    if len(candidates) == 0:
        if prev_roi is not None:
            return prev_roi, True, 0, component_count
        return None, False, 0, component_count

    candidates.sort(key=lambda d: d["score"], reverse=True)
    roi = _crop_expand_roi(candidates[0]["roi"], h, w, pad=5)
    return roi, False, len(candidates), component_count


def box_green_ratio(image_hwc_float, box):
    img_u8 = to_uint8_image(image_hwc_float)
    img_h, img_w = img_u8.shape[:2]
    x1, y1, x2, y2 = box
    x1_i = int(max(0, min(img_w - 1, math.floor(x1))))
    y1_i = int(max(0, min(img_h - 1, math.floor(y1))))
    x2_i = int(max(x1_i + 1, min(img_w, math.ceil(x2))))
    y2_i = int(max(y1_i + 1, min(img_h, math.ceil(y2))))
    crop = img_u8[y1_i:y2_i, x1_i:x2_i]
    if crop.size == 0:
        return 0.0

    r = crop[..., 0].astype(np.int16)
    g = crop[..., 1].astype(np.int16)
    b = crop[..., 2].astype(np.int16)
    mask = (
        (g > 70) &
        (g > r + 22) &
        (g > b + 18)
    )
    return float(mask.mean())


def box_yellow_ratio(image_hwc_float, box):
    img_u8 = to_uint8_image(image_hwc_float)
    img_h, img_w = img_u8.shape[:2]
    x1, y1, x2, y2 = box
    x1_i = int(max(0, min(img_w - 1, math.floor(x1))))
    y1_i = int(max(0, min(img_h - 1, math.floor(y1))))
    x2_i = int(max(x1_i + 1, min(img_w, math.ceil(x2))))
    y2_i = int(max(y1_i + 1, min(img_h, math.ceil(y2))))
    crop = img_u8[y1_i:y2_i, x1_i:x2_i]
    if crop.size == 0:
        return 0.0

    r = crop[..., 0].astype(np.int16)
    g = crop[..., 1].astype(np.int16)
    b = crop[..., 2].astype(np.int16)
    mask = (
        (r > 95) &
        (g > 55) &
        (b < 130) &
        (r > b + 35) &
        (g > b + 12)
    )
    return float(mask.mean())


def filter_detections_by_magenta_ratio(detections, image_hwc_float, min_ratio=0.02):
    filtered = []
    for det in detections:
        ratio = box_magenta_ratio(image_hwc_float, det["box"])
        if ratio < min_ratio:
            continue
        det = dict(det)
        det["magenta_ratio"] = ratio
        filtered.append(det)
    return filtered


def filter_detections_by_yellow_ratio(detections, image_hwc_float, min_ratio=0.02):
    filtered = []
    for det in detections:
        ratio = box_yellow_ratio(image_hwc_float, det["box"])
        if ratio < min_ratio:
            continue
        det = dict(det)
        det["yellow_ratio"] = ratio
        filtered.append(det)
    return filtered


def _is_ship_distractor_label(label):
    label = (label or "").lower()
    keywords = (
        "ship", "boat", "vessel", "freighter", "cargo",
        "large vessel", "large cargo", "moored boat",
    )
    return any(keyword in label for keyword in keywords)


def _is_tugboat_target_label(label):
    label = (label or "").lower()
    keywords = ("tug", "tugboat", "towboat", "yellow", "vessel03")
    return any(keyword in label for keyword in keywords)


def filter_detections_by_ship_distractor_conflict(
    target_detections,
    distractor_detections,
    img_h,
    img_w,
    max_ship_iou=0.45,
    max_ship_near=0.92,
    min_ship_score=0.25,
    yellow_override_ratio=0.08,
):
    filtered = []
    for det in target_detections:
        if float(det.get("yellow_ratio", 0.0)) >= yellow_override_ratio:
            filtered.append(det)
            continue

        reject = False
        reject_label = None
        reject_iou = 0.0
        reject_near = 0.0
        for dis in distractor_detections:
            if float(dis.get("score", 0.0)) < min_ship_score:
                continue
            if not _is_ship_distractor_label(dis.get("label")):
                continue
            iou = box_iou(det["box"], dis["box"])
            dist = box_center_distance_norm(det["box"], dis["box"], img_h, img_w)
            near = max(0.0, 1.0 - dist / 0.18)
            if iou >= max_ship_iou or near >= max_ship_near:
                reject = True
                reject_label = dis.get("label")
                reject_iou = iou
                reject_near = near
                break
        if reject:
            det = dict(det)
            det["rejected_ship_label"] = reject_label
            det["rejected_ship_iou"] = reject_iou
            det["rejected_ship_near"] = reject_near
            continue
        filtered.append(det)
    return filtered


def task3_yellow_tugboat_override(det, stage2_margin, min_yellow_ratio=0.09, min_score=0.26, min_margin=-0.02):
    if det is None:
        return False
    if not _is_tugboat_target_label(det.get("label")):
        return False
    yellow_ratio = float(det.get("yellow_ratio", 0.0))
    score = float(det.get("final_score", det.get("score", 0.0)))
    if yellow_ratio < min_yellow_ratio:
        return False
    if score < min_score:
        return False
    if float(stage2_margin) < min_margin:
        return False
    return True


def save_raw_with_roi(
    obs,
    step_idx,
    save_dir,
    roi=None,
    state_xy=None,
    dino_box=None,
    rejected_dino_box=None,
    distractor_boxes=None,
):
    os.makedirs(save_dir, exist_ok=True)
    img_vis = to_uint8_image(obs.image)
    out_path = os.path.join(save_dir, f"raw_step_{step_idx:02d}.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_vis)
    ax.axis("off")

    if roi is not None:
        x1, y1, x2, y2 = roi
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

    if dino_box is not None:
        x1, y1, x2, y2 = dino_box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)

    if rejected_dino_box is not None:
        x1, y1, x2, y2 = rejected_dino_box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="white",
            facecolor="none",
            linestyle="--",
            alpha=0.75,
        )
        ax.add_patch(rect)

    if distractor_boxes is not None:
        for box in distractor_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="orange", facecolor="none")
            ax.add_patch(rect)

    if state_xy is not None:
        px, py = state_xy
        ax.plot(px, py, marker="x", markersize=10, markeredgewidth=2, color="red")

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[Save] {out_path}")


def save_heatmap_overlay(obs, weights, step_idx, save_dir, prefix="target", roi=None, state_xy=None):
    os.makedirs(save_dir, exist_ok=True)
    img_vis = to_uint8_image(obs.image)

    num_patches = weights.shape[1]
    grid_size = int(num_patches ** 0.5)
    w_map = weights[0].reshape(grid_size, grid_size).detach().cpu().numpy()
    w_map = (w_map - w_map.min()) / (w_map.max() - w_map.min() + 1e-8)

    h, w_img, _ = obs.image.shape
    heat = Image.fromarray((w_map * 255).astype(np.uint8)).resize((w_img, h), Image.BILINEAR)
    heat = np.array(heat) / 255.0

    out_path = os.path.join(save_dir, f"{prefix}_overlay_step_{step_idx:02d}.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_vis)
    ax.imshow(heat, cmap="jet", alpha=0.5)
    ax.axis("off")

    if roi is not None:
        x1, y1, x2, y2 = roi
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

    if state_xy is not None:
        px, py = state_xy
        ax.plot(px, py, marker="x", markersize=10, markeredgewidth=2, color="red")

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[Save] {out_path}")


def expand_box(box, img_h, img_w, scale=1.35, min_size=10):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(float(x2 - x1), float(min_size))
    bh = max(float(y2 - y1), float(min_size))
    bw *= scale
    bh *= scale

    nx1 = max(0.0, cx - 0.5 * bw)
    ny1 = max(0.0, cy - 0.5 * bh)
    nx2 = min(float(img_w), cx + 0.5 * bw)
    ny2 = min(float(img_h), cy + 0.5 * bh)
    return nx1, ny1, nx2, ny2


def crop_box_from_image(image_hwc_float, box, scale=1.35, min_size=10):
    img_u8 = to_uint8_image(image_hwc_float)
    img_h, img_w = img_u8.shape[:2]
    x1, y1, x2, y2 = expand_box(box, img_h, img_w, scale=scale, min_size=min_size)
    x1_i = int(max(0, min(img_w - 1, math.floor(x1))))
    y1_i = int(max(0, min(img_h - 1, math.floor(y1))))
    x2_i = int(max(x1_i + 1, min(img_w, math.ceil(x2))))
    y2_i = int(max(y1_i + 1, min(img_h, math.ceil(y2))))
    crop = img_u8[y1_i:y2_i, x1_i:x2_i]
    if crop.size == 0:
        return None
    return Image.fromarray(crop)


def make_stage2_candidates(target_detections, distractor_detections, top_k_target=5, top_k_distractor=5, iou_threshold=0.85):
    candidates = []

    def add_candidates(source_name, dets, limit):
        for det in dets[:limit]:
            if any(box_iou(det["box"], prev["box"]) >= iou_threshold for prev in candidates):
                continue
            cand = dict(det)
            cand["source"] = source_name
            candidates.append(cand)

    add_candidates("target", target_detections, top_k_target)
    add_candidates("distractor", distractor_detections, top_k_distractor)
    return candidates


def find_stage2_match_for_box(stage2_results, box, min_iou=0.70):
    best = None
    best_iou = 0.0
    for res in stage2_results:
        iou = box_iou(res["box"], box)
        if iou > best_iou:
            best = res
            best_iou = iou

    if best is None or best_iou < min_iou:
        return None, best_iou
    return best, best_iou


def save_stage2_crop_sheet(stage2_results, step_idx, save_dir):
    if len(stage2_results) == 0:
        return None

    out_path = os.path.join(save_dir, f"stage2_crops_step_{step_idx:02d}.png")
    cell_w = 220
    cell_h = 255
    cols = min(4, len(stage2_results))
    rows = int(math.ceil(len(stage2_results) / cols))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))

    for idx, res in enumerate(stage2_results):
        crop = res.get("crop")
        if crop is None:
            continue
        crop_vis = crop.convert("RGB")
        crop_vis.thumbnail((cell_w, 165))

        cell = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
        x = (cell_w - crop_vis.width) // 2
        cell.paste(crop_vis, (x, 8))

        draw = ImageDraw.Draw(cell)
        text_lines = [
            f"#{idx} {res['source']} {res['label'][:22]}",
            f"dino={res['dino_score']:.3f}",
            f"t={res['clip_target_score']:.3f} n={res['clip_negative_score']:.3f}",
            f"margin={res['clip_margin']:.3f}",
            f"T: {res['clip_target_prompt'][:24]}",
            f"N: {res['clip_negative_prompt'][:24]}",
        ]
        y = 176
        for line in text_lines:
            draw.text((8, y), line, fill=(0, 0, 0))
            y += 13

        sheet.paste(cell, ((idx % cols) * cell_w, (idx // cols) * cell_h))

    sheet.save(out_path)
    print(f"[Save] {out_path}")
    return out_path


def detections_to_patch_weights(detections, img_h, img_w, grid_size=14, sigma=1.2):
    heat = np.zeros((grid_size, grid_size), dtype=np.float32)

    for det in detections:
        # Prefer the target-vs-distractor adjusted score when present.
        score = max(0.0, float(det.get("final_score", det.get("score", 0.0))))
        x1, y1, x2, y2 = det["box"]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        gx = (cx / max(img_w, 1)) * grid_size - 0.5
        gy = (cy / max(img_h, 1)) * grid_size - 0.5

        for yy in range(grid_size):
            for xx in range(grid_size):
                d2 = (xx - gx) ** 2 + (yy - gy) ** 2
                heat[yy, xx] += score * math.exp(-d2 / (2 * sigma * sigma))

    if heat.max() <= 1e-8:
        heat += 1.0

    flat = heat.reshape(-1)
    flat = flat / (flat.sum() + 1e-8)
    return torch.from_numpy(flat).float().unsqueeze(0)


def filter_detections_by_geometry(detections, img_h, img_w, max_area_ratio=0.45, max_side_ratio=0.85):
    filtered = []
    img_area = float(max(img_h * img_w, 1))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area_ratio = (bw * bh) / img_area
        width_ratio = bw / max(float(img_w), 1.0)
        height_ratio = bh / max(float(img_h), 1.0)

        if area_ratio > max_area_ratio:
            continue
        if width_ratio > max_side_ratio or height_ratio > max_side_ratio:
            continue

        det = dict(det)
        det["area_ratio"] = area_ratio
        filtered.append(det)

    return filtered


def filter_ball_like_detections(
    detections,
    img_h,
    img_w,
    min_area_ratio=0.0003,
    max_aspect=2.4,
):
    filtered = []
    img_area = float(max(img_h * img_w, 1))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area_ratio = (bw * bh) / img_area
        aspect = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))

        if area_ratio < min_area_ratio:
            continue
        if aspect > max_aspect:
            continue

        det = dict(det)
        det["ball_area_ratio"] = area_ratio
        det["ball_aspect"] = aspect
        filtered.append(det)

    return filtered


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-8:
        return 0.0
    return inter / union


def box_center_distance_norm(box_a, box_b, img_h, img_w):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)

    diag = math.sqrt(float(img_h * img_h + img_w * img_w))
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / max(diag, 1e-6)


def rank_target_detections(
    target_detections,
    distractor_detections,
    img_h,
    img_w,
    distractor_penalty=0.8,
    nearby_penalty=0.4,
    nearby_dist_norm=0.18,
    min_distractor_score=0.25,
):
    ranked = []

    for det in target_detections:
        max_iou = 0.0
        max_near = 0.0
        best_distractor = None
        best_conflict = 0.0

        for dis in distractor_detections:
            if float(dis.get("score", 0.0)) < min_distractor_score:
                continue

            iou = box_iou(det["box"], dis["box"])
            dist = box_center_distance_norm(det["box"], dis["box"], img_h, img_w)
            near = max(0.0, 1.0 - dist / max(nearby_dist_norm, 1e-6))
            conflict = iou + near

            if conflict > best_conflict:
                best_distractor = dis
                best_conflict = conflict

            max_iou = max(max_iou, iou)
            max_near = max(max_near, near)

        final_score = float(det["score"]) - distractor_penalty * max_iou - nearby_penalty * max_near

        det = dict(det)
        det["target_score"] = float(det["score"])
        det["distractor_iou"] = float(max_iou)
        det["distractor_near"] = float(max_near)
        det["distractor_label"] = None if best_distractor is None else best_distractor.get("label")
        det["final_score"] = float(final_score)
        ranked.append(det)

    ranked.sort(key=lambda d: d["final_score"], reverse=True)
    return ranked


class GroundingDinoLocalizer:
    def __init__(self, device, model_id="IDEA-Research/grounding-dino-tiny"):
        if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
            raise ImportError(
                "transformers is not installed. Install first:\n"
                "  pip install transformers"
            )
        self.device = torch.device(device)
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image_hwc_float, prompt_text, box_threshold=0.2, text_threshold=0.2):
        img_u8 = to_uint8_image(image_hwc_float)
        pil = Image.fromarray(img_u8)
        inputs = self.processor(images=pil, text=prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]], device=self.device)  # (H, W)
        # transformers versions differ:
        # - older/newer releases may use `threshold`
        # - some releases use `box_threshold`
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]

        labels = results.get("text_labels", results.get("labels", []))
        detections = []
        for score, box, label in zip(results["scores"], results["boxes"], labels):
            x1, y1, x2, y2 = box.detach().cpu().tolist()
            detections.append({
                "score": float(score.detach().cpu().item()),
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "label": str(label),
            })
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections


class Stage2ClipReranker:
    def __init__(
        self,
        device,
        model_name="ViT-B-32",
        pretrained="openai",
        target_prompts=None,
        negative_prompts=None,
        crop_scale=1.35,
    ):
        if ClipVisionEncoder is None:
            raise ImportError(
                "open_clip is not available through models.clip_encoder_single. "
                "Disable STAGE2_ENABLE or install/check open_clip."
            )

        self.device = torch.device(device)
        self.crop_scale = crop_scale
        self.target_prompts = target_prompts or [
            "a photo of a magenta spherical ball",
            "a magenta ball",
            "a glossy magenta sphere",
        ]
        self.negative_prompts = negative_prompts or [
            "a navigation buoy",
            "a cone shaped buoy",
            "an iceberg",
            "a rock on the water",
            "ocean water",
            "a wave on the ocean",
            "sky and clouds",
        ]

        self.encoder = ClipVisionEncoder(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            freeze=True,
            image_size=224,
        ).to(self.device).eval()

        with torch.no_grad():
            self.target_text_features = self.encoder.encode_text(self.target_prompts)
            self.negative_text_features = self.encoder.encode_text(self.negative_prompts)

    def rerank(self, image_hwc_float, candidates):
        if len(candidates) == 0:
            return []

        crops = []
        valid_candidates = []
        for cand in candidates:
            crop = crop_box_from_image(image_hwc_float, cand["box"], scale=self.crop_scale)
            if crop is None:
                continue
            crops.append(crop)
            valid_candidates.append(cand)

        if len(crops) == 0:
            return []

        tensors = []
        for crop in crops:
            crop_for_clip = crop.convert("RGB").resize((224, 224), Image.BICUBIC)
            arr = np.asarray(crop_for_clip).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            tensors.append(tensor)
        image_tensor = torch.cat(tensors, dim=0)

        with torch.no_grad():
            image_features = self.encoder.forward_global(image_tensor)
            target_scores = image_features @ self.target_text_features.T
            negative_scores = image_features @ self.negative_text_features.T

        results = []
        for idx, cand in enumerate(valid_candidates):
            target_row = target_scores[idx]
            negative_row = negative_scores[idx]
            target_score, target_idx = torch.max(target_row, dim=0)
            negative_score, negative_idx = torch.max(negative_row, dim=0)
            margin = target_score - negative_score

            res = dict(cand)
            res["crop"] = crops[idx]
            res["dino_score"] = float(cand.get("final_score", cand.get("score", 0.0)))
            res["clip_target_score"] = float(target_score.detach().cpu().item())
            res["clip_negative_score"] = float(negative_score.detach().cpu().item())
            res["clip_margin"] = float(margin.detach().cpu().item())
            res["clip_target_prompt"] = self.target_prompts[int(target_idx.detach().cpu().item())]
            res["clip_negative_prompt"] = self.negative_prompts[int(negative_idx.detach().cpu().item())]
            results.append(res)

        results.sort(key=lambda d: d["clip_margin"], reverse=True)
        return results


def append_log_row(
    csv_path,
    row_dict,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("localhost", int(port)))
        except OSError:
            return False
    return True


def candidate_base_ports(preferred_base_port, preferred_worker_id, max_worker_tries):
    ports = [preferred_base_port]

    for base in [5004, 5014, 5024, 5034, 5104, 5204, 5304, 5404]:
        if base not in ports:
            ports.append(base)

    for _ in range(24):
        base = random.randint(20000, 58000)
        if base + preferred_worker_id + max_worker_tries < 60000 and base not in ports:
            ports.append(base)

    return ports


def create_env_with_retry(
    file_name,
    behavior_name,
    no_graphics,
    time_scale,
    base_port,
    preferred_worker_id,
    max_worker_tries=8,
):
    last_err = None
    for candidate_base_port_value in candidate_base_ports(base_port, preferred_worker_id, max_worker_tries):
        for offset in range(max_worker_tries):
            worker_id = preferred_worker_id + offset
            port = candidate_base_port_value + worker_id

            if not is_port_available(port):
                print(f"[3.1] port={port} already busy before Unity init, trying next...")
                continue

            try:
                env = UnitySingleAgentEnv(
                    file_name=file_name,
                    behavior_name=behavior_name,
                    no_graphics=no_graphics,
                    worker_id=worker_id,
                    time_scale=time_scale,
                    base_port=candidate_base_port_value,
                )
                print(f"[3.1] using worker_id={worker_id}, base_port={candidate_base_port_value}, port={port}")
                return env
            except UnityWorkerInUseException as e:
                last_err = e
                print(f"[3.1] worker_id={worker_id}, base_port={candidate_base_port_value} in use, trying next...")
                continue
            except OSError as e:
                last_err = e
                if "Address already in use" in str(e):
                    print(f"[3.1] port={port} became busy during Unity init, trying next...")
                    continue
                raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to create Unity environment with retry.")


def build_debug_task_config(task_name):
    task = (task_name or "magenta_ball").strip().lower().replace("-", "_")
    harbor_negatives = [
        "a pier", "a dock", "a port", "a harbor", "a quayside", "a seawall",
        "a harbor terminal", "a cargo terminal", "a concrete ground tile",
        "a concrete block", "a concrete pavement", "an asphalt road",
        "an industrial road", "a bridge", "an overpass", "a port building",
        "a concrete building", "a metal structure", "a crane", "a harbor crane",
        "ocean water", "a wave on the ocean", "sky and clouds",
    ]

    if task in ("magenta", "magenta_ball", "purple_ball", "task1"):
        return {
            "task_name": "magenta_ball",
            "target_prompts": ["magenta ball", "magenta sphere", "small magenta ball", "small magenta sphere"],
            "distractor_prompts": [
                "thin vertical buoy", "channel marker", "navigation marker pole",
                "floating marker pole", "cone shaped buoy", "ship", "boat",
                "vessel", "moored boat", "freighter", "tugboat",
            ] + harbor_negatives,
            "stage2_target_prompts": [
                "a photo of a magenta spherical target ball",
                "a small magenta ball",
                "a glossy magenta sphere",
                "a floating magenta target",
            ],
            "stage2_negative_prompts": [
                "a navigation buoy", "a cone shaped buoy", "a channel marker",
                "a thin marker pole", "a ship", "a boat", "a vessel",
                "a freighter", "a tugboat",
            ] + harbor_negatives,
            "use_ball_geometry_filter": True,
            "use_magenta_filter": True,
            "use_magenta_fallback": True,
            "min_magenta_ratio": 0.015,
            "fallback_min_magenta_ratio": 0.05,
            "dino_max_area_ratio": 0.45,
            "dino_max_side_ratio": 0.85,
            "stage2_crop_scale": 1.35,
        }

    if task in ("green", "green_buoy", "task2"):
        return {
            "task_name": "green_buoy",
            "target_prompts": [
                "green buoy", "green navigation buoy", "green floating buoy",
                "green cone buoy", "green conical buoy",
                "green channel marker buoy", "green marker buoy on the water",
                "green maritime buoy target", "green navigation marker on water",
            ],
            "distractor_prompts": [
                "magenta ball", "magenta target ball", "red buoy", "black buoy",
                "yellow buoy", "thin marker pole", "channel marker pole",
                "ship", "boat", "vessel", "freighter", "tugboat",
            ] + harbor_negatives,
            "stage2_target_prompts": [
                "a photo of a green navigation buoy",
                "a green floating buoy",
                "a green cone buoy",
                "a green conical navigation buoy",
                "a green channel marker buoy",
                "a green maritime marker on the water",
            ],
            "stage2_negative_prompts": [
                "a magenta target ball", "a red buoy", "a black buoy",
                "a yellow buoy", "a thin marker pole",
                "a cone shaped buoy that is not green", "a ship", "a boat",
                "a vessel", "a freighter", "a tugboat",
            ] + harbor_negatives,
            "use_ball_geometry_filter": False,
            "use_magenta_filter": False,
            "use_magenta_fallback": False,
            "use_green_fallback": True,
            "min_magenta_ratio": 0.0,
            "fallback_min_magenta_ratio": 0.0,
            "dino_visible_score": 0.24,
            "fallback_min_green_ratio": 0.02,
            "dino_max_area_ratio": 0.35,
            "dino_max_side_ratio": 0.75,
            "stage2_crop_scale": 1.45,
            "stage2_gate_margin_threshold": 0.005,
        }

    if task in ("yellow_tugboat", "tugboat", "vessel03", "vessl03", "task3", "yellow_tugboat_marker", "tugboat_marker", "task3_marker"):
        use_marker = task in ("yellow_tugboat_marker", "tugboat_marker", "task3_marker")
        target_prompts = [
            "bright green sphere",
            "green ball",
            "bright green spherical marker on top of the tugboat",
            "green ball marker on the boat",
            "bright green target marker above the yellow tugboat",
            "green sphere target marker",
            "neon green marker ball",
            "green round marker on the tugboat cabin",
        ] if use_marker else [
            "yellow tugboat", "yellow harbor tugboat", "small yellow tugboat",
            "yellow towboat", "yellow working tugboat with a cabin",
            "yellow TH Tugboat vessel", "Vessel03 yellow tugboat",
        ]
        stage2_target_prompts = [
            "a bright green sphere",
            "a green ball",
            "a bright green spherical marker",
            "a green ball marker on top of a boat",
            "a neon green target marker",
            "a green round marker above a tugboat",
            "a bright green sphere target",
        ] if use_marker else [
            "a photo of a yellow harbor tugboat",
            "a small yellow tugboat",
            "a yellow towboat with a cabin",
            "a yellow working tugboat in a harbor",
            "the Vessel03 yellow tugboat",
        ]
        return {
            "task_name": "yellow_tugboat_marker" if use_marker else "yellow_tugboat",
            "target_prompts": target_prompts,
            "distractor_prompts": [
                "magenta ball", "magenta target ball", "green buoy", "green navigation buoy", "red buoy",
                "black buoy", "yellow buoy", "thin marker pole",
                "channel marker", "navigation buoy", "large cargo ship",
                "freighter", "large vessel", "yellow tugboat", "yellow harbor tugboat",
            ] + harbor_negatives,
            "stage2_target_prompts": stage2_target_prompts,
            "stage2_negative_prompts": [
                "a magenta target ball", "a green navigation buoy", "a red buoy",
                "a black buoy", "a yellow buoy marker", "a thin marker pole",
                "a channel marker", "a large cargo ship", "a freighter",
                "a large vessel", "a yellow tugboat without a green marker",
            ] + harbor_negatives,
            "use_ball_geometry_filter": False,
            "use_magenta_filter": False,
            "use_magenta_fallback": False,
            "use_yellow_filter": False if use_marker else True,
            "min_yellow_ratio": 0.0 if use_marker else 0.018,
            "use_green_fallback": True if use_marker else False,
            "fallback_min_green_ratio": 0.005 if use_marker else 0.03,
            "reject_ship_distractor_conflicts": False if use_marker else True,
            "ship_conflict_max_iou": 0.45,
            "ship_conflict_max_near": 0.92,
            "ship_conflict_yellow_override_ratio": 0.08,
            "stage2_yellow_override": False if use_marker else True,
            "stage2_yellow_override_min_ratio": 0.09,
            "stage2_yellow_override_min_score": 0.26,
            "stage2_yellow_override_min_margin": -0.02,
            "min_magenta_ratio": 0.0,
            "fallback_min_magenta_ratio": 0.0,
            "dino_visible_score": 0.20 if use_marker else 0.22,
            "stage2_gate_margin_threshold": 0.0 if use_marker else 0.005,
            "dino_max_area_ratio": 0.18 if use_marker else 0.22,
            "dino_max_side_ratio": 0.55 if use_marker else 0.60,
            "stage2_crop_scale": 1.55 if use_marker else 1.25,
        }

    raise ValueError("Unknown HARBOR_TASK. Use magenta_ball, green_buoy, or yellow_tugboat.")


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HARBOR_TASK = os.environ.get("HARBOR_TASK") or (sys.argv[1] if len(sys.argv) > 1 else "magenta_ball")
    task_cfg = build_debug_task_config(HARBOR_TASK)
    TASK_SAVE_NAMES = {
        "magenta_ball": "task1_magenta_ball",
        "green_buoy": "task2_green_buoy",
        "yellow_tugboat": "task3_yellow_tugboat",
        "yellow_tugboat_marker": "task3_yellow_tugboat_marker",
    }
    task_save_name = TASK_SAVE_NAMES.get(task_cfg["task_name"], task_cfg["task_name"])
    SAVE_DIR = os.path.join(
        PROJECT_ROOT,
        "debug_frames_grounding_dino_target_state_harbor_negative",
        task_save_name,
    )
    os.makedirs(SAVE_DIR, exist_ok=True)

    for fn in os.listdir(SAVE_DIR):
        fp = os.path.join(SAVE_DIR, fn)
        if os.path.isfile(fp):
            os.remove(fp)

    CSV_PATH = os.path.join(SAVE_DIR, "target_state_log.csv")

    PROMPT_SET_NAME = task_cfg["task_name"]
    DISTRACTOR_PROMPTS = list(task_cfg["distractor_prompts"])
    STAGE2_TARGET_PROMPTS = list(task_cfg["stage2_target_prompts"])
    STAGE2_NEGATIVE_PROMPTS = list(task_cfg["stage2_negative_prompts"])
    DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    DINO_BOX_THRESHOLD = float(os.environ.get("DINO_BOX_THRESHOLD", "0.20"))
    DINO_TEXT_THRESHOLD = float(os.environ.get("DINO_TEXT_THRESHOLD", "0.20"))
    default_dino_visible_score = str(task_cfg.get("dino_visible_score", 0.28))
    DINO_VISIBLE_SCORE = float(os.environ.get("DINO_VISIBLE_SCORE", default_dino_visible_score))
    DINO_MAX_AREA_RATIO = float(os.environ.get("DINO_MAX_AREA_RATIO", str(task_cfg["dino_max_area_ratio"])))
    DINO_MAX_SIDE_RATIO = float(os.environ.get("DINO_MAX_SIDE_RATIO", str(task_cfg["dino_max_side_ratio"])))
    DINO_USE_DISTRACTOR_BRANCH = True
    DINO_DISTRACTOR_MODE = os.environ.get("DINO_DISTRACTOR_MODE", "visual_only")
    DINO_DISTRACTOR_IOU_PENALTY = 0.8
    DINO_DISTRACTOR_NEAR_PENALTY = 0.35
    DINO_DISTRACTOR_NEAR_DIST = 0.18
    DINO_DISTRACTOR_MIN_SCORE = 0.25
    DINO_USE_BALL_GEOMETRY_FILTER = os.environ.get(
        "DINO_USE_BALL_GEOMETRY_FILTER",
        "1" if task_cfg["use_ball_geometry_filter"] else "0",
    ) != "0"
    DINO_BALL_MIN_AREA_RATIO = 0.0003
    DINO_BALL_MAX_ASPECT = 2.4
    DINO_USE_MAGENTA_FILTER = os.environ.get(
        "DINO_USE_MAGENTA_FILTER",
        "1" if task_cfg["use_magenta_filter"] else "0",
    ) != "0"
    DINO_MIN_MAGENTA_RATIO = float(os.environ.get("DINO_MIN_MAGENTA_RATIO", str(task_cfg["min_magenta_ratio"])))
    DINO_USE_YELLOW_FILTER = os.environ.get(
        "DINO_USE_YELLOW_FILTER",
        "1" if task_cfg.get("use_yellow_filter", False) else "0",
    ) != "0"
    DINO_MIN_YELLOW_RATIO = float(os.environ.get("DINO_MIN_YELLOW_RATIO", str(task_cfg.get("min_yellow_ratio", 0.0))))
    REJECT_SHIP_DISTRACTOR_CONFLICTS = os.environ.get(
        "REJECT_SHIP_DISTRACTOR_CONFLICTS",
        "1" if task_cfg.get("reject_ship_distractor_conflicts", False) else "0",
    ) != "0"
    SHIP_CONFLICT_MAX_IOU = float(os.environ.get("SHIP_CONFLICT_MAX_IOU", str(task_cfg.get("ship_conflict_max_iou", 0.45))))
    SHIP_CONFLICT_MAX_NEAR = float(os.environ.get("SHIP_CONFLICT_MAX_NEAR", str(task_cfg.get("ship_conflict_max_near", 0.92))))
    SHIP_CONFLICT_YELLOW_OVERRIDE_RATIO = float(os.environ.get(
        "SHIP_CONFLICT_YELLOW_OVERRIDE_RATIO",
        str(task_cfg.get("ship_conflict_yellow_override_ratio", 0.08)),
    ))
    STAGE2_YELLOW_OVERRIDE = os.environ.get(
        "STAGE2_YELLOW_OVERRIDE",
        "1" if task_cfg.get("stage2_yellow_override", False) else "0",
    ) != "0"
    STAGE2_YELLOW_OVERRIDE_MIN_RATIO = float(os.environ.get(
        "STAGE2_YELLOW_OVERRIDE_MIN_RATIO",
        str(task_cfg.get("stage2_yellow_override_min_ratio", 0.09)),
    ))
    STAGE2_YELLOW_OVERRIDE_MIN_SCORE = float(os.environ.get(
        "STAGE2_YELLOW_OVERRIDE_MIN_SCORE",
        str(task_cfg.get("stage2_yellow_override_min_score", 0.26)),
    ))
    STAGE2_YELLOW_OVERRIDE_MIN_MARGIN = float(os.environ.get(
        "STAGE2_YELLOW_OVERRIDE_MIN_MARGIN",
        str(task_cfg.get("stage2_yellow_override_min_margin", -0.02)),
    ))
    USE_MAGENTA_FALLBACK = os.environ.get(
        "USE_MAGENTA_FALLBACK",
        "1" if task_cfg["use_magenta_fallback"] else "0",
    ) != "0"
    FALLBACK_MIN_MAGENTA_RATIO = float(os.environ.get("FALLBACK_MIN_MAGENTA_RATIO", str(task_cfg["fallback_min_magenta_ratio"])))
    USE_GREEN_FALLBACK = os.environ.get(
        "USE_GREEN_FALLBACK",
        "1" if task_cfg.get("use_green_fallback", False) else "0",
    ) != "0"
    FALLBACK_MIN_GREEN_RATIO = float(os.environ.get("FALLBACK_MIN_GREEN_RATIO", str(task_cfg.get("fallback_min_green_ratio", 0.03))))
    DRAW_REJECTED_TARGET_CANDIDATE = os.environ.get("DRAW_REJECTED_TARGET_CANDIDATE", "0") == "1"
    DRAW_DISTRACTOR_BOXES = os.environ.get("DRAW_DISTRACTOR_BOXES", "1") == "1"
    HIDE_DISTRACTOR_OVERLAPPING_TARGET = True
    STAGE2_ENABLE = os.environ.get("STAGE2_ENABLE", "1") == "1"
    STAGE2_APPLY_TO_FINAL = os.environ.get("STAGE2_APPLY_TO_FINAL", "0") == "1"
    STAGE2_CLIP_MODEL = os.environ.get("STAGE2_CLIP_MODEL", "ViT-B-32")
    STAGE2_CLIP_PRETRAINED = os.environ.get("STAGE2_CLIP_PRETRAINED", "openai")
    STAGE2_TOP_K_TARGET = int(os.environ.get("STAGE2_TOP_K_TARGET", "5"))
    STAGE2_TOP_K_DISTRACTOR = int(os.environ.get("STAGE2_TOP_K_DISTRACTOR", "5"))
    STAGE2_CROP_SCALE = float(os.environ.get("STAGE2_CROP_SCALE", str(task_cfg["stage2_crop_scale"])))
    STAGE2_GATE_VISIBLE = os.environ.get("STAGE2_GATE_VISIBLE", "1") == "1"
    STAGE2_GATE_MARGIN_THRESHOLD = float(os.environ.get(
        "STAGE2_GATE_MARGIN_THRESHOLD",
        str(task_cfg.get("stage2_gate_margin_threshold", 0.0)),
    ))
    STAGE2_GATE_MIN_IOU = float(os.environ.get("STAGE2_GATE_MIN_IOU", "0.70"))

    target_prompt_text = ". ".join(task_cfg["target_prompts"])
    if not target_prompt_text.endswith("."):
        target_prompt_text += "."
    distractor_prompt_text = ". ".join(DISTRACTOR_PROMPTS)
    if not distractor_prompt_text.endswith("."):
        distractor_prompt_text += "."

    print("[1] script started")
    print("[2] device:", DEVICE)
    print("[2.0] harbor_task:", task_cfg["task_name"])
    print("[2.0] save_dir:", SAVE_DIR)
    print("[2.1] prompt_set:", PROMPT_SET_NAME)
    print("[2.2] target_prompt_text:", target_prompt_text)
    print("[2.3] distractor_branch:", DINO_USE_DISTRACTOR_BRANCH)
    print("[2.4] distractor_prompt_text:", distractor_prompt_text)
    print("[2.5] dino_visible_score:", DINO_VISIBLE_SCORE)
    print("[2.6] distractor_mode:", DINO_DISTRACTOR_MODE)
    print("[2.7] stage2_enable:", STAGE2_ENABLE)
    print("[2.8] stage2_apply_to_final:", STAGE2_APPLY_TO_FINAL)
    print("[2.9] stage2_clip:", STAGE2_CLIP_MODEL, STAGE2_CLIP_PRETRAINED)
    print("[2.10] stage2_gate_visible:", STAGE2_GATE_VISIBLE)
    print("[2.11] stage2_gate_margin_threshold:", STAGE2_GATE_MARGIN_THRESHOLD)
    print("[2.12] stage2_target_prompts:", STAGE2_TARGET_PROMPTS)
    print("[2.13] stage2_negative_prompts:", STAGE2_NEGATIVE_PROMPTS)
    print("[2.14] ball_geometry_filter:", DINO_USE_BALL_GEOMETRY_FILTER)
    print("[2.15] magenta_filter:", DINO_USE_MAGENTA_FILTER)
    print("[2.16] magenta_fallback:", USE_MAGENTA_FALLBACK)
    print("[2.17] green_fallback:", USE_GREEN_FALLBACK)
    print("[2.18] yellow_filter:", DINO_USE_YELLOW_FILTER)
    print("[2.19] reject_ship_distractor_conflicts:", REJECT_SHIP_DISTRACTOR_CONFLICTS)
    print("[2.20] stage2_yellow_override:", STAGE2_YELLOW_OVERRIDE)

    base_port = int(os.environ.get("UNITY_BASE_PORT", "5004"))
    preferred_worker_id = int(os.environ.get("UNITY_WORKER_ID", "0"))
    behavior_name = os.environ.get("UNITY_BEHAVIOR_NAME", "USV?team=0")

    print("[3] creating Unity env...")
    env = create_env_with_retry(
        file_name=None,
        behavior_name=behavior_name,
        no_graphics=False,
        time_scale=1.0,
        base_port=base_port,
        preferred_worker_id=preferred_worker_id,
        max_worker_tries=12,
    )
    print("[4] Unity env created")

    print("[5] resetting env...")
    obs = env.reset()
    print("[6] env reset done")
    print("[6.1] obs.image shape:", obs.image.shape)
    print("[6.2] obs.vector shape:", obs.vector.shape)
    print("[6.3] obs.image dtype:", obs.image.dtype)
    print("[6.4] obs.image min/max:", obs.image.min(), obs.image.max())

    print("[7] loading Grounding DINO...")
    dino = GroundingDinoLocalizer(device=DEVICE, model_id=DINO_MODEL_ID)
    print("[8] Grounding DINO loaded")

    stage2 = None
    if STAGE2_ENABLE:
        print("[8.1] loading stage2 CLIP reranker...")
        stage2 = Stage2ClipReranker(
            device=DEVICE,
            model_name=STAGE2_CLIP_MODEL,
            pretrained=STAGE2_CLIP_PRETRAINED,
            target_prompts=STAGE2_TARGET_PROMPTS,
            negative_prompts=STAGE2_NEGATIVE_PROMPTS,
            crop_scale=STAGE2_CROP_SCALE,
        )
        print("[8.2] stage2 CLIP reranker loaded")

    fixed_action = np.array([0.5, 1.0], dtype=np.float32)

    warmup_steps = 50
    print(f"[9] warmup for {warmup_steps} steps with action {fixed_action} ...")
    for t in range(warmup_steps):
        obs, reward, done, info = env.step(fixed_action)
        print(f"[Warmup {t:02d}] reward={reward:.4f}, done={done}, vec_head={obs.vector[:8]}")
        if done:
            print(f"[Warmup {t:02d}] episode done during warmup, resetting env...")
            obs = env.reset()

    print("[10] start saving Grounding DINO debug frames...")

    record_start_after_steps = 30
    save_every = 10
    default_num_saved = 100 if task_cfg["task_name"] in ("green_buoy", "yellow_tugboat", "yellow_tugboat_marker") else 20
    num_saved = int(os.environ.get("NUM_SAVED", str(default_num_saved)))
    step_counter = 0
    saved_idx = 0

    direction_correct_count = 0
    direction_valid_count = 0
    target_best_in_ball_count = 0
    target_on_ball_sum = 0.0
    target_on_ball_valid_count = 0
    dino_visible_count = 0
    final_visible_count = 0
    fallback_visible_count = 0

    prev_ball_roi = None
    prev_green_roi = None
    fallback_streak = 0
    MAX_FALLBACK_STREAK = 5

    while saved_idx < num_saved:
        obs, reward, done, info = env.step(fixed_action)
        step_counter += 1
        print(f"[Move {step_counter:03d}] reward={reward:.4f}, done={done}")
        print(f"[Move {step_counter:03d}] vec_head={obs.vector[:8]}")

        if step_counter >= record_start_after_steps and (step_counter - record_start_after_steps) % save_every == 0:
            target_detections = dino.detect(
                obs.image,
                prompt_text=target_prompt_text,
                box_threshold=DINO_BOX_THRESHOLD,
                text_threshold=DINO_TEXT_THRESHOLD,
            )
            raw_target_count = len(target_detections)
            target_detections = filter_detections_by_geometry(
                target_detections,
                obs.image.shape[0],
                obs.image.shape[1],
                max_area_ratio=DINO_MAX_AREA_RATIO,
                max_side_ratio=DINO_MAX_SIDE_RATIO,
            )
            if DINO_USE_BALL_GEOMETRY_FILTER:
                target_detections = filter_ball_like_detections(
                    target_detections,
                    obs.image.shape[0],
                    obs.image.shape[1],
                    min_area_ratio=DINO_BALL_MIN_AREA_RATIO,
                    max_aspect=DINO_BALL_MAX_ASPECT,
                )
            if DINO_USE_MAGENTA_FILTER:
                target_detections = filter_detections_by_magenta_ratio(
                    target_detections,
                    obs.image,
                    min_ratio=DINO_MIN_MAGENTA_RATIO,
                )
            if DINO_USE_YELLOW_FILTER:
                target_detections = filter_detections_by_yellow_ratio(
                    target_detections,
                    obs.image,
                    min_ratio=DINO_MIN_YELLOW_RATIO,
                )

            if DINO_USE_DISTRACTOR_BRANCH:
                distractor_detections = dino.detect(
                    obs.image,
                    prompt_text=distractor_prompt_text,
                    box_threshold=DINO_BOX_THRESHOLD,
                    text_threshold=DINO_TEXT_THRESHOLD,
                )
                raw_distractor_count = len(distractor_detections)
                distractor_detections = filter_detections_by_geometry(
                    distractor_detections,
                    obs.image.shape[0],
                    obs.image.shape[1],
                    max_area_ratio=DINO_MAX_AREA_RATIO,
                    max_side_ratio=DINO_MAX_SIDE_RATIO,
                )
            else:
                distractor_detections = []
                raw_distractor_count = 0

            if REJECT_SHIP_DISTRACTOR_CONFLICTS:
                target_detections = filter_detections_by_ship_distractor_conflict(
                    target_detections,
                    distractor_detections,
                    obs.image.shape[0],
                    obs.image.shape[1],
                    max_ship_iou=SHIP_CONFLICT_MAX_IOU,
                    max_ship_near=SHIP_CONFLICT_MAX_NEAR,
                    min_ship_score=DINO_DISTRACTOR_MIN_SCORE,
                    yellow_override_ratio=SHIP_CONFLICT_YELLOW_OVERRIDE_RATIO,
                )

            distractor_iou_penalty = DINO_DISTRACTOR_IOU_PENALTY if DINO_DISTRACTOR_MODE == "penalty" else 0.0
            distractor_near_penalty = DINO_DISTRACTOR_NEAR_PENALTY if DINO_DISTRACTOR_MODE == "penalty" else 0.0

            detections = rank_target_detections(
                target_detections,
                distractor_detections,
                obs.image.shape[0],
                obs.image.shape[1],
                distractor_penalty=distractor_iou_penalty,
                nearby_penalty=distractor_near_penalty,
                nearby_dist_norm=DINO_DISTRACTOR_NEAR_DIST,
                min_distractor_score=DINO_DISTRACTOR_MIN_SCORE,
            )

            raw_dino_count = raw_target_count
            target_count = len(target_detections)
            distractor_count = len(distractor_detections)
            dino_count = len(detections)
            best_det = detections[0] if dino_count > 0 else None
            second_det = detections[1] if dino_count > 1 else None
            best_distractor_det = distractor_detections[0] if distractor_count > 0 else None

            best_score = 0.0 if best_det is None else float(best_det["final_score"])
            second_score = 0.0 if second_det is None else float(second_det["final_score"])
            best_target_score = 0.0 if best_det is None else float(best_det["target_score"])
            score_margin = best_score - second_score
            dino_visible_before_stage2 = bool(best_score >= DINO_VISIBLE_SCORE)
            dino_visible = dino_visible_before_stage2

            stage2_results = []
            stage2_best = None
            stage2_gate_match = None
            stage2_gate_match_iou = 0.0
            stage2_gate_margin = 0.0
            stage2_gate_pass = None
            stage2_gate_checked = False
            stage2_crop_sheet = None
            visible_source = "none"
            if stage2 is not None:
                stage2_candidates = make_stage2_candidates(
                    detections,
                    distractor_detections,
                    top_k_target=STAGE2_TOP_K_TARGET,
                    top_k_distractor=STAGE2_TOP_K_DISTRACTOR,
                )
                stage2_results = stage2.rerank(obs.image, stage2_candidates)
                stage2_best = stage2_results[0] if len(stage2_results) > 0 else None
                stage2_crop_sheet = save_stage2_crop_sheet(stage2_results, saved_idx, SAVE_DIR)

            if STAGE2_GATE_VISIBLE and stage2 is not None and dino_visible_before_stage2 and best_det is not None:
                stage2_gate_checked = True
                stage2_gate_match, stage2_gate_match_iou = find_stage2_match_for_box(
                    stage2_results,
                    best_det["box"],
                    min_iou=STAGE2_GATE_MIN_IOU,
                )
                if stage2_gate_match is not None:
                    stage2_gate_margin = float(stage2_gate_match["clip_margin"])
                    stage2_gate_pass = bool(stage2_gate_margin >= STAGE2_GATE_MARGIN_THRESHOLD)
                    if (
                        not stage2_gate_pass
                        and STAGE2_YELLOW_OVERRIDE
                        and task_cfg["task_name"] == "yellow_tugboat"
                        and task3_yellow_tugboat_override(
                            best_det,
                            stage2_gate_margin,
                            min_yellow_ratio=STAGE2_YELLOW_OVERRIDE_MIN_RATIO,
                            min_score=STAGE2_YELLOW_OVERRIDE_MIN_SCORE,
                            min_margin=STAGE2_YELLOW_OVERRIDE_MIN_MARGIN,
                        )
                    ):
                        stage2_gate_pass = True
                else:
                    stage2_gate_pass = False

                dino_visible = bool(dino_visible and stage2_gate_pass)

            if dino_visible:
                dino_visible_count += 1
                visible_source = "dino_stage2" if stage2_gate_checked else "dino"

            target_weights = detections_to_patch_weights(detections, obs.image.shape[0], obs.image.shape[1], grid_size=14)
            target_state = extract_target_state_from_weights(
                target_weights,
                img_h=obs.image.shape[0],
                img_w=obs.image.shape[1],
            )

            if best_det is not None:
                x1, y1, x2, y2 = best_det["box"]
                target_state["best_x"] = 0.5 * (x1 + x2)
                target_state["best_y"] = 0.5 * (y1 + y2)
                target_state["best_score"] = best_score
                target_state["visible"] = best_score
                target_state["score_margin"] = score_margin

                dir_left = 0.0
                dir_center = 0.0
                dir_right = 0.0
                cx = target_state["best_x"]
                if cx < obs.image.shape[1] / 3:
                    dir_left = 1.0
                elif cx < 2 * obs.image.shape[1] / 3:
                    dir_center = 1.0
                else:
                    dir_right = 1.0
                target_state["dir_left"] = dir_left
                target_state["dir_center"] = dir_center
                target_state["dir_right"] = dir_right

            hint_xy = None

            ball_roi, roi_used_fallback, candidate_count, component_count = find_ball_roi(
                obs.image,
                hint_xy=hint_xy,
                prev_roi=prev_ball_roi,
            )

            if roi_used_fallback:
                fallback_streak += 1
            else:
                fallback_streak = 0

            if fallback_streak > MAX_FALLBACK_STREAK:
                prev_ball_roi = None
            elif ball_roi is not None:
                prev_ball_roi = ball_roi

            fallback_visible = False
            fallback_box = None
            fallback_magenta_ratio = 0.0
            fallback_green_ratio = 0.0
            if USE_MAGENTA_FALLBACK and not dino_visible and ball_roi is not None and not roi_used_fallback:
                fallback_box = tuple(float(v) for v in ball_roi)
                fallback_magenta_ratio = box_magenta_ratio(obs.image, fallback_box)
                fallback_visible = bool(fallback_magenta_ratio >= FALLBACK_MIN_MAGENTA_RATIO)

            green_roi = None
            green_roi_used_fallback = False
            green_candidate_count = 0
            green_component_count = 0
            if USE_GREEN_FALLBACK and not dino_visible:
                green_roi, green_roi_used_fallback, green_candidate_count, green_component_count = find_green_roi(
                    obs.image,
                    prev_roi=prev_green_roi,
                )
                if green_roi is not None and not green_roi_used_fallback:
                    fallback_box = tuple(float(v) for v in green_roi)
                    fallback_green_ratio = box_green_ratio(obs.image, fallback_box)
                    fallback_visible = bool(fallback_green_ratio >= FALLBACK_MIN_GREEN_RATIO)
                if green_roi_used_fallback:
                    pass
                elif green_roi is not None:
                    prev_green_roi = green_roi

            if fallback_visible:
                dino_visible = True
                visible_source = "green_fallback" if fallback_green_ratio > 0.0 else "magenta_fallback"
                fallback_visible_count += 1
                target_state = target_state_from_box(
                    fallback_box,
                    img_h=obs.image.shape[0],
                    img_w=obs.image.shape[1],
                    score=fallback_green_ratio if fallback_green_ratio > 0.0 else fallback_magenta_ratio,
                    score_margin=fallback_green_ratio if fallback_green_ratio > 0.0 else fallback_magenta_ratio,
                )
                target_weights = detections_to_patch_weights(
                    [{
                        "box": fallback_box,
                        "final_score": fallback_green_ratio if fallback_green_ratio > 0.0 else fallback_magenta_ratio,
                        "score": fallback_green_ratio if fallback_green_ratio > 0.0 else fallback_magenta_ratio,
                    }],
                    obs.image.shape[0],
                    obs.image.shape[1],
                    grid_size=14,
                )

            if dino_visible:
                final_visible_count += 1

            target_map = weights_to_map(target_weights)
            target_on_ball = None
            target_best_in_ball = None

            if ball_roi is not None:
                target_on_ball = roi_mean_from_patch_map(
                    target_map, *ball_roi,
                    img_h=obs.image.shape[0], img_w=obs.image.shape[1]
                )
                if dino_visible:
                    target_best_in_ball = target_state_hit_ball_roi(target_state, ball_roi)

            gt_dir = gt_direction_from_ball_roi(ball_roi, img_w=obs.image.shape[1])
            pred_dir = predicted_direction_from_state(target_state) if dino_visible else None
            direction_correct = None
            if gt_dir is not None and pred_dir is not None:
                direction_correct = (gt_dir == pred_dir)
                direction_valid_count += 1
                if direction_correct:
                    direction_correct_count += 1

            if target_best_in_ball is True:
                target_best_in_ball_count += 1
            if target_on_ball is not None and dino_visible:
                target_on_ball_sum += target_on_ball
                target_on_ball_valid_count += 1

            print(f"\n[Save Step {saved_idx:02d}] at env step {step_counter}")
            print("[DINO TARGET]")
            print("  raw_target_count           :", raw_target_count)
            print("  target_count               :", target_count)
            print("  final_target_count         :", dino_count)
            print("  dino_final_score           :", best_score)
            print("  dino_target_score          :", best_target_score)
            print("  dino_best_magenta_ratio    :", 0.0 if best_det is None else best_det.get("magenta_ratio", 0.0))
            print("  dino_best_yellow_ratio     :", 0.0 if best_det is None else best_det.get("yellow_ratio", 0.0))
            print("  dino_second_final_score    :", second_score)
            print("  dino_score_margin          :", score_margin)
            print("  dino_visible_before_stage2 :", dino_visible_before_stage2)
            print("  dino_visible               :", dino_visible)
            print("  visible_source             :", visible_source)
            print("  best_label                 :", None if best_det is None else best_det["label"])
            print("  best_box                   :", None if best_det is None else tuple(round(v, 2) for v in best_det["box"]))
            print("  best_distractor_iou        :", 0.0 if best_det is None else best_det["distractor_iou"])
            print("  best_distractor_near       :", 0.0 if best_det is None else best_det["distractor_near"])
            print("  best_distractor_label      :", None if best_det is None else best_det["distractor_label"])
            print("[DINO DISTRACTOR]")
            print("  raw_distractor_count       :", raw_distractor_count)
            print("  distractor_count           :", distractor_count)
            print("  best_distractor_score      :", 0.0 if best_distractor_det is None else best_distractor_det["score"])
            print("  best_distractor_label      :", None if best_distractor_det is None else best_distractor_det["label"])
            print("  best_distractor_box        :", None if best_distractor_det is None else tuple(round(v, 2) for v in best_distractor_det["box"]))
            print("[STAGE2 CLIP RERANK]")
            print("  stage2_enabled             :", stage2 is not None)
            print("  stage2_candidate_count     :", len(stage2_results))
            print("  stage2_best_source         :", None if stage2_best is None else stage2_best["source"])
            print("  stage2_best_label          :", None if stage2_best is None else stage2_best["label"])
            print("  stage2_best_margin         :", 0.0 if stage2_best is None else stage2_best["clip_margin"])
            print("  stage2_best_target_score   :", 0.0 if stage2_best is None else stage2_best["clip_target_score"])
            print("  stage2_best_negative_score :", 0.0 if stage2_best is None else stage2_best["clip_negative_score"])
            print("  stage2_best_target_prompt  :", None if stage2_best is None else stage2_best["clip_target_prompt"])
            print("  stage2_best_negative_prompt:", None if stage2_best is None else stage2_best["clip_negative_prompt"])
            print("  stage2_gate_visible        :", STAGE2_GATE_VISIBLE)
            print("  stage2_gate_checked        :", stage2_gate_checked)
            print("  stage2_gate_pass           :", stage2_gate_pass)
            print("  stage2_gate_match_iou      :", stage2_gate_match_iou)
            print("  stage2_gate_margin         :", stage2_gate_margin)
            print("  stage2_gate_label          :", None if stage2_gate_match is None else stage2_gate_match["label"])
            print("  stage2_gate_negative_prompt:", None if stage2_gate_match is None else stage2_gate_match["clip_negative_prompt"])
            print("  stage2_crop_sheet          :", stage2_crop_sheet)
            print("[ROI]")
            print("  component_count            :", component_count)
            print("  candidate_count            :", candidate_count)
            print("  auto_ball_roi              :", ball_roi)
            print("  roi_used_fallback          :", roi_used_fallback)
            print("  fallback_streak            :", fallback_streak)
            print("[MAGENTA FALLBACK]")
            print("  fallback_enabled           :", USE_MAGENTA_FALLBACK)
            print("  fallback_visible           :", fallback_visible)
            print("  fallback_box               :", None if fallback_box is None else tuple(round(v, 2) for v in fallback_box))
            print("  fallback_magenta_ratio     :", fallback_magenta_ratio)
            print("[GREEN FALLBACK]")
            print("  fallback_enabled           :", USE_GREEN_FALLBACK)
            print("  green_component_count      :", green_component_count)
            print("  green_candidate_count      :", green_candidate_count)
            print("  green_roi                  :", green_roi)
            print("  green_roi_used_fallback    :", green_roi_used_fallback)
            print("  fallback_green_ratio       :", fallback_green_ratio)

            row = {
                "save_idx": saved_idx,
                "env_step": step_counter,
                "prompt_set": PROMPT_SET_NAME,
                "prompt_text": target_prompt_text,
                "target_prompt_text": target_prompt_text,
                "distractor_prompt_text": distractor_prompt_text,
                "distractor_mode": DINO_DISTRACTOR_MODE,
                "dino_visible_score_threshold": DINO_VISIBLE_SCORE,
                "raw_dino_count": raw_dino_count,
                "dino_count": dino_count,
                "raw_target_count": raw_target_count,
                "target_count": target_count,
                "raw_distractor_count": raw_distractor_count,
                "distractor_count": distractor_count,
                "dino_best_label": None if best_det is None else best_det["label"],
                "dino_best_score": best_score,
                "dino_best_magenta_ratio": 0.0 if best_det is None else best_det.get("magenta_ratio", 0.0),
                "dino_best_yellow_ratio": 0.0 if best_det is None else best_det.get("yellow_ratio", 0.0),
                "dino_target_score": best_target_score,
                "dino_second_score": second_score,
                "dino_score_margin": score_margin,
                "dino_visible_before_stage2": dino_visible_before_stage2,
                "dino_visible": dino_visible,
                "visible_source": visible_source,
                "dino_box": None if best_det is None else tuple(round(v, 2) for v in best_det["box"]),
                "fallback_visible": fallback_visible,
                "fallback_box": None if fallback_box is None else tuple(round(v, 2) for v in fallback_box),
                "fallback_magenta_ratio": fallback_magenta_ratio,
                "fallback_green_ratio": fallback_green_ratio,
                "green_roi": green_roi,
                "green_roi_used_fallback": green_roi_used_fallback,
                "green_component_count": green_component_count,
                "green_candidate_count": green_candidate_count,
                "dino_distractor_iou": 0.0 if best_det is None else best_det["distractor_iou"],
                "dino_distractor_near": 0.0 if best_det is None else best_det["distractor_near"],
                "dino_distractor_label": None if best_det is None else best_det["distractor_label"],
                "best_distractor_label": None if best_distractor_det is None else best_distractor_det["label"],
                "best_distractor_score": 0.0 if best_distractor_det is None else best_distractor_det["score"],
                "best_distractor_box": None if best_distractor_det is None else tuple(round(v, 2) for v in best_distractor_det["box"]),
                "stage2_enabled": stage2 is not None,
                "stage2_apply_to_final": STAGE2_APPLY_TO_FINAL,
                "stage2_candidate_count": len(stage2_results),
                "stage2_best_source": None if stage2_best is None else stage2_best["source"],
                "stage2_best_label": None if stage2_best is None else stage2_best["label"],
                "stage2_best_box": None if stage2_best is None else tuple(round(v, 2) for v in stage2_best["box"]),
                "stage2_best_dino_score": 0.0 if stage2_best is None else stage2_best["dino_score"],
                "stage2_best_target_score": 0.0 if stage2_best is None else stage2_best["clip_target_score"],
                "stage2_best_negative_score": 0.0 if stage2_best is None else stage2_best["clip_negative_score"],
                "stage2_best_margin": 0.0 if stage2_best is None else stage2_best["clip_margin"],
                "stage2_best_target_prompt": None if stage2_best is None else stage2_best["clip_target_prompt"],
                "stage2_best_negative_prompt": None if stage2_best is None else stage2_best["clip_negative_prompt"],
                "stage2_gate_visible": STAGE2_GATE_VISIBLE,
                "stage2_gate_checked": stage2_gate_checked,
                "stage2_gate_pass": stage2_gate_pass,
                "stage2_gate_margin_threshold": STAGE2_GATE_MARGIN_THRESHOLD,
                "stage2_gate_min_iou": STAGE2_GATE_MIN_IOU,
                "stage2_gate_match_iou": stage2_gate_match_iou,
                "stage2_gate_margin": stage2_gate_margin,
                "stage2_gate_label": None if stage2_gate_match is None else stage2_gate_match["label"],
                "stage2_gate_source": None if stage2_gate_match is None else stage2_gate_match["source"],
                "stage2_gate_target_score": 0.0 if stage2_gate_match is None else stage2_gate_match["clip_target_score"],
                "stage2_gate_negative_score": 0.0 if stage2_gate_match is None else stage2_gate_match["clip_negative_score"],
                "stage2_gate_target_prompt": None if stage2_gate_match is None else stage2_gate_match["clip_target_prompt"],
                "stage2_gate_negative_prompt": None if stage2_gate_match is None else stage2_gate_match["clip_negative_prompt"],
                "stage2_crop_sheet": stage2_crop_sheet,
                "component_count": component_count,
                "candidate_count": candidate_count,
                "roi_used_fallback": roi_used_fallback,
                "ball_roi": ball_roi,
                "target_best_x": target_state["best_x"],
                "target_best_y": target_state["best_y"],
                "target_best_in_ball": target_best_in_ball,
                "target_on_ball": target_on_ball,
                "gt_dir": gt_dir,
                "pred_dir": pred_dir,
                "direction_correct": direction_correct,
            }
            append_log_row(CSV_PATH, row)

            state_xy = None if not dino_visible else (target_state["best_x"], target_state["best_y"])
            if dino_visible and visible_source in ("magenta_fallback", "green_fallback"):
                best_box = fallback_box
            else:
                best_box = None if best_det is None or not dino_visible else best_det["box"]
            rejected_box = None
            if DRAW_REJECTED_TARGET_CANDIDATE and best_det is not None and not dino_visible:
                rejected_box = best_det["box"]
            distractor_boxes = []
            if DRAW_DISTRACTOR_BOXES:
                for det in distractor_detections:
                    if len(distractor_boxes) >= 3:
                        break
                    if float(det.get("score", 0.0)) < DINO_DISTRACTOR_MIN_SCORE:
                        continue
                    if HIDE_DISTRACTOR_OVERLAPPING_TARGET and best_det is not None:
                        iou = box_iou(det["box"], best_det["box"])
                        near = box_center_distance_norm(det["box"], best_det["box"], obs.image.shape[0], obs.image.shape[1])
                        if iou > 0.25 or near < 0.08:
                            continue
                    distractor_boxes.append(det["box"])
            if USE_GREEN_FALLBACK and green_roi is not None and not green_roi_used_fallback:
                visual_roi = green_roi
            else:
                visual_roi = None if roi_used_fallback else ball_roi

            save_raw_with_roi(
                obs,
                saved_idx,
                SAVE_DIR,
                roi=visual_roi,
                state_xy=state_xy,
                dino_box=best_box,
                rejected_dino_box=rejected_box,
                distractor_boxes=distractor_boxes,
            )
            save_heatmap_overlay(
                obs,
                target_weights,
                saved_idx,
                SAVE_DIR,
                prefix="target",
                roi=visual_roi,
                state_xy=state_xy,
            )

            saved_idx += 1

        if done:
            print(f"[Move {step_counter:03d}] episode done, resetting and continue collecting...")
            obs = env.reset()
            continue

    print("\n[SUMMARY]")
    direction_accuracy = direction_correct_count / max(direction_valid_count, 1)
    target_best_in_ball_ratio = target_best_in_ball_count / max(saved_idx, 1)
    target_on_ball_mean = target_on_ball_sum / max(target_on_ball_valid_count, 1)

    print("  saved_frames                :", saved_idx)
    print("  dino_visible_count          :", dino_visible_count)
    print("  dino_visible_ratio          :", dino_visible_count / max(saved_idx, 1))
    print("  fallback_visible_count      :", fallback_visible_count)
    print("  final_visible_count         :", final_visible_count)
    print("  final_visible_ratio         :", final_visible_count / max(saved_idx, 1))
    print("  direction_valid_count       :", direction_valid_count)
    print("  direction_accuracy          :", direction_accuracy)
    print("  target_best_in_ball_ratio   :", target_best_in_ball_ratio)
    print("  target_on_ball_mean         :", target_on_ball_mean)
    print("  save_dir                    :", SAVE_DIR)

    env.close()
    print("[11] env closed")


if __name__ == "__main__":
    main()
