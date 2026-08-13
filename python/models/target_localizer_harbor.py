from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple
import time

import numpy as np
import torch

from scripts.debug_grounding_dino_target_state_harbor_negative import (
    GroundingDinoLocalizer,
    Stage2ClipReranker,
    box_magenta_ratio,
    box_green_ratio,
    box_yellow_ratio,
    filter_ball_like_detections,
    filter_detections_by_geometry,
    filter_detections_by_magenta_ratio,
    filter_detections_by_ship_distractor_conflict,
    filter_detections_by_yellow_ratio,
    find_ball_roi,
    find_green_roi,
    find_stage2_match_for_box,
    make_stage2_candidates,
    rank_target_detections,
    task3_yellow_tugboat_override,
)


TARGET_STATE_DIM = 8


HARBOR_TARGET_PROMPTS = [
    "magenta ball",
    "magenta sphere",
    "small magenta ball",
    "small magenta spherical target",
    "floating magenta target ball",
]

HARBOR_GREEN_BUOY_TARGET_PROMPTS = [
    "green buoy",
    "green navigation buoy",
    "green floating buoy",
    "green cone buoy",
    "green conical buoy",
    "green channel marker buoy",
    "green marker buoy on the water",
    "green maritime buoy target",
    "green navigation marker on water",
]

HARBOR_YELLOW_TUGBOAT_TARGET_PROMPTS = [
    "yellow tugboat",
    "yellow harbor tugboat",
    "small yellow tugboat",
    "yellow towboat",
    "yellow working tugboat with a cabin",
    "yellow TH Tugboat vessel",
    "Vessel03 yellow tugboat",
]

HARBOR_TUGBOAT_MARKER_TARGET_PROMPTS = [
    "bright green sphere",
    "green ball",
    "bright green spherical marker on top of the tugboat",
    "green ball marker on the boat",
    "bright green target marker above the yellow tugboat",
    "green sphere target marker",
    "neon green marker ball",
    "green round marker on the tugboat cabin",
]

HARBOR_DISTRACTOR_PROMPTS = [
    "thin vertical buoy",
    "channel marker",
    "navigation marker pole",
    "floating marker pole",
    "cone shaped buoy",
    "port",
    "harbor",
    "pier",
    "dock",
    "quayside",
    "seawall",
    "terminal",
    "harbor terminal",
    "cargo terminal",
    "concrete ground",
    "concrete tile",
    "concrete block",
    "pavement",
    "road",
    "industrial road",
    "bridge",
    "overpass",
    "port building",
    "concrete building",
    "metal structure",
    "crane",
    "harbor crane",
    "container",
    "ship",
    "boat",
    "vessel",
    "moored boat",
    "freighter",
    "tugboat",
    "ocean water",
    "wave",
]

HARBOR_GREEN_BUOY_DISTRACTOR_PROMPTS = [
    "magenta ball",
    "magenta target ball",
    "red buoy",
    "black buoy",
    "yellow buoy",
    "thin marker pole",
    "channel marker pole",
    "port",
    "harbor",
    "pier",
    "dock",
    "terminal",
    "road",
    "concrete ground",
    "ship",
    "boat",
    "vessel",
    "freighter",
    "tugboat",
    "ocean water",
    "wave",
]

HARBOR_YELLOW_TUGBOAT_DISTRACTOR_PROMPTS = [
    "magenta ball",
    "magenta target ball",
    "green buoy",
    "red buoy",
    "black buoy",
    "yellow buoy",
    "thin marker pole",
    "channel marker",
    "navigation buoy",
    "port",
    "harbor",
    "pier",
    "dock",
    "quayside",
    "terminal",
    "road",
    "concrete ground",
    "harbor crane",
    "container",
    "large cargo ship",
    "freighter",
    "large vessel",
    "ocean water",
    "wave",
]

HARBOR_STAGE2_NEGATIVE_PROMPTS = [
    "a navigation buoy",
    "a cone shaped buoy",
    "a channel marker",
    "a thin marker pole",
    "a pier",
    "a dock",
    "a port",
    "a harbor",
    "a quayside",
    "a seawall",
    "a harbor terminal",
    "a cargo terminal",
    "a concrete ground tile",
    "a concrete block",
    "a concrete pavement",
    "an asphalt road",
    "an industrial road",
    "a bridge",
    "an overpass",
    "a port building",
    "a concrete building",
    "a metal structure",
    "a crane",
    "a harbor crane",
    "a ship",
    "a boat",
    "a vessel",
    "a freighter",
    "a tugboat",
    "ocean water",
    "a wave on the ocean",
    "sky and clouds",
]

HARBOR_GREEN_BUOY_STAGE2_TARGET_PROMPTS = [
    "a photo of a green navigation buoy",
    "a green floating buoy",
    "a green cone buoy",
    "a green conical navigation buoy",
    "a green channel marker buoy",
    "a green maritime marker on the water",
]

HARBOR_GREEN_BUOY_STAGE2_NEGATIVE_PROMPTS = [
    "a magenta target ball",
    "a red buoy",
    "a black buoy",
    "a yellow buoy",
    "a thin marker pole",
    "a cone shaped buoy that is not green",
    "a pier",
    "a dock",
    "a port",
    "a harbor terminal",
    "a concrete pavement",
    "a road",
    "a ship",
    "a boat",
    "a vessel",
    "a freighter",
    "a tugboat",
    "ocean water",
    "a wave on the ocean",
    "sky and clouds",
]

HARBOR_YELLOW_TUGBOAT_STAGE2_TARGET_PROMPTS = [
    "a photo of a yellow harbor tugboat",
    "a small yellow tugboat",
    "a yellow towboat with a cabin",
    "a yellow working tugboat in a harbor",
    "the Vessel03 yellow tugboat",
]

HARBOR_TUGBOAT_MARKER_STAGE2_TARGET_PROMPTS = [
    "a bright green sphere",
    "a green ball",
    "a bright green spherical marker",
    "a green ball marker on top of a boat",
    "a neon green target marker",
    "a green round marker above a tugboat",
    "a bright green sphere target",
]

HARBOR_YELLOW_TUGBOAT_STAGE2_NEGATIVE_PROMPTS = [
    "a magenta target ball",
    "a green buoy",
    "a red buoy",
    "a black buoy",
    "a yellow buoy marker",
    "a thin marker pole",
    "a channel marker",
    "a pier",
    "a dock",
    "a port",
    "a harbor terminal",
    "a concrete pavement",
    "a road",
    "a harbor crane",
    "a container",
    "a large cargo ship",
    "a freighter",
    "a large vessel",
    "ocean water",
    "a wave on the ocean",
    "sky and clouds",
]


@dataclass
class HarborTargetLocalizerConfig:
    task_name: str = "magenta_ball"
    device: str = "cuda"
    target_prompts: Sequence[str] = field(default_factory=lambda: HARBOR_TARGET_PROMPTS)
    distractor_prompts: Sequence[str] = field(default_factory=lambda: HARBOR_DISTRACTOR_PROMPTS)
    stage2_target_prompts: Sequence[str] = field(default_factory=lambda: [
        "a photo of a magenta spherical target ball",
        "a small magenta ball",
        "a glossy magenta sphere",
        "a floating magenta target",
    ])
    stage2_negative_prompts: Sequence[str] = field(default_factory=lambda: HARBOR_STAGE2_NEGATIVE_PROMPTS)
    dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    dino_box_threshold: float = 0.20
    dino_text_threshold: float = 0.20
    dino_visible_score: float = 0.28
    dino_max_area_ratio: float = 0.45
    dino_max_side_ratio: float = 0.85
    use_ball_geometry_filter: bool = True
    ball_min_area_ratio: float = 0.0003
    ball_max_aspect: float = 2.4
    use_magenta_filter: bool = True
    min_magenta_ratio: float = 0.015
    use_yellow_filter: bool = False
    min_yellow_ratio: float = 0.0
    reject_ship_distractor_conflicts: bool = False
    ship_conflict_max_iou: float = 0.45
    ship_conflict_max_near: float = 0.92
    ship_conflict_yellow_override_ratio: float = 0.08
    stage2_yellow_override: bool = False
    stage2_yellow_override_min_ratio: float = 0.09
    stage2_yellow_override_min_score: float = 0.26
    stage2_yellow_override_min_margin: float = -0.02
    use_distractor_branch: bool = True
    distractor_mode: str = "visual_only"
    distractor_iou_penalty: float = 0.8
    distractor_near_penalty: float = 0.35
    distractor_near_dist: float = 0.18
    distractor_min_score: float = 0.25
    stage2_enable: bool = True
    stage2_clip_model: str = "ViT-B-32"
    stage2_clip_pretrained: str = "openai"
    stage2_top_k_target: int = 5
    stage2_top_k_distractor: int = 5
    stage2_crop_scale: float = 1.35
    stage2_gate_visible: bool = True
    stage2_gate_margin_threshold: float = 0.0
    stage2_gate_min_iou: float = 0.70
    use_magenta_fallback: bool = True
    fallback_min_magenta_ratio: float = 0.05
    use_green_fallback: bool = False
    fallback_min_green_ratio: float = 0.03

    @property
    def target_prompt_text(self) -> str:
        return _join_prompt_text(self.target_prompts)

    @property
    def distractor_prompt_text(self) -> str:
        return _join_prompt_text(self.distractor_prompts)


@dataclass
class HarborTargetLocalizationResult:
    visible: bool
    state: np.ndarray
    box: Optional[Tuple[float, float, float, float]]
    center_xy: Tuple[float, float]
    center_norm_xy: Tuple[float, float]
    direction: str
    visible_source: str
    cache_age_steps: int

    dino_visible_before_stage2: bool
    dino_label: Optional[str]
    dino_score: float
    dino_second_score: float
    dino_score_margin: float
    dino_magenta_ratio: float
    dino_yellow_ratio: float
    raw_target_count: int
    target_count: int
    raw_distractor_count: int
    distractor_count: int

    fallback_visible: bool
    fallback_magenta_ratio: float

    stage2_checked: bool
    stage2_pass: Optional[bool]
    stage2_margin: float
    stage2_match_iou: float
    stage2_target_score: float
    stage2_negative_score: float
    stage2_target_prompt: Optional[str]
    stage2_negative_prompt: Optional[str]

    debug: Dict[str, Any] = field(default_factory=dict)


class HarborTargetLocalizer:
    def __init__(self, config: Optional[HarborTargetLocalizerConfig] = None):
        self.config = config or HarborTargetLocalizerConfig()
        self.last_profile: Dict[str, float] = {}
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.config.device = "cpu"

        self.dino = GroundingDinoLocalizer(
            device=self.config.device,
            model_id=self.config.dino_model_id,
        )

        self.stage2 = None
        if self.config.stage2_enable:
            self.stage2 = Stage2ClipReranker(
                device=self.config.device,
                model_name=self.config.stage2_clip_model,
                pretrained=self.config.stage2_clip_pretrained,
                target_prompts=self.config.stage2_target_prompts,
                negative_prompts=self.config.stage2_negative_prompts,
                crop_scale=self.config.stage2_crop_scale,
            )

    def localize(self, image_hwc_float: np.ndarray) -> HarborTargetLocalizationResult:
        cfg = self.config
        img_h, img_w = image_hwc_float.shape[:2]
        profile_t0 = time.perf_counter()
        dino_ms = 0.0
        clip_ms = 0.0
        target_dino_ms = 0.0
        distractor_dino_ms = 0.0

        t0 = time.perf_counter()
        target_detections = self.dino.detect(
            image_hwc_float,
            prompt_text=cfg.target_prompt_text,
            box_threshold=cfg.dino_box_threshold,
            text_threshold=cfg.dino_text_threshold,
        )
        target_dino_ms = (time.perf_counter() - t0) * 1000.0
        dino_ms += target_dino_ms
        raw_target_count = len(target_detections)
        target_detections = filter_detections_by_geometry(
            target_detections,
            img_h,
            img_w,
            max_area_ratio=cfg.dino_max_area_ratio,
            max_side_ratio=cfg.dino_max_side_ratio,
        )
        if cfg.use_ball_geometry_filter:
            target_detections = filter_ball_like_detections(
                target_detections,
                img_h,
                img_w,
                min_area_ratio=cfg.ball_min_area_ratio,
                max_aspect=cfg.ball_max_aspect,
            )
        if cfg.use_magenta_filter:
            target_detections = filter_detections_by_magenta_ratio(
                target_detections,
                image_hwc_float,
                min_ratio=cfg.min_magenta_ratio,
            )
        if cfg.use_yellow_filter:
            target_detections = filter_detections_by_yellow_ratio(
                target_detections,
                image_hwc_float,
                min_ratio=cfg.min_yellow_ratio,
            )

        distractor_detections = []
        raw_distractor_count = 0
        if cfg.use_distractor_branch:
            t0 = time.perf_counter()
            distractor_detections = self.dino.detect(
                image_hwc_float,
                prompt_text=cfg.distractor_prompt_text,
                box_threshold=cfg.dino_box_threshold,
                text_threshold=cfg.dino_text_threshold,
            )
            distractor_dino_ms = (time.perf_counter() - t0) * 1000.0
            dino_ms += distractor_dino_ms
            raw_distractor_count = len(distractor_detections)
            distractor_detections = filter_detections_by_geometry(
                distractor_detections,
                img_h,
                img_w,
                max_area_ratio=cfg.dino_max_area_ratio,
                max_side_ratio=cfg.dino_max_side_ratio,
            )

        if cfg.reject_ship_distractor_conflicts:
            target_detections = filter_detections_by_ship_distractor_conflict(
                target_detections,
                distractor_detections,
                img_h,
                img_w,
                max_ship_iou=cfg.ship_conflict_max_iou,
                max_ship_near=cfg.ship_conflict_max_near,
                min_ship_score=cfg.distractor_min_score,
                yellow_override_ratio=cfg.ship_conflict_yellow_override_ratio,
            )

        distractor_iou_penalty = cfg.distractor_iou_penalty if cfg.distractor_mode == "penalty" else 0.0
        distractor_near_penalty = cfg.distractor_near_penalty if cfg.distractor_mode == "penalty" else 0.0
        detections = rank_target_detections(
            target_detections,
            distractor_detections,
            img_h,
            img_w,
            distractor_penalty=distractor_iou_penalty,
            nearby_penalty=distractor_near_penalty,
            nearby_dist_norm=cfg.distractor_near_dist,
            min_distractor_score=cfg.distractor_min_score,
        )

        best_det = detections[0] if len(detections) > 0 else None
        second_det = detections[1] if len(detections) > 1 else None
        best_score = 0.0 if best_det is None else float(best_det["final_score"])
        second_score = 0.0 if second_det is None else float(second_det["final_score"])
        score_margin = best_score - second_score
        dino_magenta_ratio = 0.0 if best_det is None else float(best_det.get("magenta_ratio", 0.0))
        dino_yellow_ratio = 0.0 if best_det is None else float(best_det.get("yellow_ratio", 0.0))

        dino_visible_before_stage2 = bool(best_score >= cfg.dino_visible_score)
        visible = dino_visible_before_stage2
        visible_source = "dino" if visible else "none"

        stage2_results = []
        stage2_match = None
        stage2_checked = False
        stage2_pass = None
        stage2_match_iou = 0.0
        stage2_margin = 0.0
        stage2_target_score = 0.0
        stage2_negative_score = 0.0
        stage2_target_prompt = None
        stage2_negative_prompt = None

        if self.stage2 is not None:
            stage2_candidates = make_stage2_candidates(
                detections,
                distractor_detections,
                top_k_target=cfg.stage2_top_k_target,
                top_k_distractor=cfg.stage2_top_k_distractor,
            )
            t0 = time.perf_counter()
            stage2_results = self.stage2.rerank(image_hwc_float, stage2_candidates)
            clip_ms = (time.perf_counter() - t0) * 1000.0

        if cfg.stage2_gate_visible and self.stage2 is not None and dino_visible_before_stage2 and best_det is not None:
            stage2_checked = True
            stage2_match, stage2_match_iou = find_stage2_match_for_box(
                stage2_results,
                best_det["box"],
                min_iou=cfg.stage2_gate_min_iou,
            )
            if stage2_match is not None:
                stage2_margin = float(stage2_match["clip_margin"])
                stage2_target_score = float(stage2_match["clip_target_score"])
                stage2_negative_score = float(stage2_match["clip_negative_score"])
                stage2_target_prompt = stage2_match["clip_target_prompt"]
                stage2_negative_prompt = stage2_match["clip_negative_prompt"]
                stage2_pass = bool(stage2_margin >= cfg.stage2_gate_margin_threshold)
                if (
                    not stage2_pass
                    and cfg.stage2_yellow_override
                    and cfg.task_name == "yellow_tugboat"
                    and task3_yellow_tugboat_override(
                        best_det,
                        stage2_margin,
                        min_yellow_ratio=cfg.stage2_yellow_override_min_ratio,
                        min_score=cfg.stage2_yellow_override_min_score,
                        min_margin=cfg.stage2_yellow_override_min_margin,
                    )
                ):
                    stage2_pass = True
            else:
                stage2_pass = False
            visible = bool(visible and stage2_pass)
            visible_source = "dino_stage2" if visible else "none"

        fallback_visible = False
        fallback_magenta_ratio = 0.0
        fallback_green_ratio = 0.0
        fallback_box = None
        if cfg.use_magenta_fallback and not visible:
            fallback_box, roi_used_fallback, _, _ = find_ball_roi(image_hwc_float)
            if fallback_box is not None and not roi_used_fallback:
                fallback_magenta_ratio = box_magenta_ratio(image_hwc_float, fallback_box)
                fallback_visible = bool(fallback_magenta_ratio >= cfg.fallback_min_magenta_ratio)
                if fallback_visible:
                    visible = True
                    visible_source = "magenta_fallback"

        if cfg.use_green_fallback and not visible:
            fallback_box, roi_used_fallback, _, _ = find_green_roi(image_hwc_float)
            if fallback_box is not None and not roi_used_fallback:
                fallback_green_ratio = box_green_ratio(image_hwc_float, fallback_box)
                fallback_visible = bool(fallback_green_ratio >= cfg.fallback_min_green_ratio)
                if fallback_visible:
                    visible = True
                    visible_source = "green_fallback"

        final_box = None
        final_score = 0.0
        final_margin = 0.0
        if visible_source == "magenta_fallback":
            final_box = fallback_box
            final_score = fallback_magenta_ratio
            final_margin = fallback_magenta_ratio
        elif visible_source == "green_fallback":
            final_box = fallback_box
            final_score = fallback_green_ratio
            final_margin = fallback_green_ratio
        elif visible and best_det is not None:
            final_box = best_det["box"]
            final_score = best_score
            final_margin = stage2_margin

        state, center_xy, center_norm_xy, direction = _build_harbor_state(
            visible=visible,
            box=final_box,
            img_h=img_h,
            img_w=img_w,
            score=final_score,
            age_norm=0.0,
        )
        total_ms = (time.perf_counter() - profile_t0) * 1000.0
        self.last_profile = {
            "dino_ms": float(dino_ms),
            "dino_target_ms": float(target_dino_ms),
            "dino_distractor_ms": float(distractor_dino_ms),
            "clip_ms": float(clip_ms),
            "filter_state_ms": float(max(0.0, total_ms - dino_ms - clip_ms)),
            "localizer_internal_ms": float(total_ms),
            "stage2_candidate_count": float(len(stage2_candidates) if self.stage2 is not None else 0),
            "raw_target_count": float(raw_target_count),
            "raw_distractor_count": float(raw_distractor_count),
        }

        return HarborTargetLocalizationResult(
            visible=visible,
            state=state,
            box=None if final_box is None else tuple(float(v) for v in final_box),
            center_xy=center_xy,
            center_norm_xy=center_norm_xy,
            direction=direction,
            visible_source=visible_source,
            cache_age_steps=0,
            dino_visible_before_stage2=dino_visible_before_stage2,
            dino_label=None if best_det is None else best_det.get("label"),
            dino_score=best_score,
            dino_second_score=second_score,
            dino_score_margin=score_margin,
            dino_magenta_ratio=dino_magenta_ratio,
            dino_yellow_ratio=dino_yellow_ratio,
            raw_target_count=raw_target_count,
            target_count=len(target_detections),
            raw_distractor_count=raw_distractor_count,
            distractor_count=len(distractor_detections),
            fallback_visible=fallback_visible,
            fallback_magenta_ratio=fallback_magenta_ratio,
            stage2_checked=stage2_checked,
            stage2_pass=stage2_pass,
            stage2_margin=stage2_margin,
            stage2_match_iou=stage2_match_iou,
            stage2_target_score=stage2_target_score,
            stage2_negative_score=stage2_negative_score,
            stage2_target_prompt=stage2_target_prompt,
            stage2_negative_prompt=stage2_negative_prompt,
            debug={
                "target_detections": detections,
                "distractor_detections": distractor_detections,
                "stage2_results": stage2_results,
                "stage2_match": stage2_match,
                "fallback_green_ratio": fallback_green_ratio,
            },
        )

    @staticmethod
    def augment_vector(vector: np.ndarray, result: HarborTargetLocalizationResult) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return np.concatenate([vector, result.state.astype(np.float32)], axis=0)

    @staticmethod
    def zero_result(age_norm: float = 1.0) -> HarborTargetLocalizationResult:
        state = np.zeros((TARGET_STATE_DIM,), dtype=np.float32)
        state[7] = float(age_norm)
        return HarborTargetLocalizationResult(
            visible=False,
            state=state,
            box=None,
            center_xy=(0.0, 0.0),
            center_norm_xy=(0.0, 0.0),
            direction="none",
            visible_source="zero",
            cache_age_steps=0,
            dino_visible_before_stage2=False,
            dino_label=None,
            dino_score=0.0,
            dino_second_score=0.0,
            dino_score_margin=0.0,
            dino_magenta_ratio=0.0,
            dino_yellow_ratio=0.0,
            raw_target_count=0,
            target_count=0,
            raw_distractor_count=0,
            distractor_count=0,
            fallback_visible=False,
            fallback_magenta_ratio=0.0,
            stage2_checked=False,
            stage2_pass=None,
            stage2_margin=0.0,
            stage2_match_iou=0.0,
            stage2_target_score=0.0,
            stage2_negative_score=0.0,
            stage2_target_prompt=None,
            stage2_negative_prompt=None,
        )

    @staticmethod
    def with_cache_age(result: HarborTargetLocalizationResult, age_steps: int, max_age_steps: int) -> HarborTargetLocalizationResult:
        age_norm = min(float(age_steps) / max(float(max_age_steps), 1.0), 1.0)
        state = result.state.astype(np.float32).copy()
        state[7] = age_norm
        return replace(
            result,
            state=state,
            cache_age_steps=int(age_steps),
            visible_source="cached_" + result.visible_source,
        )


def _join_prompt_text(prompts: Sequence[str]) -> str:
    text = ". ".join(prompts)
    if not text.endswith("."):
        text += "."
    return text


def build_harbor_task_config(
    task_name: str,
    device: str = "cuda",
    dino_visible_score: float = 0.28,
    stage2_gate_margin_threshold: float = 0.0,
    fallback_min_magenta_ratio: float = 0.05,
) -> HarborTargetLocalizerConfig:
    task = (task_name or "magenta_ball").strip().lower()
    task = task.replace("-", "_")

    if task in ("magenta", "magenta_ball", "purple_ball", "task1"):
        return HarborTargetLocalizerConfig(
            task_name="magenta_ball",
            device=device,
            dino_visible_score=dino_visible_score,
            stage2_gate_margin_threshold=stage2_gate_margin_threshold,
            fallback_min_magenta_ratio=fallback_min_magenta_ratio,
            target_prompts=HARBOR_TARGET_PROMPTS,
            distractor_prompts=HARBOR_DISTRACTOR_PROMPTS,
            stage2_negative_prompts=HARBOR_STAGE2_NEGATIVE_PROMPTS,
            use_ball_geometry_filter=True,
            use_magenta_filter=True,
            use_magenta_fallback=True,
        )

    if task in ("green", "green_buoy", "task2"):
        return HarborTargetLocalizerConfig(
            task_name="green_buoy",
            device=device,
            dino_visible_score=0.24 if dino_visible_score == 0.28 else dino_visible_score,
            stage2_gate_margin_threshold=stage2_gate_margin_threshold,
            fallback_min_magenta_ratio=fallback_min_magenta_ratio,
            target_prompts=HARBOR_GREEN_BUOY_TARGET_PROMPTS,
            distractor_prompts=HARBOR_GREEN_BUOY_DISTRACTOR_PROMPTS,
            stage2_target_prompts=HARBOR_GREEN_BUOY_STAGE2_TARGET_PROMPTS,
            stage2_negative_prompts=HARBOR_GREEN_BUOY_STAGE2_NEGATIVE_PROMPTS,
            use_ball_geometry_filter=False,
            use_magenta_filter=False,
            use_magenta_fallback=False,
            use_green_fallback=True,
            fallback_min_green_ratio=0.02,
            dino_max_area_ratio=0.35,
            dino_max_side_ratio=0.75,
            stage2_crop_scale=1.45,
        )

    if task in ("yellow_tugboat", "tugboat", "vessel03", "vessl03", "task3", "yellow_tugboat_marker", "tugboat_marker", "task3_marker"):
        use_marker = task in ("yellow_tugboat_marker", "tugboat_marker", "task3_marker")
        return HarborTargetLocalizerConfig(
            task_name="yellow_tugboat_marker" if use_marker else "yellow_tugboat",
            device=device,
            dino_visible_score=(0.20 if use_marker else 0.22) if dino_visible_score == 0.28 else dino_visible_score,
            stage2_gate_margin_threshold=(0.0 if use_marker else 0.005) if stage2_gate_margin_threshold == 0.0 else stage2_gate_margin_threshold,
            fallback_min_magenta_ratio=fallback_min_magenta_ratio,
            target_prompts=HARBOR_TUGBOAT_MARKER_TARGET_PROMPTS if use_marker else HARBOR_YELLOW_TUGBOAT_TARGET_PROMPTS,
            distractor_prompts=HARBOR_YELLOW_TUGBOAT_DISTRACTOR_PROMPTS + (["yellow tugboat", "yellow harbor tugboat"] if use_marker else []),
            stage2_target_prompts=HARBOR_TUGBOAT_MARKER_STAGE2_TARGET_PROMPTS if use_marker else HARBOR_YELLOW_TUGBOAT_STAGE2_TARGET_PROMPTS,
            stage2_negative_prompts=HARBOR_YELLOW_TUGBOAT_STAGE2_NEGATIVE_PROMPTS,
            use_ball_geometry_filter=False,
            use_magenta_filter=False,
            use_yellow_filter=False if use_marker else True,
            min_yellow_ratio=0.0 if use_marker else 0.018,
            use_green_fallback=True if use_marker else False,
            fallback_min_green_ratio=0.005 if use_marker else 0.03,
            reject_ship_distractor_conflicts=False if use_marker else True,
            ship_conflict_max_iou=0.45,
            ship_conflict_max_near=0.92,
            ship_conflict_yellow_override_ratio=0.08,
            stage2_yellow_override=False if use_marker else True,
            stage2_yellow_override_min_ratio=0.09,
            stage2_yellow_override_min_score=0.26,
            stage2_yellow_override_min_margin=-0.02,
            use_magenta_fallback=False,
            dino_max_area_ratio=0.18 if use_marker else 0.22,
            dino_max_side_ratio=0.55 if use_marker else 0.60,
            stage2_crop_scale=1.55 if use_marker else 1.25,
        )

    raise ValueError(
        "Unknown HARBOR_TASK. Use one of: magenta_ball, green_buoy, yellow_tugboat."
    )


def _build_harbor_state(
    visible: bool,
    box: Optional[Tuple[float, float, float, float]],
    img_h: int,
    img_w: int,
    score: float,
    age_norm: float,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], str]:
    state = np.zeros((TARGET_STATE_DIM,), dtype=np.float32)
    state[7] = float(age_norm)
    if not visible or box is None:
        return state, (0.0, 0.0), (0.0, 0.0), "none"

    x1, y1, x2, y2 = box
    cx = 0.5 * (float(x1) + float(x2))
    cy = 0.5 * (float(y1) + float(y2))
    x_norm = cx / max(float(img_w), 1.0)
    y_norm = cy / max(float(img_h), 1.0)

    direction = "center"
    dir_left = 0.0
    dir_center = 0.0
    dir_right = 0.0
    if cx < img_w / 3:
        dir_left = 1.0
        direction = "left"
    elif cx < 2 * img_w / 3:
        dir_center = 1.0
        direction = "center"
    else:
        dir_right = 1.0
        direction = "right"

    state[:] = np.array(
        [
            1.0,
            x_norm,
            y_norm,
            dir_left,
            dir_center,
            dir_right,
            float(score),
            float(age_norm),
        ],
        dtype=np.float32,
    )
    return state, (cx, cy), (x_norm, y_norm), direction
