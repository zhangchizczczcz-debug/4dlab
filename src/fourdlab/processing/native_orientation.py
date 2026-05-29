"""Native orientation template planning and mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from fourdlab.processing.native_calibration import (
    CifCalibrationConfig,
    peak_list_to_calibrated_qxy,
)
from fourdlab.processing.native_matching import SparseMatchResult, match_sparse_template
from fourdlab.processing.native_templates import (
    NativeCrystal,
    generate_cif_zone_preview,
    rotated_zone_template,
    zone_axes_in_cone,
)
from fourdlab.processing.peak_detection import PeakDetectionResult, PeakList

ProgressCallback = Callable[[int, int], None]
StopCallback = Callable[[], bool]


class NativeOrientationCancelled(RuntimeError):
    """Raised when native orientation mapping is cancelled."""


@dataclass(frozen=True)
class NativeOrientationConfig:
    mode: Literal["in_plane", "out_of_plane"]
    calibration: CifCalibrationConfig
    angle_step_in_plane: float = 2.0
    angle_step_zone_axis: float = 2.0
    num_matches: int = 5
    min_number_peaks: int = 3
    zone_range_mode: Literal["fixed", "center_cone"] = "fixed"
    zone_axis_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    zone_angle_deg: float = 10.0


@dataclass
class NativeOrientationTemplate:
    qxy: np.ndarray
    intensity: np.ndarray
    hkl: np.ndarray | None
    matrix: np.ndarray
    zone_axis: np.ndarray
    in_plane_angle_deg: float


@dataclass
class NativeOrientationPlan:
    config: NativeOrientationConfig
    templates: list[NativeOrientationTemplate]


@dataclass
class NativeOrientationMapResult:
    score_topn: np.ndarray
    matrix_topn: np.ndarray
    zone_axis_topn: np.ndarray
    in_plane_angle_topn_deg: np.ndarray
    matched_peak_count_topn: np.ndarray
    mean_residual_topn: np.ndarray

    best_score: np.ndarray
    second_score: np.ndarray
    confidence_gap: np.ndarray
    confidence_ratio: np.ndarray
    matched_peak_count: np.ndarray
    mean_residual_q: np.ndarray
    valid_mask: np.ndarray

    best_matrix: np.ndarray
    best_zone_axis: np.ndarray
    in_plane_angle_deg: np.ndarray
    tilt_deg: np.ndarray
    tilt_azimuth_deg: np.ndarray


def build_native_orientation_plan(
    native_crystal: NativeCrystal,
    config: NativeOrientationConfig,
) -> NativeOrientationPlan:
    """Build a reusable native orientation/template bank."""

    cal = config.calibration
    zone_axes = [_normalize(np.asarray(cal.zone_axis, dtype=np.float64))]
    if config.mode == "out_of_plane" and config.zone_range_mode == "center_cone":
        zone_axes = zone_axes_in_cone(
            config.zone_axis_center,
            config.zone_angle_deg,
            config.angle_step_zone_axis,
        )

    step = max(0.1, float(config.angle_step_in_plane))
    angles = np.arange(0.0, 360.0, step, dtype=np.float64)
    if angles.size == 0:
        angles = np.asarray([0.0], dtype=np.float64)
    templates: list[NativeOrientationTemplate] = []
    for zone_axis in zone_axes:
        preview = generate_cif_zone_preview(
            native_crystal,
            cal.k_max,
            zone_axis=tuple(float(v) for v in zone_axis),
            sigma_excitation=cal.sigma_excitation,
            intensity_weight_mode=cal.intensity_weight_mode,
        )
        for angle in angles:
            qxy, matrix = rotated_zone_template(preview, float(angle))
            if qxy.shape[0] == 0:
                continue
            templates.append(
                NativeOrientationTemplate(
                    qxy=qxy,
                    intensity=preview.intensity.copy(),
                    hkl=preview.hkl.copy(),
                    matrix=matrix,
                    zone_axis=preview.zone_axis.copy(),
                    in_plane_angle_deg=float(angle),
                )
            )
    if not templates:
        raise ValueError("Native orientation plan has no templates.")
    return NativeOrientationPlan(config=config, templates=templates)


def match_native_orientation(
    plan: NativeOrientationPlan,
    peaks: PeakList,
) -> list[tuple[int, SparseMatchResult]]:
    """Return top-N native template matches for one peak list."""

    cal = plan.config.calibration
    exp_qxy, exp_intensity = peak_list_to_calibrated_qxy(peaks, cal)
    if exp_qxy.shape[0] < int(plan.config.min_number_peaks):
        return []
    matches: list[tuple[int, SparseMatchResult]] = []
    for idx, template in enumerate(plan.templates):
        match = match_sparse_template(
            template.qxy,
            template.intensity,
            exp_qxy,
            exp_intensity,
            tolerance_q=cal.match_tolerance_q,
            min_number_peaks=plan.config.min_number_peaks,
            intensity_weight_mode=cal.intensity_weight_mode,
        )
        matches.append((idx, match))
    matches.sort(key=lambda item: item[1].score, reverse=True)
    return matches[: max(1, int(plan.config.num_matches))]


def run_native_orientation_map(
    plan: NativeOrientationPlan,
    peaks: PeakDetectionResult,
    *,
    progress: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> NativeOrientationMapResult:
    """Match a native orientation plan across a full scan in scan_y, scan_x order."""

    scan_y, scan_x = peaks.scan_shape
    topn = max(1, int(plan.config.num_matches))
    score_topn = np.full((scan_y, scan_x, topn), np.nan, dtype=np.float64)
    matrix_topn = np.full((scan_y, scan_x, topn, 3, 3), np.nan, dtype=np.float64)
    zone_axis_topn = np.full((scan_y, scan_x, topn, 3), np.nan, dtype=np.float64)
    angle_topn = np.full((scan_y, scan_x, topn), np.nan, dtype=np.float64)
    count_topn = np.zeros((scan_y, scan_x, topn), dtype=np.int32)
    residual_topn = np.full((scan_y, scan_x, topn), np.nan, dtype=np.float64)

    total = scan_y * scan_x
    done = 0
    for y in range(scan_y):
        for x in range(scan_x):
            if stop_requested is not None and stop_requested():
                raise NativeOrientationCancelled(f"Native orientation cancelled at scan ({y}, {x}).")
            matches = match_native_orientation(plan, peaks.peaks[y][x])
            for rank, (template_idx, match) in enumerate(matches[:topn]):
                template = plan.templates[template_idx]
                score_topn[y, x, rank] = match.score
                matrix_topn[y, x, rank] = template.matrix
                zone_axis_topn[y, x, rank] = template.zone_axis
                angle_topn[y, x, rank] = template.in_plane_angle_deg
                count_topn[y, x, rank] = match.matched_peak_count
                residual_topn[y, x, rank] = match.mean_residual_q
            done += 1
            if progress is not None:
                progress(done, total)
            if stop_requested is not None and stop_requested():
                raise NativeOrientationCancelled(f"Native orientation cancelled after {done}/{total} patterns.")

    best_score = score_topn[:, :, 0]
    second_score = (
        score_topn[:, :, 1]
        if topn > 1
        else np.full((scan_y, scan_x), np.nan, dtype=np.float64)
    )
    confidence_gap = best_score - np.nan_to_num(second_score, nan=0.0)
    confidence_ratio = best_score / np.maximum(np.nan_to_num(second_score, nan=0.0), 1.0e-12)
    matched_peak_count = count_topn[:, :, 0]
    mean_residual_q = residual_topn[:, :, 0]
    valid_mask = (
        np.isfinite(best_score)
        & (matched_peak_count >= int(plan.config.min_number_peaks))
        & np.isfinite(mean_residual_q)
    )
    best_matrix = matrix_topn[:, :, 0]
    best_zone_axis = zone_axis_topn[:, :, 0]
    in_plane_angle = angle_topn[:, :, 0]
    tilt_deg, tilt_azimuth = _tilt_arrays(best_zone_axis, plan.config.zone_axis_center)
    tilt_deg = np.where(valid_mask, tilt_deg, np.nan)
    tilt_azimuth = np.where(valid_mask, tilt_azimuth, np.nan)

    return NativeOrientationMapResult(
        score_topn=score_topn,
        matrix_topn=matrix_topn,
        zone_axis_topn=zone_axis_topn,
        in_plane_angle_topn_deg=angle_topn,
        matched_peak_count_topn=count_topn,
        mean_residual_topn=residual_topn,
        best_score=best_score,
        second_score=second_score,
        confidence_gap=confidence_gap,
        confidence_ratio=confidence_ratio,
        matched_peak_count=matched_peak_count,
        mean_residual_q=mean_residual_q,
        valid_mask=valid_mask,
        best_matrix=best_matrix,
        best_zone_axis=best_zone_axis,
        in_plane_angle_deg=in_plane_angle,
        tilt_deg=tilt_deg,
        tilt_azimuth_deg=tilt_azimuth,
    )


def orientation_color_image_native(result: NativeOrientationMapResult, mode: str) -> np.ndarray:
    """Render a diagnostic RGB orientation image from native map fields."""

    value = _normalize_2d(result.best_score)
    if mode == "in_plane":
        hue = np.nan_to_num(np.mod(result.in_plane_angle_deg, 360.0) / 360.0)
        return _hsv_to_rgb(hue, np.ones_like(hue), value)
    zone = np.abs(np.nan_to_num(result.best_zone_axis, nan=0.0))
    denom = np.maximum(np.max(zone, axis=2, keepdims=True), 1.0e-12)
    return np.clip(zone / denom * value[:, :, None], 0.0, 1.0)


def native_display_array(result: NativeOrientationMapResult, label: str) -> np.ndarray:
    """Return one map array in scan_y, scan_x order for GUI display."""

    mapping = {
        "Best score": result.best_score,
        "Second score": result.second_score,
        "Confidence gap": result.confidence_gap,
        "Confidence ratio": result.confidence_ratio,
        "Matched peak count": result.matched_peak_count,
        "Mean residual": result.mean_residual_q,
        "In-plane angle": result.in_plane_angle_deg,
        "Tilt magnitude": result.tilt_deg,
        "Tilt azimuth": result.tilt_azimuth_deg,
        "Zone axis x": result.best_zone_axis[:, :, 0],
        "Zone axis y": result.best_zone_axis[:, :, 1],
        "Zone axis z": result.best_zone_axis[:, :, 2],
        "Valid mask": result.valid_mask.astype(np.float64),
    }
    return np.asarray(mapping.get(label, result.best_score), dtype=np.float64)


def _tilt_arrays(best_zone_axis: np.ndarray, reference: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    ref = _normalize(np.asarray(reference, dtype=np.float64))
    zone = np.asarray(best_zone_axis, dtype=np.float64)
    norm = np.linalg.norm(zone, axis=2, keepdims=True)
    zone_norm = zone / np.maximum(norm, 1.0e-12)
    cos_tilt = np.clip(np.sum(zone_norm * ref, axis=2), -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_tilt))
    projection = zone_norm - cos_tilt[:, :, None] * ref
    tilt_azimuth = np.degrees(np.arctan2(projection[:, :, 1], projection[:, :, 0]))
    return tilt_deg, tilt_azimuth


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError("Vector must be non-zero.")
    return arr / norm


def _normalize_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _hsv_to_rgb(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    h = np.mod(hue, 1.0) * 6.0
    i = np.floor(h).astype(np.int64)
    f = h - i
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    rgb = np.zeros(hue.shape + (3,), dtype=np.float64)
    choices = i % 6
    masks = [choices == n for n in range(6)]
    rgb[masks[0]] = np.stack((val, t, p), axis=-1)[masks[0]]
    rgb[masks[1]] = np.stack((q, val, p), axis=-1)[masks[1]]
    rgb[masks[2]] = np.stack((p, val, t), axis=-1)[masks[2]]
    rgb[masks[3]] = np.stack((p, q, val), axis=-1)[masks[3]]
    rgb[masks[4]] = np.stack((t, p, val), axis=-1)[masks[4]]
    rgb[masks[5]] = np.stack((val, p, q), axis=-1)[masks[5]]
    return np.clip(rgb, 0.0, 1.0)
