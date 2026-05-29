"""Native CIF-to-experiment calibration for Diffraction Analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from fourdlab.processing.native_templates import (
    NativeCrystal,
    generate_cif_zone_preview,
)
from fourdlab.processing.peak_detection import PeakList


@dataclass(frozen=True)
class CifCalibrationConfig:
    qx_center: float
    qy_center: float
    q_pixel_size: float
    flip_qy: bool = True
    detector_rotation_deg: float = 0.0
    k_max: float = 2.0
    min_q: float = 0.0
    center_exclusion_q: float = 0.0
    match_tolerance_q: float = 0.04
    zone_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    sigma_excitation: float = 0.03
    intensity_weight_mode: Literal["uniform", "raw", "sqrt", "log"] = "sqrt"


@dataclass
class RadialCalibrationResult:
    q_pixels: np.ndarray
    experimental_profile: np.ndarray
    cif_q: np.ndarray
    cif_profile: np.ndarray
    candidate_pixel_sizes: np.ndarray
    scores: np.ndarray
    best_q_pixel_size: float
    top_pixel_sizes: np.ndarray
    top_scores: np.ndarray


@dataclass
class VectorCalibrationResult:
    qx_center: float
    qy_center: float
    q_pixel_size: float
    flip_qy: bool
    detector_rotation_deg: float
    zone_axis: tuple[float, float, float]
    matched_peak_count: int
    mean_residual_q: float
    median_residual_q: float
    recommended_match_tolerance_q: float
    quality: Literal["good", "warning", "bad"]
    suggestions: list[str]


def peak_list_to_calibrated_qxy(
    peaks: PeakList,
    config: CifCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert detected peak pixels into calibrated reciprocal qxy vectors."""

    qx = np.asarray(peaks.qx, dtype=np.float64) - float(config.qx_center)
    qy = np.asarray(peaks.qy, dtype=np.float64) - float(config.qy_center)

    if config.flip_qy:
        qy *= -1.0

    qx = qx * float(config.q_pixel_size)
    qy = qy * float(config.q_pixel_size)

    theta = np.deg2rad(float(config.detector_rotation_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    qx_rot = cos_t * qx - sin_t * qy
    qy_rot = sin_t * qx + cos_t * qy

    qxy = np.column_stack([qx_rot, qy_rot])
    intensity = np.asarray(peaks.intensity, dtype=np.float64)

    radius = np.linalg.norm(qxy, axis=1)
    keep = (
        (radius >= max(float(config.min_q), float(config.center_exclusion_q)))
        & (radius <= float(config.k_max))
        & np.isfinite(intensity)
        & (intensity > 0)
    )

    return qxy[keep], intensity[keep]


def run_radial_cif_calibration(
    radial_source,
    native_crystal: NativeCrystal,
    config: CifCalibrationConfig,
    pixel_min: float,
    pixel_max: float,
    steps: int,
) -> RadialCalibrationResult:
    """Coarsely calibrate q-pixel size by radial profile agreement."""

    q_pixels, exp_profile = _radial_source_to_profile(radial_source)
    weighted_exp = _normalize_1d(np.asarray(exp_profile, dtype=np.float64) * q_pixels)
    preview = generate_cif_zone_preview(
        native_crystal,
        config.k_max,
        zone_axis=config.zone_axis,
        sigma_excitation=config.sigma_excitation,
        intensity_weight_mode=config.intensity_weight_mode,
    )
    candidate_pixel_sizes = np.linspace(float(pixel_min), float(pixel_max), max(2, int(steps)))
    scores = np.zeros_like(candidate_pixel_sizes)
    for idx, pixel_size in enumerate(candidate_pixel_sizes):
        cif_at_exp = np.interp(
            q_pixels * float(pixel_size),
            preview.q_profile,
            preview.intensity_profile,
            left=0.0,
            right=0.0,
        )
        scores[idx] = float(np.dot(weighted_exp, _normalize_1d(cif_at_exp)))
    order = np.argsort(scores)[::-1]
    top = order[: min(5, order.size)]
    best = int(order[0])
    return RadialCalibrationResult(
        q_pixels=q_pixels,
        experimental_profile=np.asarray(exp_profile, dtype=np.float64),
        cif_q=preview.q_profile,
        cif_profile=preview.intensity_profile,
        candidate_pixel_sizes=candidate_pixel_sizes,
        scores=scores,
        best_q_pixel_size=float(candidate_pixel_sizes[best]),
        top_pixel_sizes=candidate_pixel_sizes[top],
        top_scores=scores[top],
    )


def run_vector_cif_calibration(
    peaks: PeakList | list[PeakList],
    native_crystal: NativeCrystal,
    initial_config: CifCalibrationConfig,
    *,
    search_q_center_px: float = 2.0,
    search_q_pixel_fraction: float = 0.1,
    search_rotation_deg: float = 5.0,
    try_flip_qy: bool = True,
) -> VectorCalibrationResult:
    """Refine center, q-pixel size, detector rotation, and qy flip."""

    peak_list = _merge_peak_lists(peaks) if isinstance(peaks, list) else peaks
    preview = generate_cif_zone_preview(
        native_crystal,
        initial_config.k_max,
        zone_axis=initial_config.zone_axis,
        sigma_excitation=initial_config.sigma_excitation,
        intensity_weight_mode=initial_config.intensity_weight_mode,
    )
    template_qxy = np.column_stack((preview.qx, preview.qy))
    if template_qxy.shape[0] == 0:
        raise ValueError("CIF template has no usable peaks for the selected zone axis.")

    center_offsets = np.linspace(-float(search_q_center_px), float(search_q_center_px), 5)
    pixel_values = float(initial_config.q_pixel_size) * (
        1.0 + np.linspace(-float(search_q_pixel_fraction), float(search_q_pixel_fraction), 5)
    )
    pixel_values = pixel_values[pixel_values > 0]
    rotations = np.linspace(-float(search_rotation_deg), float(search_rotation_deg), 7)
    flips = [bool(initial_config.flip_qy)]
    if try_flip_qy:
        flips = [True, False]

    best_score = -np.inf
    best_config = initial_config
    best_residuals = np.zeros(0, dtype=np.float64)
    best_count = 0
    best_median = np.inf
    tolerance = max(float(initial_config.match_tolerance_q), 1.0e-6)

    for flip in flips:
        for dx in center_offsets:
            for dy in center_offsets:
                for pixel_size in pixel_values:
                    pixel_penalty = abs(float(pixel_size) - float(initial_config.q_pixel_size)) / max(
                        float(initial_config.q_pixel_size),
                        1.0e-12,
                    )
                    for rotation in rotations:
                        candidate = CifCalibrationConfig(
                            qx_center=float(initial_config.qx_center) + float(dx),
                            qy_center=float(initial_config.qy_center) + float(dy),
                            q_pixel_size=float(pixel_size),
                            flip_qy=flip,
                            detector_rotation_deg=float(initial_config.detector_rotation_deg) + float(rotation),
                            k_max=initial_config.k_max,
                            min_q=initial_config.min_q,
                            center_exclusion_q=initial_config.center_exclusion_q,
                            match_tolerance_q=initial_config.match_tolerance_q,
                            zone_axis=initial_config.zone_axis,
                            sigma_excitation=initial_config.sigma_excitation,
                            intensity_weight_mode=initial_config.intensity_weight_mode,
                        )
                        exp_qxy, _intensity = peak_list_to_calibrated_qxy(peak_list, candidate)
                        count, residuals = _match_template_to_experiment(template_qxy, exp_qxy, tolerance)
                        median = float(np.median(residuals)) if residuals.size else np.inf
                        mean = float(np.mean(residuals)) if residuals.size else np.inf
                        score = count - 0.35 * median / tolerance - 0.2 * mean / tolerance - 0.5 * pixel_penalty
                        if score > best_score:
                            best_score = float(score)
                            best_config = candidate
                            best_residuals = residuals
                            best_count = int(count)
                            best_median = median

    mean_residual = float(np.mean(best_residuals)) if best_residuals.size else float("nan")
    median_residual = float(best_median) if np.isfinite(best_median) else float("nan")
    recommended = max(float(initial_config.match_tolerance_q), median_residual * 1.5 if np.isfinite(median_residual) else 0.04)
    quality = _quality(best_count, median_residual, initial_config.match_tolerance_q)
    suggestions = _suggestions(
        peak_list,
        best_count,
        median_residual,
        initial_config,
        best_config,
        template_qxy.shape[0],
    )
    return VectorCalibrationResult(
        qx_center=best_config.qx_center,
        qy_center=best_config.qy_center,
        q_pixel_size=best_config.q_pixel_size,
        flip_qy=best_config.flip_qy,
        detector_rotation_deg=best_config.detector_rotation_deg,
        zone_axis=best_config.zone_axis,
        matched_peak_count=best_count,
        mean_residual_q=mean_residual,
        median_residual_q=median_residual,
        recommended_match_tolerance_q=float(recommended),
        quality=quality,
        suggestions=suggestions,
    )


def radial_profile_from_peak_list(
    peaks: PeakList,
    *,
    qx_center: float,
    qy_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a pixel-radius profile from one detected peak list."""

    if peaks.count == 0:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    dx = np.asarray(peaks.qx, dtype=np.float64) - float(qx_center)
    dy = np.asarray(peaks.qy, dtype=np.float64) - float(qy_center)
    radius = np.rint(np.sqrt(dx**2 + dy**2)).astype(np.int64)
    max_radius = int(radius.max()) if radius.size else 0
    summed = np.bincount(radius, weights=np.asarray(peaks.intensity, dtype=np.float64), minlength=max_radius + 1)
    counts = np.bincount(radius, minlength=max_radius + 1)
    return np.arange(max_radius + 1, dtype=np.float64), summed / np.maximum(counts, 1)


def radial_profile_from_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a radial profile from a centered BVM image."""

    arr = np.asarray(image, dtype=np.float64)
    cy = (arr.shape[0] - 1) / 2.0
    cx = (arr.shape[1] - 1) / 2.0
    yy, xx = np.indices(arr.shape, dtype=np.float64)
    radius = np.rint(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)).astype(np.int64)
    max_radius = int(radius.max())
    summed = np.bincount(radius.ravel(), weights=arr.ravel(), minlength=max_radius + 1)
    counts = np.bincount(radius.ravel(), minlength=max_radius + 1)
    return np.arange(max_radius + 1, dtype=np.float64), summed / np.maximum(counts, 1)


def _radial_source_to_profile(radial_source) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(radial_source, tuple):
        return (
            np.asarray(radial_source[0], dtype=np.float64),
            np.asarray(radial_source[1], dtype=np.float64),
        )
    return radial_profile_from_image(np.asarray(radial_source, dtype=np.float64))


def _match_template_to_experiment(
    template_qxy: np.ndarray,
    exp_qxy: np.ndarray,
    tolerance: float,
) -> tuple[int, np.ndarray]:
    if template_qxy.size == 0 or exp_qxy.size == 0:
        return 0, np.zeros(0, dtype=np.float64)
    tree = cKDTree(exp_qxy)
    distances, indices = tree.query(template_qxy, k=1, distance_upper_bound=float(tolerance))
    valid = np.isfinite(distances) & (indices < exp_qxy.shape[0])
    return int(np.count_nonzero(valid)), distances[valid].astype(np.float64)


def _merge_peak_lists(peaks: list[PeakList]) -> PeakList:
    if not peaks:
        return PeakList(
            qx=np.zeros(0, dtype=np.float64),
            qy=np.zeros(0, dtype=np.float64),
            intensity=np.zeros(0, dtype=np.float64),
            refined=np.zeros(0, dtype=bool),
        )
    return PeakList(
        qx=np.concatenate([p.qx for p in peaks]),
        qy=np.concatenate([p.qy for p in peaks]),
        intensity=np.concatenate([p.intensity for p in peaks]),
        refined=np.concatenate([p.refined for p in peaks]),
    )


def _quality(count: int, median: float, tolerance: float) -> Literal["good", "warning", "bad"]:
    if count >= 6 and np.isfinite(median) and median <= float(tolerance):
        return "good"
    if count >= 3 and np.isfinite(median) and median <= float(tolerance) * 2.0:
        return "warning"
    return "bad"


def _suggestions(
    peaks: PeakList,
    count: int,
    median: float,
    initial: CifCalibrationConfig,
    best: CifCalibrationConfig,
    template_count: int,
) -> list[str]:
    suggestions: list[str] = []
    if peaks.count < 8:
        suggestions.append("peak detection has too few high-q peaks")
    if count < min(6, max(3, template_count // 4)):
        suggestions.append("selected zone axis may be wrong")
        suggestions.append("try larger k_max")
    if np.isfinite(median) and median > initial.match_tolerance_q:
        suggestions.append("q pixel size may be off")
    if best.flip_qy != initial.flip_qy:
        suggestions.append("check qy flip")
    if best.center_exclusion_q <= 0 and peaks.count > 0:
        suggestions.append("increase center exclusion")
    if not suggestions:
        suggestions.append("calibration is consistent enough for native orientation matching")
    return suggestions


def _normalize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - float(np.min(arr))
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return np.zeros_like(arr)
    return arr / norm
