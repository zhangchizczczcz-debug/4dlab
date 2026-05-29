"""Native orientation template planning and mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from fourdlab.processing.native_calibration import (
    CifCalibrationConfig,
    peak_list_to_calibrated_qxy,
)
from fourdlab.processing.native_matching import (
    NativeFastMatch,
    PreparedNativeTemplateBank,
    SparseMatchResult,
    match_prepared_template_bank_cpu,
    match_sparse_template,
    prepare_native_template_bank,
)
from fourdlab.processing.native_templates import (
    NativeCrystal,
    generate_cif_zone_preview,
    rotated_zone_template,
    triangle_zone_samples,
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
    zone_range_mode: Literal["fixed", "center_cone", "three_vertices"] = "fixed"
    zone_axis_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    zone_angle_deg: float = 10.0
    zone_axis_vertices: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0))
    num_workers: int = 1
    use_gpu: bool = False
    gpu_chunk_size: int = 4096


@dataclass
class NativeOrientationTemplate:
    qxy: np.ndarray
    intensity: np.ndarray
    hkl: np.ndarray | None
    matrix: np.ndarray
    zone_axis_lattice: np.ndarray
    zone_axis_cart: np.ndarray
    zone_label: str
    zone_barycentric: np.ndarray | None
    zone_color_rgb: np.ndarray | None
    in_plane_angle_deg: float

    @property
    def zone_axis(self) -> np.ndarray:
        """Backward-compatible Cartesian zone-axis alias."""

        return self.zone_axis_cart


@dataclass
class NativeOrientationPlan:
    config: NativeOrientationConfig
    templates: list[NativeOrientationTemplate]
    prepared_bank: PreparedNativeTemplateBank
    gpu_bank: object | None = None


@dataclass
class NativeOrientationMapResult:
    template_index_topn: np.ndarray
    score_topn: np.ndarray
    matrix_topn: np.ndarray
    zone_axis_topn: np.ndarray
    zone_axis_lattice_topn: np.ndarray
    zone_axis_cart_topn: np.ndarray
    zone_barycentric_topn: np.ndarray
    zone_color_rgb_topn: np.ndarray
    zone_label_topn: np.ndarray
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
    best_zone_axis_lattice: np.ndarray
    best_zone_axis_cart: np.ndarray
    best_zone_color_rgb: np.ndarray
    best_zone_barycentric: np.ndarray
    in_plane_angle_deg: np.ndarray
    tilt_deg: np.ndarray
    tilt_azimuth_deg: np.ndarray


def build_native_orientation_plan(
    native_crystal: NativeCrystal,
    config: NativeOrientationConfig,
) -> NativeOrientationPlan:
    """Build a reusable native orientation/template bank."""

    cal = config.calibration
    zone_samples: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]] = [
        (_normalize(np.asarray(cal.zone_axis, dtype=np.float64)), None, None, _format_zone_label(cal.zone_axis))
    ]
    if config.mode == "out_of_plane" and config.zone_range_mode == "center_cone":
        zone_samples = [
            (axis, None, None, _format_zone_label(axis))
            for axis in zone_axes_in_cone(
            config.zone_axis_center,
            config.zone_angle_deg,
            config.angle_step_zone_axis,
            )
        ]
    elif config.mode == "out_of_plane" and config.zone_range_mode == "three_vertices":
        subdivisions = max(1, int(round(30.0 / max(float(config.angle_step_zone_axis), 0.25))))
        zone_samples = triangle_zone_samples(config.zone_axis_vertices, subdivisions)

    step = max(0.1, float(config.angle_step_in_plane))
    angles = np.arange(0.0, 360.0, step, dtype=np.float64)
    if angles.size == 0:
        angles = np.asarray([0.0], dtype=np.float64)
    templates: list[NativeOrientationTemplate] = []
    for zone_axis, barycentric, zone_color, zone_label in zone_samples:
        preview = generate_cif_zone_preview(
            native_crystal,
            cal.k_max,
            zone_axis=tuple(float(v) for v in zone_axis),
            sigma_excitation=cal.sigma_excitation,
            intensity_weight_mode=cal.intensity_weight_mode,
            zone_label=zone_label,
            zone_barycentric=barycentric,
            zone_color_rgb=zone_color,
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
                    zone_axis_lattice=preview.zone_axis_lattice.copy(),
                    zone_axis_cart=preview.zone_axis_cart.copy(),
                    zone_label=preview.zone_label,
                    zone_barycentric=None if preview.zone_barycentric is None else preview.zone_barycentric.copy(),
                    zone_color_rgb=None if preview.zone_color_rgb is None else preview.zone_color_rgb.copy(),
                    in_plane_angle_deg=float(angle),
                )
            )
    if not templates:
        raise ValueError("Native orientation plan has no templates.")
    prepared_bank = prepare_native_template_bank(
        templates,
        intensity_weight_mode=cal.intensity_weight_mode,
    )
    gpu_bank = None
    if config.use_gpu:
        try:
            from fourdlab.processing.native_matching_gpu import (
                native_gpu_available,
                prepare_native_template_bank_gpu,
            )

            if native_gpu_available():
                gpu_bank = prepare_native_template_bank_gpu(prepared_bank)
        except Exception:
            gpu_bank = None
    return NativeOrientationPlan(config=config, templates=templates, prepared_bank=prepared_bank, gpu_bank=gpu_bank)


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


def match_native_orientation_fast(
    plan: NativeOrientationPlan,
    peaks: PeakList,
) -> NativeFastMatch:
    """Fast top-N native matching using a prepared template bank."""

    cal = plan.config.calibration
    exp_qxy, exp_intensity = peak_list_to_calibrated_qxy(peaks, cal)
    if exp_qxy.shape[0] < int(plan.config.min_number_peaks):
        return NativeFastMatch(
            template_indices=np.full(max(1, int(plan.config.num_matches)), -1, dtype=np.int32),
            scores=np.full(max(1, int(plan.config.num_matches)), np.nan, dtype=np.float64),
            matched_peak_counts=np.zeros(max(1, int(plan.config.num_matches)), dtype=np.int32),
            mean_residual_q=np.full(max(1, int(plan.config.num_matches)), np.nan, dtype=np.float64),
            median_residual_q=np.full(max(1, int(plan.config.num_matches)), np.nan, dtype=np.float64),
        )
    if plan.config.use_gpu and plan.gpu_bank is not None:
        try:
            from fourdlab.processing.native_matching_gpu import match_prepared_template_bank_gpu

            return match_prepared_template_bank_gpu(
                plan.gpu_bank,
                exp_qxy,
                exp_intensity,
                tolerance_q=cal.match_tolerance_q,
                min_number_peaks=plan.config.min_number_peaks,
                num_matches=plan.config.num_matches,
                chunk_size=plan.config.gpu_chunk_size,
            )
        except Exception:
            pass
    return match_prepared_template_bank_cpu(
        plan.prepared_bank,
        exp_qxy,
        exp_intensity,
        tolerance_q=cal.match_tolerance_q,
        min_number_peaks=plan.config.min_number_peaks,
        num_matches=plan.config.num_matches,
        intensity_weight_mode=cal.intensity_weight_mode,
    )


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
    template_index_topn = np.full((scan_y, scan_x, topn), -1, dtype=np.int32)
    score_topn = np.full((scan_y, scan_x, topn), np.nan, dtype=np.float64)
    matrix_topn = np.full((scan_y, scan_x, topn, 3, 3), np.nan, dtype=np.float64)
    zone_axis_topn = np.full((scan_y, scan_x, topn, 3), np.nan, dtype=np.float64)
    zone_axis_lattice_topn = np.full((scan_y, scan_x, topn, 3), np.nan, dtype=np.float64)
    zone_axis_cart_topn = np.full((scan_y, scan_x, topn, 3), np.nan, dtype=np.float64)
    zone_barycentric_topn = np.full((scan_y, scan_x, topn, 3), np.nan, dtype=np.float64)
    zone_color_rgb_topn = np.full((scan_y, scan_x, topn, 3), np.nan, dtype=np.float64)
    zone_label_topn = np.full((scan_y, scan_x, topn), "", dtype="<U32")
    angle_topn = np.full((scan_y, scan_x, topn), np.nan, dtype=np.float64)
    count_topn = np.zeros((scan_y, scan_x, topn), dtype=np.int32)
    residual_topn = np.full((scan_y, scan_x, topn), np.nan, dtype=np.float64)

    total = scan_y * scan_x
    done = 0
    def run_one(y_idx: int, x_idx: int) -> tuple[int, int, NativeFastMatch]:
        if stop_requested is not None and stop_requested():
            raise NativeOrientationCancelled(f"Native orientation cancelled at scan ({y_idx}, {x_idx}).")
        return y_idx, x_idx, match_native_orientation_fast(plan, peaks.peaks[y_idx][x_idx])

    def store(y_idx: int, x_idx: int, match: NativeFastMatch) -> None:
        for rank, template_idx in enumerate(match.template_indices[:topn]):
            if int(template_idx) < 0:
                continue
            template = plan.templates[int(template_idx)]
            template_index_topn[y_idx, x_idx, rank] = int(template_idx)
            score_topn[y_idx, x_idx, rank] = match.scores[rank]
            matrix_topn[y_idx, x_idx, rank] = template.matrix
            zone_axis_topn[y_idx, x_idx, rank] = template.zone_axis_cart
            zone_axis_lattice_topn[y_idx, x_idx, rank] = template.zone_axis_lattice
            zone_axis_cart_topn[y_idx, x_idx, rank] = template.zone_axis_cart
            if template.zone_barycentric is not None:
                zone_barycentric_topn[y_idx, x_idx, rank] = template.zone_barycentric
            if template.zone_color_rgb is not None:
                zone_color_rgb_topn[y_idx, x_idx, rank] = template.zone_color_rgb
            else:
                zone_color_rgb_topn[y_idx, x_idx, rank] = _zone_axis_fallback_color(template.zone_axis_lattice)
            zone_label_topn[y_idx, x_idx, rank] = template.zone_label
            angle_topn[y_idx, x_idx, rank] = template.in_plane_angle_deg
            count_topn[y_idx, x_idx, rank] = match.matched_peak_counts[rank]
            residual_topn[y_idx, x_idx, rank] = match.mean_residual_q[rank]

    use_workers = max(1, int(plan.config.num_workers)) > 1 and not plan.config.use_gpu
    if use_workers:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        coords = ((y, x) for y in range(scan_y) for x in range(scan_x))
        max_workers = max(1, int(plan.config.num_workers))
        max_pending = max(max_workers * 3, 1)
        pending = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                if stop_requested is not None and stop_requested():
                    raise NativeOrientationCancelled(f"Native orientation cancelled after {done}/{total} patterns.")
                while len(pending) < max_pending:
                    try:
                        y, x = next(coords)
                    except StopIteration:
                        break
                    pending.add(executor.submit(run_one, y, x))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    y, x, match = future.result()
                    store(y, x, match)
                    done += 1
                    if progress is not None:
                        progress(done, total)
    else:
        for y in range(scan_y):
            for x in range(scan_x):
                if stop_requested is not None and stop_requested():
                    raise NativeOrientationCancelled(f"Native orientation cancelled at scan ({y}, {x}).")
                store(y, x, match_native_orientation_fast(plan, peaks.peaks[y][x]))
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
    best_zone_axis_lattice = zone_axis_lattice_topn[:, :, 0]
    best_zone_axis_cart = zone_axis_cart_topn[:, :, 0]
    best_zone_color_rgb = zone_color_rgb_topn[:, :, 0]
    best_zone_barycentric = zone_barycentric_topn[:, :, 0]
    in_plane_angle = angle_topn[:, :, 0]
    tilt_deg, tilt_azimuth = _tilt_arrays(best_zone_axis_cart, plan.config.zone_axis_center)
    tilt_deg = np.where(valid_mask, tilt_deg, np.nan)
    tilt_azimuth = np.where(valid_mask, tilt_azimuth, np.nan)

    return NativeOrientationMapResult(
        template_index_topn=template_index_topn,
        score_topn=score_topn,
        matrix_topn=matrix_topn,
        zone_axis_topn=zone_axis_topn,
        zone_axis_lattice_topn=zone_axis_lattice_topn,
        zone_axis_cart_topn=zone_axis_cart_topn,
        zone_barycentric_topn=zone_barycentric_topn,
        zone_color_rgb_topn=zone_color_rgb_topn,
        zone_label_topn=zone_label_topn,
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
        best_zone_axis_lattice=best_zone_axis_lattice,
        best_zone_axis_cart=best_zone_axis_cart,
        best_zone_color_rgb=best_zone_color_rgb,
        best_zone_barycentric=best_zone_barycentric,
        in_plane_angle_deg=in_plane_angle,
        tilt_deg=tilt_deg,
        tilt_azimuth_deg=tilt_azimuth,
    )


def orientation_color_image_native(
    result: NativeOrientationMapResult,
    mode: str,
    *,
    symmetry_order: int = 1,
    angle_sign: float = 1.0,
    zero_angle_offset_deg: float = 0.0,
) -> np.ndarray:
    """Render a diagnostic RGB orientation image from native map fields."""

    value = _normalize_2d(result.best_score)
    if mode == "in_plane":
        rgb = in_plane_angle_to_rgb(
            result.in_plane_angle_deg,
            symmetry_order=symmetry_order,
            angle_sign=angle_sign,
            zero_angle_offset_deg=zero_angle_offset_deg,
        )
        return np.clip(rgb * value[:, :, None], 0.0, 1.0)
    rgb = np.asarray(result.best_zone_color_rgb, dtype=np.float64).copy()
    rgb = np.nan_to_num(rgb, nan=0.55)
    rgb = np.where(result.valid_mask[:, :, None], rgb, 0.55)
    brightness = np.where(result.valid_mask, np.clip(value, 0.18, 1.0), 1.0)
    return np.clip(rgb * brightness[:, :, None], 0.0, 1.0)


def in_plane_angle_to_rgb(
    angle_deg: np.ndarray | float,
    *,
    symmetry_order: int = 1,
    angle_sign: float = 1.0,
    zero_angle_offset_deg: float = 0.0,
) -> np.ndarray:
    """Map CIF-positive in-plane angles to RGB with one shared convention."""

    symmetry = max(1, int(symmetry_order))
    period = 360.0 / float(symmetry)
    angle = float(angle_sign) * np.asarray(angle_deg, dtype=np.float64) + float(zero_angle_offset_deg)
    hue = np.nan_to_num(np.mod(angle, period) / period)
    return _hsv_to_rgb(hue, np.ones_like(hue), np.ones_like(hue))


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
        "Zone lattice h": result.best_zone_axis_lattice[:, :, 0],
        "Zone lattice k": result.best_zone_axis_lattice[:, :, 1],
        "Zone lattice l": result.best_zone_axis_lattice[:, :, 2],
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


def _zone_axis_fallback_color(axis: np.ndarray) -> np.ndarray:
    arr = np.asarray(axis, dtype=np.float64)
    arr = arr - float(np.min(arr))
    peak = float(np.max(arr))
    if peak <= 0:
        return np.asarray([0.55, 0.55, 0.55], dtype=np.float64)
    return np.clip(arr / peak, 0.0, 1.0)


def _format_zone_label(values) -> str:
    parts = []
    for value in np.asarray(values, dtype=np.float64):
        if abs(value - round(value)) < 1.0e-6:
            parts.append(str(int(round(value))))
        else:
            parts.append(f"{value:.2g}")
    return "[" + " ".join(parts) + "]"


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
