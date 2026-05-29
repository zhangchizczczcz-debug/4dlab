"""py4DSTEM-backed orientation analysis for nanobeam diffraction data."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy.spatial import cKDTree

from fourdlab.processing.peak_detection import PeakDetectionResult, PeakList

ProgressCallback = Callable[[int, int], None]
StopCallback = Callable[[], bool]
OrientationMode = Literal["in_plane", "out_of_plane"]
ZoneRangeMode = Literal["auto", "center_cone", "three_vertices"]


class OrientationCancelled(RuntimeError):
    """Raised when full-scan orientation matching is cancelled by the caller."""


@dataclass(frozen=True)
class OrientationConfig:
    """Parameters for py4DSTEM ACOM orientation matching."""

    cif_path: Path
    mode: OrientationMode
    q_pixel_size: float
    qx_center: float
    qy_center: float
    flip_qy: bool = True
    k_max: float = 2.0
    accel_voltage: float = 300_000.0
    corr_kernel_size: float = 0.08
    sigma_excitation_error: float = 0.02
    angle_step_zone_axis: float = 2.0
    angle_step_in_plane: float = 2.0
    num_matches: int = 1
    min_number_peaks: int = 3
    use_cuda: bool = False
    num_workers: int = 1
    fiber_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    zone_range_mode: ZoneRangeMode = "auto"
    zone_axis_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    zone_angle_deg: float = 10.0
    zone_axis_vertices: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0))


@dataclass
class OrientationWorkspace:
    """Prepared py4DSTEM crystal and orientation plan."""

    crystal: object
    config: OrientationConfig


@dataclass
class OrientationResult:
    """Compact orientation-map outputs used by the 4DLAB GUI."""

    corr: np.ndarray
    angle_0_deg: np.ndarray
    angle_1_deg: np.ndarray
    in_plane_angle_deg: np.ndarray
    matrix: np.ndarray
    raw: object


@dataclass
class PixelSizeScanResult:
    """Radial profile and structure-factor agreement over q pixel sizes."""

    q_pixels: np.ndarray
    weighted_intensity: np.ndarray
    q_sf: np.ndarray
    intensity_sf: np.ndarray
    pixel_sizes: np.ndarray
    scores: np.ndarray
    best_pixel_size: float


@dataclass
class CifZonePreview:
    """py4DSTEM-generated CIF diffraction spots and radial profile."""

    qx: np.ndarray
    qy: np.ndarray
    q_radius: np.ndarray
    intensity: np.ndarray
    hkl: np.ndarray
    q_profile: np.ndarray
    intensity_profile: np.ndarray


@dataclass
class CifFitDiagnostic:
    """Nearest-neighbor diagnostic between experimental and CIF Bragg peaks."""

    exp_qx: np.ndarray
    exp_qy: np.ndarray
    sim_qx: np.ndarray
    sim_qy: np.ndarray
    matched_exp_indices: np.ndarray
    matched_sim_indices: np.ndarray
    residuals: np.ndarray
    matched_count: int
    mean_residual: float
    median_residual: float
    tolerance: float
    suggestions: list[str]


@dataclass
class NoCifTemplate:
    """Experimental in-plane template built from one detected diffraction pattern."""

    vectors_px: np.ndarray
    intensity: np.ndarray
    source_scan: tuple[int, int]
    center_exclusion_px: float


@dataclass
class PreparedNoCifMatcher:
    """Precomputed no-CIF rotation templates for fast map matching."""

    angles_deg: np.ndarray
    rotated_vectors_px: np.ndarray
    template_weight: np.ndarray
    tolerance_px: float
    min_number_peaks: int
    center_exclusion_px: float


@dataclass(frozen=True)
class NoCifInPlaneConfig:
    """Parameters for experimental-template in-plane rotation matching."""

    qx_center: float
    qy_center: float
    flip_qy: bool = True
    angle_step_deg: float = 2.0
    symmetry_order: int = 1
    match_tolerance_px: float = 3.0
    min_number_peaks: int = 3
    center_exclusion_px: float = 2.0
    num_workers: int = 1


def build_orientation_workspace(config: OrientationConfig) -> OrientationWorkspace:
    """Load a CIF, calculate structure factors, and build an orientation plan."""

    if config.q_pixel_size <= 0:
        raise ValueError("q_pixel_size must be positive.")

    crystal = load_crystal_structure(config.cif_path, config.k_max)
    crystal.orientation_plan(
        zone_axis_range=_zone_axis_range(config),
        angle_step_zone_axis=float(config.angle_step_zone_axis),
        angle_step_in_plane=float(config.angle_step_in_plane),
        accel_voltage=float(config.accel_voltage),
        corr_kernel_size=float(config.corr_kernel_size),
        sigma_excitation_error=float(config.sigma_excitation_error),
        fiber_axis=_fiber_axis(config),
        fiber_angles=_fiber_angles(config),
        CUDA=bool(config.use_cuda),
        progress_bar=False,
    )
    return OrientationWorkspace(crystal=crystal, config=config)


def build_no_cif_template(
    peaks: PeakList,
    config: NoCifInPlaneConfig,
    *,
    source_scan: tuple[int, int] = (0, 0),
) -> NoCifTemplate:
    """Build an experimental in-plane template from detected peaks."""

    vectors, intensity = peak_list_to_centered_vectors_px(peaks, config)
    radius = np.linalg.norm(vectors, axis=1) if vectors.size else np.zeros(0)
    keep = radius >= float(config.center_exclusion_px)
    vectors = vectors[keep]
    intensity = intensity[keep]
    if vectors.shape[0] < int(config.min_number_peaks):
        raise ValueError(
            f"Template has {vectors.shape[0]} usable peaks; "
            f"need at least {config.min_number_peaks}."
        )
    order = np.argsort(np.linalg.norm(vectors, axis=1))
    return NoCifTemplate(
        vectors_px=vectors[order],
        intensity=intensity[order],
        source_scan=source_scan,
        center_exclusion_px=float(config.center_exclusion_px),
    )


def match_no_cif_in_plane(
    template: NoCifTemplate,
    peaks: PeakList,
    config: NoCifInPlaneConfig,
) -> tuple[float, float]:
    """Match one pattern against an experimental template over in-plane rotations."""

    return _match_no_cif_prepared(prepare_no_cif_matcher(template, config), peaks, config)


def prepare_no_cif_matcher(
    template: NoCifTemplate,
    config: NoCifInPlaneConfig,
) -> PreparedNoCifMatcher:
    """Precompute all rotated no-CIF templates for repeated map matching."""

    angle_period = 360.0 / max(1, int(config.symmetry_order))
    step = max(0.05, float(config.angle_step_deg))
    angles = np.arange(0.0, angle_period, step, dtype=np.float64)
    if angles.size == 0:
        angles = np.asarray([0.0], dtype=np.float64)
    rotated = np.stack([rotate_vectors(template.vectors_px, angle) for angle in angles], axis=0)
    return PreparedNoCifMatcher(
        angles_deg=angles,
        rotated_vectors_px=rotated,
        template_weight=_normalize_weights(template.intensity),
        tolerance_px=max(1.0e-6, float(config.match_tolerance_px)),
        min_number_peaks=int(config.min_number_peaks),
        center_exclusion_px=float(config.center_exclusion_px),
    )


def _match_no_cif_prepared(
    matcher: PreparedNoCifMatcher,
    peaks: PeakList,
    config: NoCifInPlaneConfig,
) -> tuple[float, float]:
    """Match one pattern using a precomputed bank of rotated templates."""

    vectors, intensity = peak_list_to_centered_vectors_px(peaks, config)
    radius = np.linalg.norm(vectors, axis=1) if vectors.size else np.zeros(0)
    keep = radius >= matcher.center_exclusion_px
    vectors = vectors[keep]
    intensity = intensity[keep]
    if vectors.shape[0] < matcher.min_number_peaks:
        return np.nan, 0.0

    tree = cKDTree(vectors)
    query_weight = _normalize_weights(intensity)
    angle_count, peak_count = matcher.rotated_vectors_px.shape[:2]
    flat_vectors = matcher.rotated_vectors_px.reshape(angle_count * peak_count, 2)
    distances, indices = tree.query(
        flat_vectors,
        k=1,
        distance_upper_bound=matcher.tolerance_px,
    )
    distances = distances.reshape(angle_count, peak_count)
    indices = indices.reshape(angle_count, peak_count)
    valid = np.isfinite(distances) & (indices < vectors.shape[0])
    if not np.any(valid):
        return float(matcher.angles_deg[0]), 0.0

    closeness = np.zeros_like(distances, dtype=np.float64)
    closeness[valid] = np.exp(
        -(distances[valid] ** 2) / (2.0 * matcher.tolerance_px**2)
    )
    query_terms = np.zeros_like(distances, dtype=np.float64)
    query_terms[valid] = query_weight[indices[valid]]
    scores = np.sum(
        closeness * matcher.template_weight[np.newaxis, :] * query_terms,
        axis=1,
    )
    best_idx = int(np.argmax(scores))

    return float(matcher.angles_deg[best_idx]), max(0.0, float(scores[best_idx]))


def match_no_cif_orientation_map(
    template: NoCifTemplate,
    peaks: PeakDetectionResult,
    config: NoCifInPlaneConfig,
    *,
    progress: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> OrientationResult:
    """Match no-CIF in-plane rotations over a full detected-peak scan."""

    scan_y, scan_x = peaks.scan_shape
    corr = np.zeros((scan_y, scan_x), dtype=np.float64)
    angle = np.full((scan_y, scan_x), np.nan, dtype=np.float64)
    matrix = np.zeros((scan_y, scan_x, 3, 3), dtype=np.float64)
    workers = max(1, int(config.num_workers))
    total = scan_y * scan_x
    done = 0
    matcher = prepare_no_cif_matcher(template, config)
    if workers > 1:
        max_pending = max(workers * 4, 1)
        coords = ((y, x) for y in range(scan_y) for x in range(scan_x))

        def run_one(scan_y_idx: int, scan_x_idx: int) -> tuple[int, int, float, float]:
            if stop_requested is not None and stop_requested():
                raise OrientationCancelled(
                    f"No-CIF matching cancelled at scan ({scan_y_idx}, {scan_x_idx})."
                )
            angle_deg, score = _match_no_cif_prepared(
                matcher,
                peaks.peaks[scan_y_idx][scan_x_idx],
                config,
            )
            return scan_y_idx, scan_x_idx, angle_deg, score

        pending = set()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                if stop_requested is not None and stop_requested():
                    raise OrientationCancelled(
                        f"No-CIF matching cancelled after {done}/{total} patterns."
                    )
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
                    y, x, angle_deg, score = future.result()
                    angle[y, x] = angle_deg
                    corr[y, x] = score
                    matrix[y, x] = _rotation_matrix_z(
                        angle_deg if np.isfinite(angle_deg) else 0.0
                    )
                    done += 1
                    if progress is not None:
                        progress(done, total)
        return OrientationResult(
            corr=corr,
            angle_0_deg=np.zeros_like(angle),
            angle_1_deg=np.zeros_like(angle),
            in_plane_angle_deg=np.mod(angle, 360.0),
            matrix=matrix,
            raw=None,
        )

    for y in range(scan_y):
        for x in range(scan_x):
            if stop_requested is not None and stop_requested():
                raise OrientationCancelled(f"No-CIF matching cancelled at scan ({y}, {x}).")
            angle_deg, score = _match_no_cif_prepared(matcher, peaks.peaks[y][x], config)
            angle[y, x] = angle_deg
            corr[y, x] = score
            matrix[y, x] = _rotation_matrix_z(angle_deg if np.isfinite(angle_deg) else 0.0)
            done += 1
            if progress is not None:
                progress(done, total)
            if stop_requested is not None and stop_requested():
                raise OrientationCancelled(
                    f"No-CIF matching cancelled after {done}/{total} patterns."
                )

    return OrientationResult(
        corr=corr,
        angle_0_deg=np.zeros_like(angle),
        angle_1_deg=np.zeros_like(angle),
        in_plane_angle_deg=np.mod(angle, 360.0),
        matrix=matrix,
        raw=None,
    )


def peak_list_to_centered_vectors_px(
    peaks: PeakList,
    config: NoCifInPlaneConfig | OrientationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert peak coordinates to centered pixel vectors."""

    vectors = np.zeros((peaks.count, 2), dtype=np.float64)
    intensity = np.asarray(peaks.intensity, dtype=np.float64)
    if peaks.count:
        vectors[:, 0] = np.asarray(peaks.qx, dtype=np.float64) - float(config.qx_center)
        vectors[:, 1] = np.asarray(peaks.qy, dtype=np.float64) - float(config.qy_center)
        if config.flip_qy:
            vectors[:, 1] *= -1.0
    return vectors, intensity


def rotate_vectors(vectors: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate 2D vectors counterclockwise in pixel-vector space."""

    theta = np.deg2rad(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rot = np.asarray([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    return np.asarray(vectors, dtype=np.float64) @ rot.T


def load_crystal_structure(cif_path: str | Path, k_max: float):
    """Load a CIF and calculate structure factors without building a plan."""

    from py4DSTEM.process.diffraction import Crystal

    if not Path(cif_path).exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    crystal = Crystal.from_CIF(str(cif_path))
    crystal.calculate_structure_factors(k_max=float(k_max))
    return crystal


def build_bvm_from_peaks(
    peaks: PeakDetectionResult,
    *,
    qx_center: float,
    qy_center: float,
) -> np.ndarray:
    """Build a Bragg vector map from centered detected peak coordinates."""

    qy_size, qx_size = peaks.diffraction_shape
    out = np.zeros((qy_size, qx_size), dtype=np.float64)
    target_x = (qx_size - 1) / 2.0
    target_y = (qy_size - 1) / 2.0
    for scan_y in range(peaks.scan_shape[0]):
        for scan_x in range(peaks.scan_shape[1]):
            peak_list = peaks.peaks[scan_y][scan_x]
            if peak_list.count == 0:
                continue
            xs = np.rint(peak_list.qx - float(qx_center) + target_x).astype(np.int64)
            ys = np.rint(peak_list.qy - float(qy_center) + target_y).astype(np.int64)
            valid = (xs >= 0) & (xs < qx_size) & (ys >= 0) & (ys < qy_size)
            np.add.at(out, (ys[valid], xs[valid]), peak_list.intensity[valid])
    return out


def radial_q_profile(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate notebook-style radial q profile from a BVM image."""

    arr = np.asarray(image, dtype=np.float64)
    cy = (arr.shape[0] - 1) / 2.0
    cx = (arr.shape[1] - 1) / 2.0
    yy, xx = np.indices(arr.shape, dtype=np.float64)
    radius = np.rint(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)).astype(np.int64)
    max_radius = int(radius.max())
    summed = np.bincount(radius.ravel(), weights=arr.ravel(), minlength=max_radius + 1)
    counts = np.bincount(radius.ravel(), minlength=max_radius + 1)
    profile = summed / np.maximum(counts, 1)
    q = np.arange(max_radius + 1, dtype=np.float64)
    return q, profile


def peak_list_radial_profile(
    peaks: PeakList,
    *,
    qx_center: float,
    qy_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a radial profile from one scan pixel's detected Bragg peaks."""

    if peaks.count == 0:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    dx = np.asarray(peaks.qx, dtype=np.float64) - float(qx_center)
    dy = np.asarray(peaks.qy, dtype=np.float64) - float(qy_center)
    radius = np.rint(np.sqrt(dx**2 + dy**2)).astype(np.int64)
    max_radius = int(radius.max()) if radius.size else 0
    summed = np.bincount(
        radius,
        weights=np.asarray(peaks.intensity, dtype=np.float64),
        minlength=max_radius + 1,
    )
    counts = np.bincount(radius, minlength=max_radius + 1)
    profile = summed / np.maximum(counts, 1)
    return np.arange(max_radius + 1, dtype=np.float64), profile


def structure_factor_profile(
    crystal,
    k_max: float,
    *,
    points: int = 250,
    zone_z_tol: float = 0.01,
    zone_axis: tuple[float, float, float] | None = None,
    sigma_excitation_error: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial structure-factor profile from py4DSTEM-generated diffraction."""

    preview = cif_zone_preview(
        crystal,
        k_max,
        zone_axis=(0.0, 0.0, 1.0) if zone_axis is None else zone_axis,
        sigma_excitation_error=sigma_excitation_error,
        points=points,
    )
    return preview.q_profile, preview.intensity_profile


def cif_zone_preview(
    crystal,
    k_max: float,
    *,
    zone_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    zone_z_tol: float = 0.01,
    sigma_excitation_error: float = 0.02,
    points: int = 250,
) -> CifZonePreview:
    """Return py4DSTEM-generated diffraction for one selected beam direction."""

    pattern = crystal.generate_diffraction_pattern(
        zone_axis_lattice=np.asarray(zone_axis, dtype=np.float64),
        sigma_excitation_error=float(sigma_excitation_error),
        tol_excitation_error_mult=max(
            float(zone_z_tol) / max(float(sigma_excitation_error), 1.0e-12),
            0.1,
        ),
        k_max=float(k_max),
    )
    qx, qy, intensity, hkl = _pattern_arrays(pattern)
    q_radius = np.sqrt(qx**2 + qy**2)
    q_profile, intensity_profile = _radial_profile_from_q(
        q_radius,
        intensity,
        k_max=float(k_max),
        points=int(points),
    )
    return CifZonePreview(
        qx=np.asarray(qx, dtype=np.float64),
        qy=np.asarray(qy, dtype=np.float64),
        q_radius=np.asarray(q_radius, dtype=np.float64),
        intensity=np.asarray(intensity, dtype=np.float64),
        hkl=hkl,
        q_profile=q_profile,
        intensity_profile=intensity_profile,
    )


def scan_q_pixel_sizes(
    radial_source: np.ndarray | tuple[np.ndarray, np.ndarray],
    crystal,
    *,
    k_max: float,
    pixel_min: float,
    pixel_max: float,
    steps: int,
    zone_z_tol: float = 0.01,
    zone_axis: tuple[float, float, float] | None = None,
    sigma_excitation_error: float = 0.02,
) -> PixelSizeScanResult:
    """Score q-pixel sizes by matching BVM radial profile to structure factors."""

    if isinstance(radial_source, tuple):
        q_pixels, intensity = radial_source
        q_pixels = np.asarray(q_pixels, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)
    else:
        q_pixels, intensity = radial_q_profile(radial_source)
    weighted = intensity * q_pixels
    q_sf, intensity_sf = structure_factor_profile(
        crystal,
        k_max,
        zone_z_tol=zone_z_tol,
        zone_axis=zone_axis,
        sigma_excitation_error=sigma_excitation_error,
    )
    pixel_sizes = np.linspace(float(pixel_min), float(pixel_max), int(max(2, steps)))
    scores = np.zeros_like(pixel_sizes)
    weighted_norm = _normalize_1d(weighted)
    for idx, pixel_size in enumerate(pixel_sizes):
        sf_at_exp = np.interp(q_pixels * pixel_size, q_sf, intensity_sf, left=0.0, right=0.0)
        scores[idx] = float(np.dot(weighted_norm, _normalize_1d(sf_at_exp)))
    best_idx = int(np.argmax(scores))
    return PixelSizeScanResult(
        q_pixels=q_pixels,
        weighted_intensity=weighted,
        q_sf=q_sf,
        intensity_sf=intensity_sf,
        pixel_sizes=pixel_sizes,
        scores=scores,
        best_pixel_size=float(pixel_sizes[best_idx]),
    )


def diagnose_cif_fit(
    peaks: PeakList,
    crystal,
    *,
    qx_center: float,
    qy_center: float,
    flip_qy: bool,
    q_pixel_size: float,
    k_max: float,
    zone_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    zone_z_tol: float = 0.01,
    sigma_excitation_error: float = 0.02,
    tolerance: float | None = None,
) -> CifFitDiagnostic:
    """Compare current experimental peaks with a py4DSTEM CIF simulation."""

    config = OrientationConfig(
        cif_path=Path(),
        mode="in_plane",
        q_pixel_size=float(q_pixel_size),
        qx_center=float(qx_center),
        qy_center=float(qy_center),
        flip_qy=bool(flip_qy),
        k_max=float(k_max),
        sigma_excitation_error=float(sigma_excitation_error),
        fiber_axis=zone_axis,
    )
    experimental = peak_list_to_pointlist(peaks, config)
    exp_qx = np.asarray(experimental.data["qx"], dtype=np.float64)
    exp_qy = np.asarray(experimental.data["qy"], dtype=np.float64)
    preview = cif_zone_preview(
        crystal,
        k_max,
        zone_axis=zone_axis,
        zone_z_tol=zone_z_tol,
        sigma_excitation_error=sigma_excitation_error,
    )
    sim_qx = preview.qx
    sim_qy = preview.qy
    match_tolerance = float(tolerance) if tolerance is not None else max(0.04, float(q_pixel_size) * 4.0)
    matched_exp, matched_sim, residuals = _nearest_peak_matches(
        exp_qx,
        exp_qy,
        sim_qx,
        sim_qy,
        match_tolerance,
    )
    mean = float(np.mean(residuals)) if residuals.size else float("nan")
    median = float(np.median(residuals)) if residuals.size else float("nan")
    suggestions = _fit_suggestions(
        peaks,
        crystal,
        qx_center=qx_center,
        qy_center=qy_center,
        flip_qy=flip_qy,
        q_pixel_size=q_pixel_size,
        k_max=k_max,
        zone_axis=zone_axis,
        zone_z_tol=zone_z_tol,
        sigma_excitation_error=sigma_excitation_error,
        tolerance=match_tolerance,
        current_count=int(matched_exp.size),
        current_median=median,
    )
    return CifFitDiagnostic(
        exp_qx=exp_qx,
        exp_qy=exp_qy,
        sim_qx=sim_qx,
        sim_qy=sim_qy,
        matched_exp_indices=matched_exp,
        matched_sim_indices=matched_sim,
        residuals=residuals,
        matched_count=int(matched_exp.size),
        mean_residual=mean,
        median_residual=median,
        tolerance=match_tolerance,
        suggestions=suggestions,
    )


def _fit_suggestions(
    peaks: PeakList,
    crystal,
    *,
    qx_center: float,
    qy_center: float,
    flip_qy: bool,
    q_pixel_size: float,
    k_max: float,
    zone_axis: tuple[float, float, float],
    zone_z_tol: float,
    sigma_excitation_error: float,
    tolerance: float,
    current_count: int,
    current_median: float,
) -> list[str]:
    suggestions: list[str] = []
    flipped_config = OrientationConfig(
        cif_path=Path(),
        mode="in_plane",
        q_pixel_size=float(q_pixel_size),
        qx_center=float(qx_center),
        qy_center=float(qy_center),
        flip_qy=not flip_qy,
    )
    flipped_points = peak_list_to_pointlist(peaks, flipped_config)
    preview = cif_zone_preview(
        crystal,
        k_max,
        zone_axis=zone_axis,
        zone_z_tol=zone_z_tol,
        sigma_excitation_error=sigma_excitation_error,
    )
    flipped_matches, _flipped_sim, _flipped_residuals = _nearest_peak_matches(
        np.asarray(flipped_points.data["qx"], dtype=np.float64),
        np.asarray(flipped_points.data["qy"], dtype=np.float64),
        preview.qx,
        preview.qy,
        tolerance,
    )
    if int(flipped_matches.size) > current_count:
        suggestions.append("flip qy gives more nearest-neighbor matches")
    try:
        q_pixels, radial_intensity = peak_list_radial_profile(
            peaks,
            qx_center=qx_center,
            qy_center=qy_center,
        )
        pixel_scan = scan_q_pixel_sizes(
            (q_pixels, radial_intensity),
            crystal,
            k_max=k_max,
            pixel_min=max(float(q_pixel_size) * 0.5, 1.0e-8),
            pixel_max=max(float(q_pixel_size) * 1.5, 2.0e-8),
            steps=21,
            zone_axis=zone_axis,
            zone_z_tol=zone_z_tol,
            sigma_excitation_error=sigma_excitation_error,
        )
        if abs(pixel_scan.best_pixel_size - float(q_pixel_size)) > max(float(q_pixel_size) * 0.05, 1.0e-8):
            suggestions.append(f"radial scan prefers q pixel size {pixel_scan.best_pixel_size:.6g}")
    except Exception:
        pass
    if current_count < max(3, min(8, peaks.count // 4)):
        suggestions.append("few matched peaks; check zone axis, center, k max, or peak detection")
    if np.isfinite(current_median) and current_median > tolerance * 0.75:
        suggestions.append("matched residual is high; q pixel size or diffraction center may need adjustment")
    if not suggestions:
        suggestions.append("current calibration is internally consistent for the selected zone")
    return suggestions


def _pattern_arrays(pattern) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pattern.data
    fields = data.dtype.names or ()
    qx = np.asarray(data["qx"], dtype=np.float64) if "qx" in fields else np.zeros(0, dtype=np.float64)
    qy = np.asarray(data["qy"], dtype=np.float64) if "qy" in fields else np.zeros(0, dtype=np.float64)
    intensity = (
        np.asarray(data["intensity"], dtype=np.float64)
        if "intensity" in fields
        else np.ones(qx.shape, dtype=np.float64)
    )
    if all(field in fields for field in ("h", "k", "l")):
        hkl = np.stack(
            (
                np.asarray(data["h"], dtype=np.int64),
                np.asarray(data["k"], dtype=np.int64),
                np.asarray(data["l"], dtype=np.int64),
            ),
            axis=1,
        )
    else:
        hkl = np.zeros((qx.size, 3), dtype=np.int64)
    return qx, qy, intensity, hkl


def _radial_profile_from_q(
    q_radius: np.ndarray,
    intensity: np.ndarray,
    *,
    k_max: float,
    points: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.linspace(0.0, float(k_max), max(2, int(points)))
    profile = np.zeros_like(q)
    if q_radius.size == 0:
        return q, profile
    step = q[1] - q[0] if q.size > 1 else 1.0
    indices = np.clip(np.rint(q_radius / max(step, 1.0e-12)).astype(np.int64), 0, q.size - 1)
    np.add.at(profile, indices, np.asarray(intensity, dtype=np.float64))
    if np.max(profile) > 0:
        profile = profile / np.max(profile)
    return q, profile


def _nearest_peak_matches(
    exp_qx: np.ndarray,
    exp_qy: np.ndarray,
    sim_qx: np.ndarray,
    sim_qy: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if exp_qx.size == 0 or sim_qx.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    tree = cKDTree(np.column_stack((sim_qx, sim_qy)))
    distances, indices = tree.query(np.column_stack((exp_qx, exp_qy)), k=1)
    keep = distances <= float(tolerance)
    return (
        np.flatnonzero(keep).astype(np.int64),
        indices[keep].astype(np.int64),
        distances[keep].astype(np.float64),
    )


def _zone_spot_mask(
    crystal,
    zone_axis: tuple[float, float, float] | None,
    zone_z_tol: float,
) -> np.ndarray:
    g_vec = np.asarray(crystal.g_vec_all, dtype=np.float64)
    if zone_axis is None:
        return np.abs(g_vec[2]) < float(zone_z_tol)
    axis = _normalize_zone_axis(zone_axis)
    return np.abs(axis @ g_vec) < float(zone_z_tol)


def _normalize_zone_axis(zone_axis: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    axis = np.asarray(zone_axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("zone_axis must be non-zero.")
    return axis / norm


def _zone_projection_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    basis_x = np.cross(helper, axis)
    basis_x /= max(float(np.linalg.norm(basis_x)), 1.0e-12)
    basis_y = np.cross(axis, basis_x)
    basis_y /= max(float(np.linalg.norm(basis_y)), 1.0e-12)
    return basis_x, basis_y


def _crystal_hkl(crystal, mask: np.ndarray) -> np.ndarray:
    for names in (("h", "k", "l"), ("g_h", "g_k", "g_l")):
        values = []
        for name in names:
            value = getattr(crystal, name, None)
            if value is None:
                break
            values.append(np.asarray(value)[mask])
        if len(values) == 3:
            return np.stack(values, axis=1).astype(np.int64, copy=False)
    return np.zeros((int(np.count_nonzero(mask)), 3), dtype=np.int64)


def orientation_color_image(
    result: OrientationResult,
    mode: OrientationMode,
    *,
    symmetry_order: int = 1,
) -> np.ndarray:
    """Render fitted orientations as an RGB image."""

    corr_scale = _normalize_2d(result.corr)
    if mode == "in_plane":
        symmetry = max(1, int(symmetry_order))
        period = 360.0 / float(symmetry)
        hue = np.nan_to_num(np.mod(result.in_plane_angle_deg, period) / period)
        sat = np.ones_like(hue)
        val = corr_scale
        return _hsv_to_rgb(hue, sat, val)

    zone = np.asarray(result.matrix[:, :, :, 2], dtype=np.float64)
    rgb = np.abs(zone)
    rgb = rgb / np.maximum(np.max(rgb, axis=2, keepdims=True), 1e-12)
    return np.clip(rgb * corr_scale[:, :, None], 0.0, 1.0)


def generate_fit_patterns(workspace: OrientationWorkspace, orientation) -> list[object]:
    """Generate simulated Bragg patterns for each requested orientation match."""

    patterns = []
    for idx in range(int(workspace.config.num_matches)):
        patterns.append(
            workspace.crystal.generate_diffraction_pattern(
                orientation,
                ind_orientation=idx,
                sigma_excitation_error=float(workspace.config.sigma_excitation_error),
            )
        )
    return patterns


def match_current_orientation(
    workspace: OrientationWorkspace,
    peaks: PeakList,
) -> object:
    """Match the orientation for one diffraction pattern."""

    pointlist = peak_list_to_pointlist(peaks, workspace.config)
    return workspace.crystal.match_single_pattern(
        pointlist,
        num_matches_return=int(workspace.config.num_matches),
        min_number_peaks=int(workspace.config.min_number_peaks),
        plot_corr=False,
        verbose=False,
    )


def match_orientation_map(
    workspace: OrientationWorkspace,
    peaks: PeakDetectionResult,
    *,
    progress: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> OrientationResult:
    """Match orientations across a full scan using detected peaks."""

    from py4DSTEM.process.diffraction.utils import OrientationMap

    scan_y, scan_x = peaks.scan_shape
    orientation_map = OrientationMap(
        num_x=scan_y,
        num_y=scan_x,
        num_matches=int(workspace.config.num_matches),
    )
    total = scan_y * scan_x
    done = 0
    workers = max(1, int(workspace.config.num_workers))
    if workers > 1 and not workspace.config.use_cuda:
        max_pending = max(workers * 2, 1)
        coords = ((y, x) for y in range(scan_y) for x in range(scan_x))

        def run_one(scan_y_idx: int, scan_x_idx: int):
            if stop_requested is not None and stop_requested():
                raise OrientationCancelled(
                    f"Orientation matching cancelled at scan ({scan_y_idx}, {scan_x_idx})."
                )
            orientation = match_current_orientation(workspace, peaks.peaks[scan_y_idx][scan_x_idx])
            return scan_y_idx, scan_x_idx, orientation

        pending = set()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                if stop_requested is not None and stop_requested():
                    raise OrientationCancelled(
                        f"Orientation matching cancelled after {done}/{total} patterns."
                    )
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
                    y, x, orientation = future.result()
                    orientation_map.set_orientation(orientation, y, x)
                    done += 1
                    if progress is not None:
                        progress(done, total)
        return orientation_result_from_map(orientation_map)

    for y in range(scan_y):
        for x in range(scan_x):
            if stop_requested is not None and stop_requested():
                raise OrientationCancelled(f"Orientation matching cancelled at scan ({y}, {x}).")
            orientation = match_current_orientation(workspace, peaks.peaks[y][x])
            orientation_map.set_orientation(orientation, y, x)
            done += 1
            if progress is not None:
                progress(done, total)
            if stop_requested is not None and stop_requested():
                raise OrientationCancelled(
                    f"Orientation matching cancelled after {done}/{total} patterns."
                )

    return orientation_result_from_map(orientation_map)


def orientation_result_from_map(orientation_map) -> OrientationResult:
    """Extract GUI-friendly arrays from a py4DSTEM OrientationMap."""

    angles = np.rad2deg(np.asarray(orientation_map.angles[:, :, 0, :], dtype=np.float64))
    return OrientationResult(
        corr=np.asarray(orientation_map.corr[:, :, 0], dtype=np.float64),
        angle_0_deg=angles[:, :, 0],
        angle_1_deg=angles[:, :, 1],
        in_plane_angle_deg=np.mod(angles[:, :, 2], 360.0),
        matrix=np.asarray(orientation_map.matrix[:, :, 0, :, :], dtype=np.float64),
        raw=orientation_map,
    )


def peak_list_to_pointlist(peaks: PeakList, config: OrientationConfig):
    """Convert 4DLAB peak coordinates to py4DSTEM PointList q-vectors."""

    from emdfile import PointList

    data = np.zeros(
        peaks.count,
        dtype=[("qx", np.float64), ("qy", np.float64), ("intensity", np.float64)],
    )
    if peaks.count:
        data["qx"] = (np.asarray(peaks.qx, dtype=np.float64) - float(config.qx_center)) * float(
            config.q_pixel_size
        )
        qy = (np.asarray(peaks.qy, dtype=np.float64) - float(config.qy_center)) * float(
            config.q_pixel_size
        )
        if config.flip_qy:
            qy *= -1.0
        data["qy"] = qy
        data["intensity"] = np.asarray(peaks.intensity, dtype=np.float64)
    return PointList(data=data)


def orientation_cuda_available() -> bool:
    """Return whether py4DSTEM/CuPy CUDA orientation matching can be requested."""

    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _normalize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.nanmin(arr) if arr.size else arr
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return np.zeros_like(arr)
    return arr / norm


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    if total <= 0:
        if arr.size == 0:
            return arr
        return np.ones_like(arr) / float(arr.size)
    return arr / total


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    return np.asarray(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


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


def _zone_axis_range(config: OrientationConfig):
    if config.mode == "in_plane":
        return "fiber"
    if config.zone_range_mode == "center_cone":
        return "fiber"
    if config.zone_range_mode == "three_vertices":
        vertices = np.asarray(config.zone_axis_vertices, dtype=np.float64)
        if vertices.shape != (3, 3):
            raise ValueError("zone_axis_vertices must be a 3 x 3 set of zone axes.")
        for idx, vertex in enumerate(vertices):
            if float(np.linalg.norm(vertex)) <= 0:
                raise ValueError(f"zone_axis_vertices[{idx}] must be non-zero.")
        return vertices
    return "auto"


def _fiber_axis(config: OrientationConfig):
    if config.mode == "out_of_plane" and config.zone_range_mode == "center_cone":
        axis = np.asarray(config.zone_axis_center, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 0:
            raise ValueError("zone_axis_center must be non-zero.")
        return axis / norm
    if config.mode != "in_plane":
        return None
    axis = np.asarray(config.fiber_axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("fiber_axis must be non-zero.")
    return axis / norm


def _fiber_angles(config: OrientationConfig):
    if config.mode == "out_of_plane" and config.zone_range_mode == "center_cone":
        angle = float(config.zone_angle_deg)
        if angle < 0:
            raise ValueError("zone_angle_deg must be zero or positive.")
        return np.asarray([angle, 360.0], dtype=np.float64)
    if config.mode != "in_plane":
        return None
    return np.asarray([0.0, 360.0], dtype=np.float64)
