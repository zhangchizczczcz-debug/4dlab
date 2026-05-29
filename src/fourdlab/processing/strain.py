"""Lattice-vector strain analysis from detected nanobeam diffraction peaks."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from fourdlab.processing.peak_detection import PeakDetectionResult, PeakList

ProgressCallback = Callable[[int, int], None]
StopCallback = Callable[[], bool]


class StrainCancelled(RuntimeError):
    """Raised when full-scan strain fitting is cancelled by the caller."""


@dataclass(frozen=True)
class StrainConfig:
    """Parameters for lattice-transform strain fitting."""

    qx_center: float
    qy_center: float
    flip_qy: bool = True
    q_pixel_size: float = 1.0
    match_tolerance_px: float = 3.0
    min_number_peaks: int = 4
    max_index: int = 3
    center_exclusion_px: float = 2.0
    reference_bounds: tuple[int, int, int, int] | None = None
    rotate_angle_deg: float = 0.0
    flip_theta: bool = False
    use_intensity_weights: bool = True
    num_workers: int = 1


@dataclass
class StrainFit:
    """Fitted lattice transform for one diffraction pattern."""

    hk: np.ndarray
    predicted_px: np.ndarray
    observed_px: np.ndarray
    fitted_px: np.ndarray
    beta: np.ndarray
    error_px: float
    matched_count: int


@dataclass
class PreparedStrainMatcher:
    """Precomputed lattice points for repeated strain fitting."""

    hk: np.ndarray
    predicted_px: np.ndarray
    min_number_peaks: int
    center_exclusion_px: float
    match_tolerance_px: float
    use_intensity_weights: bool


@dataclass
class StrainResult:
    """Full-scan strain outputs used by the GUI."""

    e_xx: np.ndarray
    e_yy: np.ndarray
    e_xy: np.ndarray
    theta: np.ndarray
    mask: np.ndarray
    error_px: np.ndarray
    matched_count: np.ndarray
    beta: np.ndarray
    beta_relative: np.ndarray
    reference_beta: np.ndarray
    g1_px: np.ndarray
    g2_px: np.ndarray
    config: StrainConfig


def guess_basis_from_peaks(
    peaks: PeakList,
    config: StrainConfig,
    *,
    max_candidates: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Guess two reciprocal lattice basis vectors from one peak list."""

    vectors, intensity = centered_vectors_px(peaks, config)
    if vectors.shape[0] < 2:
        raise ValueError("Need at least two peaks to guess a lattice basis.")
    radius = np.linalg.norm(vectors, axis=1)
    finite = np.isfinite(radius) & (radius >= float(config.center_exclusion_px))
    vectors = vectors[finite]
    intensity = intensity[finite]
    radius = radius[finite]
    if vectors.shape[0] < 2:
        raise ValueError("No usable non-central peaks for basis guessing.")

    order = np.lexsort((-_normalize(intensity), radius))
    order = order[: max(2, int(max_candidates))]
    candidates = vectors[order]
    candidate_radius = radius[order]
    candidate_intensity = _normalize(intensity[order])

    best_score = -np.inf
    best_pair: tuple[np.ndarray, np.ndarray] | None = None
    for i in range(candidates.shape[0]):
        for j in range(i + 1, candidates.shape[0]):
            v1 = candidates[i]
            v2 = candidates[j]
            r1 = float(candidate_radius[i])
            r2 = float(candidate_radius[j])
            if r1 <= 0 or r2 <= 0:
                continue
            cross = float(v1[0] * v2[1] - v1[1] * v2[0])
            sin_angle = abs(cross) / (r1 * r2)
            if sin_angle < 0.35:
                continue
            length_penalty = (r1 + r2) / max(float(np.nanmedian(candidate_radius)), 1.0)
            intensity_score = 0.5 * (candidate_intensity[i] + candidate_intensity[j])
            score = sin_angle + 0.25 * intensity_score - 0.08 * length_penalty
            if score > best_score:
                g1 = np.asarray(v1, dtype=np.float64)
                g2 = np.asarray(v2, dtype=np.float64)
                if cross < 0:
                    g1, g2 = g2, g1
                best_pair = (g1, g2)
                best_score = score

    if best_pair is None:
        raise ValueError("Could not find two non-collinear basis vectors.")
    return best_pair


def fit_lattice_transform(
    peaks: PeakList,
    g1_px: np.ndarray,
    g2_px: np.ndarray,
    config: StrainConfig,
) -> StrainFit:
    """Fit one diffraction pattern to the supplied reciprocal basis."""

    return _fit_lattice_transform_prepared(
        peaks,
        prepare_strain_matcher(g1_px, g2_px, config),
        config,
    )


def prepare_strain_matcher(
    g1_px: np.ndarray,
    g2_px: np.ndarray,
    config: StrainConfig,
) -> PreparedStrainMatcher:
    """Precompute lattice h/k points for repeated strain fitting."""

    hk = lattice_indices(int(config.max_index))
    predicted = hk[:, [0]] * np.asarray(g1_px, dtype=np.float64) + hk[:, [1]] * np.asarray(
        g2_px, dtype=np.float64
    )
    predicted_radius = np.linalg.norm(predicted, axis=1)
    predicted_keep = predicted_radius >= float(config.center_exclusion_px)
    hk = hk[predicted_keep]
    predicted = predicted[predicted_keep]
    return PreparedStrainMatcher(
        hk=hk,
        predicted_px=predicted,
        min_number_peaks=int(config.min_number_peaks),
        center_exclusion_px=float(config.center_exclusion_px),
        match_tolerance_px=max(1.0e-8, float(config.match_tolerance_px)),
        use_intensity_weights=bool(config.use_intensity_weights),
    )


def _fit_lattice_transform_prepared(
    peaks: PeakList,
    matcher: PreparedStrainMatcher,
    config: StrainConfig,
) -> StrainFit:
    """Fit one diffraction pattern using precomputed lattice points."""

    vectors, intensity = centered_vectors_px(peaks, config)
    radius = np.linalg.norm(vectors, axis=1) if vectors.size else np.zeros(0)
    keep = np.isfinite(radius) & (radius >= matcher.center_exclusion_px)
    vectors = vectors[keep]
    intensity = intensity[keep]
    if vectors.shape[0] < matcher.min_number_peaks:
        return _empty_fit()
    hk = matcher.hk
    predicted = matcher.predicted_px
    if predicted.shape[0] < matcher.min_number_peaks:
        return _empty_fit()

    tree = cKDTree(vectors)
    distances, indices = tree.query(predicted, k=1, distance_upper_bound=matcher.match_tolerance_px)
    valid = np.isfinite(distances) & (indices < vectors.shape[0])
    if not np.any(valid):
        return _empty_fit()

    selected: list[tuple[int, int, float]] = []
    used_observed: set[int] = set()
    valid_indices = np.flatnonzero(valid)
    for pred_idx in valid_indices[np.argsort(distances[valid])]:
        obs_idx = int(indices[pred_idx])
        if obs_idx in used_observed:
            continue
        selected.append((int(pred_idx), obs_idx, float(distances[pred_idx])))
        used_observed.add(obs_idx)

    if len(selected) < matcher.min_number_peaks:
        return _empty_fit()

    pred_idx = np.asarray([item[0] for item in selected], dtype=np.int64)
    obs_idx = np.asarray([item[1] for item in selected], dtype=np.int64)
    pred = predicted[pred_idx]
    obs = vectors[obs_idx]
    if np.linalg.matrix_rank(pred) < 2:
        return _empty_fit()

    if matcher.use_intensity_weights:
        weights = _fit_weights(intensity[obs_idx])
        pred_fit = pred * weights[:, None]
        obs_fit = obs * weights[:, None]
    else:
        pred_fit = pred
        obs_fit = obs
    transform_t, *_unused = np.linalg.lstsq(pred_fit, obs_fit, rcond=None)
    beta = np.asarray(transform_t.T, dtype=np.float64)
    fitted = pred @ transform_t
    residual = fitted - obs
    error = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    return StrainFit(
        hk=hk[pred_idx].copy(),
        predicted_px=pred,
        observed_px=obs,
        fitted_px=fitted,
        beta=beta,
        error_px=error,
        matched_count=int(pred.shape[0]),
    )


def calculate_strain_map(
    peaks: PeakDetectionResult,
    g1_px: np.ndarray,
    g2_px: np.ndarray,
    config: StrainConfig,
    *,
    progress: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> StrainResult:
    """Fit lattice transforms for a full scan and convert them to strain maps."""

    scan_y, scan_x = peaks.scan_shape
    beta = np.full((scan_y, scan_x, 2, 2), np.nan, dtype=np.float64)
    error = np.full((scan_y, scan_x), np.nan, dtype=np.float64)
    matched = np.zeros((scan_y, scan_x), dtype=np.int32)
    mask = np.zeros((scan_y, scan_x), dtype=bool)

    total = scan_y * scan_x
    workers = max(1, int(config.num_workers))
    done = _fit_strain_transforms(
        peaks,
        g1_px,
        g2_px,
        config,
        beta,
        error,
        matched,
        mask,
        progress=progress,
        stop_requested=stop_requested,
        workers=workers,
    )
    if done < total and stop_requested is not None and stop_requested():
        raise StrainCancelled(f"Strain fitting cancelled after {done}/{total} patterns.")

    reference_mask = _reference_mask(mask, config.reference_bounds)
    if not np.any(reference_mask):
        reference_mask = mask
    if not np.any(reference_mask):
        raise ValueError("No scan positions had enough matched lattice peaks.")

    reference_beta = np.nanmedian(beta[reference_mask], axis=0)
    if not np.all(np.isfinite(reference_beta)) or abs(float(np.linalg.det(reference_beta))) < 1.0e-12:
        raise ValueError("Reference lattice transform is singular or invalid.")

    inv_reference = np.linalg.inv(reference_beta)
    beta_relative = np.full_like(beta, np.nan)
    for y in range(scan_y):
        for x in range(scan_x):
            if mask[y, x]:
                beta_relative[y, x] = beta[y, x] @ inv_reference

    e_xx, e_yy, e_xy, theta = strain_from_beta(beta_relative, mask)
    e_xx, e_yy, e_xy, theta = rotate_strain_components(
        e_xx,
        e_yy,
        e_xy,
        theta,
        config.rotate_angle_deg,
        flip_theta=config.flip_theta,
    )

    return StrainResult(
        e_xx=e_xx,
        e_yy=e_yy,
        e_xy=e_xy,
        theta=theta,
        mask=mask,
        error_px=error,
        matched_count=matched,
        beta=beta,
        beta_relative=beta_relative,
        reference_beta=reference_beta,
        g1_px=np.asarray(g1_px, dtype=np.float64).copy(),
        g2_px=np.asarray(g2_px, dtype=np.float64).copy(),
        config=config,
    )


def _fit_strain_transforms(
    peaks: PeakDetectionResult,
    g1_px: np.ndarray,
    g2_px: np.ndarray,
    config: StrainConfig,
    beta: np.ndarray,
    error: np.ndarray,
    matched: np.ndarray,
    mask: np.ndarray,
    *,
    progress: ProgressCallback | None,
    stop_requested: StopCallback | None,
    workers: int,
) -> int:
    scan_y, scan_x = peaks.scan_shape
    total = scan_y * scan_x
    done = 0
    matcher = prepare_strain_matcher(g1_px, g2_px, config)

    def store_fit(scan_y_idx: int, scan_x_idx: int, fit: StrainFit) -> None:
        if fit.matched_count >= int(config.min_number_peaks) and np.all(np.isfinite(fit.beta)):
            beta[scan_y_idx, scan_x_idx] = fit.beta
            error[scan_y_idx, scan_x_idx] = fit.error_px
            matched[scan_y_idx, scan_x_idx] = fit.matched_count
            mask[scan_y_idx, scan_x_idx] = True

    if workers > 1:
        max_pending = max(workers * 4, 1)
        coords = ((y, x) for y in range(scan_y) for x in range(scan_x))

        def run_one(scan_y_idx: int, scan_x_idx: int) -> tuple[int, int, StrainFit]:
            if stop_requested is not None and stop_requested():
                raise StrainCancelled(
                    f"Strain fitting cancelled at scan ({scan_y_idx}, {scan_x_idx})."
                )
            fit = _fit_lattice_transform_prepared(
                peaks.peaks[scan_y_idx][scan_x_idx],
                matcher,
                config,
            )
            return scan_y_idx, scan_x_idx, fit

        pending = set()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                if stop_requested is not None and stop_requested():
                    raise StrainCancelled(f"Strain fitting cancelled after {done}/{total} patterns.")
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
                    y, x, fit = future.result()
                    store_fit(y, x, fit)
                    done += 1
                    if progress is not None:
                        progress(done, total)
        return done

    for y in range(scan_y):
        for x in range(scan_x):
            if stop_requested is not None and stop_requested():
                raise StrainCancelled(f"Strain fitting cancelled at scan ({y}, {x}).")
            store_fit(
                y,
                x,
                _fit_lattice_transform_prepared(peaks.peaks[y][x], matcher, config),
            )
            done += 1
            if progress is not None:
                progress(done, total)
            if stop_requested is not None and stop_requested():
                raise StrainCancelled(f"Strain fitting cancelled after {done}/{total} patterns.")
    return done


def strain_from_beta(beta_relative: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Convert relative reciprocal-lattice transforms into strain components."""

    shape = mask.shape
    e_xx = np.full(shape, np.nan, dtype=np.float64)
    e_yy = np.full(shape, np.nan, dtype=np.float64)
    e_xy = np.full(shape, np.nan, dtype=np.float64)
    theta = np.full(shape, np.nan, dtype=np.float64)
    valid = mask & np.all(np.isfinite(beta_relative), axis=(2, 3))
    if np.any(valid):
        b = beta_relative[valid]
        e_xx[valid] = 1.0 - b[:, 0, 0]
        e_yy[valid] = 1.0 - b[:, 1, 1]
        e_xy[valid] = -(b[:, 0, 1] + b[:, 1, 0]) / 2.0
        theta[valid] = (b[:, 0, 1] - b[:, 1, 0]) / 2.0
    return e_xx, e_yy, e_xy, theta


def rotate_strain_components(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    theta: np.ndarray,
    angle_deg: float,
    *,
    flip_theta: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate strain tensor components into a user-selected output frame."""

    angle = -np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    c2 = c * c
    s2 = s * s
    e_xx_rot = c2 * e_xx - 2.0 * c * s * e_xy + s2 * e_yy
    e_xy_rot = c * s * (e_xx - e_yy) + (c2 - s2) * e_xy
    e_yy_rot = s2 * e_xx + 2.0 * c * s * e_xy + c2 * e_yy
    theta_out = -theta if flip_theta else theta.copy()
    return e_xx_rot, e_yy_rot, e_xy_rot, theta_out


def polar_strain_components(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    *,
    center_y: float,
    center_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian strain components to radial/tangential scan coordinates."""

    ny, nx = e_xx.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    angle = np.arctan2(yy - float(center_y), xx - float(center_x))
    c = np.cos(angle)
    s = np.sin(angle)
    c2 = c**2
    s2 = s**2
    sc = s * c
    e_rr = e_xx * c2 + e_yy * s2 + 2.0 * e_xy * sc
    e_tt = e_xx * s2 + e_yy * c2 - 2.0 * e_xy * sc
    e_rt = (e_yy - e_xx) * sc + e_xy * (c2 - s2)
    return e_rr, e_tt, e_rt


def centered_vectors_px(peaks: PeakList, config: StrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Convert peak coordinates to centered diffraction-pixel vectors."""

    vectors = np.zeros((peaks.count, 2), dtype=np.float64)
    intensity = np.asarray(peaks.intensity, dtype=np.float64)
    if peaks.count:
        vectors[:, 0] = np.asarray(peaks.qx, dtype=np.float64) - float(config.qx_center)
        vectors[:, 1] = np.asarray(peaks.qy, dtype=np.float64) - float(config.qy_center)
        if config.flip_qy:
            vectors[:, 1] *= -1.0
    return vectors, intensity


def lattice_indices(max_index: int) -> np.ndarray:
    """Return h,k pairs up to the requested index magnitude, sorted by radius."""

    limit = max(1, int(max_index))
    values: list[tuple[int, int]] = []
    for h in range(-limit, limit + 1):
        for k in range(-limit, limit + 1):
            if h == 0 and k == 0:
                continue
            values.append((h, k))
    hk = np.asarray(values, dtype=np.int32)
    radius = np.sqrt(np.sum(hk.astype(np.float64) ** 2, axis=1))
    return hk[np.argsort(radius)]


def save_strain_result(result: StrainResult, path: str | Path) -> None:
    """Save strain result arrays as a portable compressed NumPy archive."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray([1], dtype=np.int16),
        e_xx=result.e_xx,
        e_yy=result.e_yy,
        e_xy=result.e_xy,
        theta=result.theta,
        mask=result.mask,
        error_px=result.error_px,
        matched_count=result.matched_count,
        beta=result.beta,
        beta_relative=result.beta_relative,
        reference_beta=result.reference_beta,
        g1_px=result.g1_px,
        g2_px=result.g2_px,
        qx_center=np.asarray([result.config.qx_center], dtype=np.float64),
        qy_center=np.asarray([result.config.qy_center], dtype=np.float64),
        q_pixel_size=np.asarray([result.config.q_pixel_size], dtype=np.float64),
        flip_qy=np.asarray([result.config.flip_qy], dtype=bool),
        match_tolerance_px=np.asarray([result.config.match_tolerance_px], dtype=np.float64),
        min_number_peaks=np.asarray([result.config.min_number_peaks], dtype=np.int32),
        max_index=np.asarray([result.config.max_index], dtype=np.int32),
        center_exclusion_px=np.asarray([result.config.center_exclusion_px], dtype=np.float64),
        rotate_angle_deg=np.asarray([result.config.rotate_angle_deg], dtype=np.float64),
        flip_theta=np.asarray([result.config.flip_theta], dtype=bool),
    )


def _reference_mask(mask: np.ndarray, bounds: tuple[int, int, int, int] | None) -> np.ndarray:
    if bounds is None:
        return mask.copy()
    y0, y1, x0, x1 = (int(v) for v in bounds)
    out = np.zeros_like(mask, dtype=bool)
    y0 = max(0, min(mask.shape[0], y0))
    y1 = max(0, min(mask.shape[0], y1))
    x0 = max(0, min(mask.shape[1], x0))
    x1 = max(0, min(mask.shape[1], x1))
    if y1 <= y0 or x1 <= x0:
        return out
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def _empty_fit() -> StrainFit:
    empty = np.zeros((0, 2), dtype=np.float64)
    return StrainFit(
        hk=np.zeros((0, 2), dtype=np.int32),
        predicted_px=empty.copy(),
        observed_px=empty.copy(),
        fitted_px=empty.copy(),
        beta=np.full((2, 2), np.nan, dtype=np.float64),
        error_px=np.nan,
        matched_count=0,
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if high <= low:
        return np.ones_like(arr)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _fit_weights(intensity: np.ndarray) -> np.ndarray:
    weights = np.sqrt(np.clip(np.asarray(intensity, dtype=np.float64), 0.0, None))
    finite = weights[np.isfinite(weights)]
    if finite.size == 0 or float(np.nanmax(finite)) <= 0:
        return np.ones_like(weights)
    weights = weights / max(float(np.nanmedian(finite)), 1.0e-12)
    return np.clip(weights, 0.2, 5.0)
