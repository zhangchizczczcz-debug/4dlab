"""Custom diffraction peak detection for nanobeam diffraction analysis."""

from __future__ import annotations

import importlib.util
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit

from fourdlab.io import DataCube

ProgressCallback = Callable[[int, int], None]
StopCallback = Callable[[], bool]


class PeakDetectionCancelled(RuntimeError):
    """Raised when full-scan peak detection is cancelled by the caller."""


@dataclass(frozen=True)
class PeakDetectionConfig:
    """Parameters for direct local-maximum peak detection."""

    smooth_sigma: float = 0.5
    edge_boundary: int = 15
    min_relative_intensity: float = 0.0
    min_absolute_intensity: float = 0.0
    min_peak_spacing: int = 20
    max_num_peaks: int = 50
    gaussian_radius: int = 4
    refine: bool = True
    refine_method: str = "centroid"
    use_gpu: bool = False
    num_workers: int = 1


@dataclass
class PeakList:
    """Peak coordinates for one diffraction pattern."""

    qx: np.ndarray
    qy: np.ndarray
    intensity: np.ndarray
    refined: np.ndarray

    @property
    def count(self) -> int:
        return int(self.qx.size)


@dataclass
class PeakDetectionResult:
    """Peak lists for a scan grid."""

    peaks: list[list[PeakList]]
    scan_shape: tuple[int, int]
    diffraction_shape: tuple[int, int]

    def peak_count_map(self) -> np.ndarray:
        out = np.zeros(self.scan_shape, dtype=np.int32)
        for y in range(self.scan_shape[0]):
            for x in range(self.scan_shape[1]):
                out[y, x] = self.peaks[y][x].count
        return out


def detect_peaks_in_pattern(image: np.ndarray, config: PeakDetectionConfig) -> PeakList:
    """Detect diffraction peaks in one pattern."""

    raw = np.asarray(image, dtype=np.float64)
    work = np.nan_to_num(raw, nan=0.0)
    if config.smooth_sigma > 0:
        work = gaussian_filter(work, float(config.smooth_sigma))

    valid = _valid_mask(work.shape, int(config.edge_boundary))
    if valid.any():
        peak_reference = float(np.max(work[valid]))
    else:
        peak_reference = float(np.max(work))
    threshold = max(
        float(config.min_absolute_intensity),
        peak_reference * float(config.min_relative_intensity),
    )

    min_spacing = max(1, int(config.min_peak_spacing))
    neighborhood = max(3, min_spacing * 2 + 1)
    candidates = _candidate_peaks(work, valid, threshold, neighborhood, config.use_gpu)
    if candidates.size == 0:
        return _empty_peak_list()

    scores = work[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(scores)[::-1]
    selected: list[tuple[float, float, float, bool]] = []
    min_dist2 = float(min_spacing * min_spacing)
    max_num_peaks = max(1, int(config.max_num_peaks))
    edge = int(config.edge_boundary)

    for idx in order:
        y = int(candidates[idx, 0])
        x = int(candidates[idx, 1])
        if any((x - px) ** 2 + (y - py) ** 2 < min_dist2 for px, py, _i, _r in selected):
            continue
        if config.refine:
            if config.refine_method == "gaussian":
                qx, qy, intensity, refined = refine_peak_gaussian(
                    raw,
                    x,
                    y,
                    radius=config.gaussian_radius,
                )
            else:
                qx, qy, intensity, refined = refine_peak_centroid(
                    raw,
                    x,
                    y,
                    radius=config.gaussian_radius,
                )
        else:
            qx, qy, intensity, refined = float(x), float(y), float(raw[y, x]), False
        if edge > 0 and not (edge <= qx < raw.shape[1] - edge and edge <= qy < raw.shape[0] - edge):
            continue
        selected.append((qx, qy, intensity, refined))
        if len(selected) >= max_num_peaks:
            break

    if not selected:
        return _empty_peak_list()

    return PeakList(
        qx=np.asarray([p[0] for p in selected], dtype=np.float64),
        qy=np.asarray([p[1] for p in selected], dtype=np.float64),
        intensity=np.asarray([p[2] for p in selected], dtype=np.float64),
        refined=np.asarray([p[3] for p in selected], dtype=bool),
    )


def detect_peaks_in_datacube(
    cube: DataCube,
    config: PeakDetectionConfig,
    *,
    progress: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> PeakDetectionResult:
    """Detect peaks for every scan position in a datacube."""

    scan_y, scan_x = cube.scan_shape
    workers = max(1, int(config.num_workers))
    if workers > 1:
        return _detect_peaks_in_datacube_parallel(
            cube,
            config,
            progress=progress,
            stop_requested=stop_requested,
        )
    peaks: list[list[PeakList]] = []
    total = scan_y * scan_x
    done = 0
    for y in range(scan_y):
        row: list[PeakList] = []
        for x in range(scan_x):
            if stop_requested is not None and stop_requested():
                raise PeakDetectionCancelled(f"Peak detection cancelled at scan ({y}, {x}).")
            row.append(detect_peaks_in_pattern(cube.diffraction_pattern(y, x), config))
            done += 1
            if progress is not None:
                progress(done, total)
            if stop_requested is not None and stop_requested():
                raise PeakDetectionCancelled(f"Peak detection cancelled after {done}/{total} patterns.")
        peaks.append(row)
    return PeakDetectionResult(
        peaks=peaks,
        scan_shape=cube.scan_shape,
        diffraction_shape=cube.diffraction_shape,
    )


def _detect_peaks_in_datacube_parallel(
    cube: DataCube,
    config: PeakDetectionConfig,
    *,
    progress: ProgressCallback | None,
    stop_requested: StopCallback | None,
) -> PeakDetectionResult:
    scan_y, scan_x = cube.scan_shape
    total = scan_y * scan_x
    peaks: list[list[PeakList | None]] = [[None for _x in range(scan_x)] for _y in range(scan_y)]
    max_workers = max(1, int(config.num_workers))
    max_pending = max(max_workers * 3, 1)
    done = 0

    def run_one(scan_y_idx: int, scan_x_idx: int) -> tuple[int, int, PeakList]:
        if stop_requested is not None and stop_requested():
            raise PeakDetectionCancelled(
                f"Peak detection cancelled at scan ({scan_y_idx}, {scan_x_idx})."
            )
        pattern = cube.diffraction_pattern(scan_y_idx, scan_x_idx)
        return scan_y_idx, scan_x_idx, detect_peaks_in_pattern(pattern, config)

    coords = ((y, x) for y in range(scan_y) for x in range(scan_x))
    pending = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            if stop_requested is not None and stop_requested():
                raise PeakDetectionCancelled(f"Peak detection cancelled after {done}/{total} patterns.")
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
                y, x, peak_list = future.result()
                peaks[y][x] = peak_list
                done += 1
                if progress is not None:
                    progress(done, total)

    return PeakDetectionResult(
        peaks=[[peak for peak in row if peak is not None] for row in peaks],
        scan_shape=cube.scan_shape,
        diffraction_shape=cube.diffraction_shape,
    )


def save_peak_detection_result(result: PeakDetectionResult, path: str | Path) -> None:
    """Save detected peaks as a portable compressed NumPy archive."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scan_y_values: list[int] = []
    scan_x_values: list[int] = []
    qx_values: list[float] = []
    qy_values: list[float] = []
    intensity_values: list[float] = []
    refined_values: list[bool] = []

    for scan_y in range(result.scan_shape[0]):
        for scan_x in range(result.scan_shape[1]):
            peaks = result.peaks[scan_y][scan_x]
            count = peaks.count
            scan_y_values.extend([scan_y] * count)
            scan_x_values.extend([scan_x] * count)
            qx_values.extend(peaks.qx.tolist())
            qy_values.extend(peaks.qy.tolist())
            intensity_values.extend(peaks.intensity.tolist())
            refined_values.extend(peaks.refined.tolist())

    np.savez_compressed(
        path,
        version=np.asarray([1], dtype=np.int16),
        scan_shape=np.asarray(result.scan_shape, dtype=np.int32),
        diffraction_shape=np.asarray(result.diffraction_shape, dtype=np.int32),
        scan_y=np.asarray(scan_y_values, dtype=np.int32),
        scan_x=np.asarray(scan_x_values, dtype=np.int32),
        qx=np.asarray(qx_values, dtype=np.float64),
        qy=np.asarray(qy_values, dtype=np.float64),
        intensity=np.asarray(intensity_values, dtype=np.float64),
        refined=np.asarray(refined_values, dtype=bool),
    )


def load_peak_detection_result(path: str | Path) -> PeakDetectionResult:
    """Load a peak detection archive created by `save_peak_detection_result`."""

    archive = np.load(Path(path))
    scan_shape = tuple(int(v) for v in archive["scan_shape"])
    diffraction_shape = tuple(int(v) for v in archive["diffraction_shape"])
    scan_y_values = archive["scan_y"].astype(np.int32, copy=False)
    scan_x_values = archive["scan_x"].astype(np.int32, copy=False)
    qx_values = archive["qx"].astype(np.float64, copy=False)
    qy_values = archive["qy"].astype(np.float64, copy=False)
    intensity_values = archive["intensity"].astype(np.float64, copy=False)
    refined_values = archive["refined"].astype(bool, copy=False)

    peaks: list[list[PeakList]] = []
    for scan_y in range(scan_shape[0]):
        row: list[PeakList] = []
        for scan_x in range(scan_shape[1]):
            mask = (scan_y_values == scan_y) & (scan_x_values == scan_x)
            row.append(
                PeakList(
                    qx=qx_values[mask].copy(),
                    qy=qy_values[mask].copy(),
                    intensity=intensity_values[mask].copy(),
                    refined=refined_values[mask].copy(),
                )
            )
        peaks.append(row)

    return PeakDetectionResult(
        peaks=peaks,
        scan_shape=scan_shape,
        diffraction_shape=diffraction_shape,
    )


def gpu_peak_detection_available() -> bool:
    """Return whether the optional CuPy stack is importable in this environment."""

    return (
        importlib.util.find_spec("cupy") is not None
        and importlib.util.find_spec("cupyx.scipy.ndimage") is not None
    )


def _candidate_peaks(
    work: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    neighborhood: int,
    use_gpu: bool,
) -> np.ndarray:
    if use_gpu and gpu_peak_detection_available():
        try:
            return _candidate_peaks_gpu(work, valid, threshold, neighborhood)
        except Exception:
            pass

    local_max = work == maximum_filter(work, size=neighborhood, mode="nearest")
    return np.argwhere(local_max & valid & (work >= threshold))


def _candidate_peaks_gpu(
    work: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    neighborhood: int,
) -> np.ndarray:
    import cupy as cp
    from cupyx.scipy.ndimage import maximum_filter as cupy_maximum_filter

    work_gpu = cp.asarray(work)
    valid_gpu = cp.asarray(valid)
    local_max = work_gpu == cupy_maximum_filter(work_gpu, size=neighborhood, mode="nearest")
    candidates_gpu = cp.argwhere(local_max & valid_gpu & (work_gpu >= threshold))
    return cp.asnumpy(candidates_gpu)


def refine_peak_gaussian(
    image: np.ndarray,
    qx: float,
    qy: float,
    *,
    radius: int = 4,
) -> tuple[float, float, float, bool]:
    """Refine one peak using a small 2D Gaussian fit, falling back to COM."""

    arr = np.asarray(image, dtype=np.float64)
    r = max(2, int(radius))
    x0 = int(round(float(qx)))
    y0 = int(round(float(qy)))
    xmin = max(0, x0 - r)
    xmax = min(arr.shape[1], x0 + r + 1)
    ymin = max(0, y0 - r)
    ymax = min(arr.shape[0], y0 + r + 1)
    patch = arr[ymin:ymax, xmin:xmax]
    if patch.size < 9 or not np.isfinite(patch).any():
        return float(qx), float(qy), float(arr[y0, x0]), False

    patch = np.nan_to_num(patch, nan=float(np.nanmin(patch)))
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    x_global = xx + xmin
    y_global = yy + ymin
    offset0 = float(np.percentile(patch, 10))
    amp0 = float(np.max(patch) - offset0)
    if amp0 <= 0:
        return _com_refine_patch(patch, xmin, ymin, qx, qy)

    p0 = [amp0, float(qx), float(qy), max(1.0, r / 2.0), max(1.0, r / 2.0), offset0]
    lower = [0.0, xmin, ymin, 0.3, 0.3, float(np.min(patch)) - abs(amp0)]
    upper = [
        float(np.max(patch) * 5 + 1),
        xmax - 1,
        ymax - 1,
        r * 2.5,
        r * 2.5,
        float(np.max(patch)),
    ]
    try:
        popt, _ = curve_fit(
            _gaussian2d,
            (x_global.ravel(), y_global.ravel()),
            patch.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=800,
        )
        return float(popt[1]), float(popt[2]), float(popt[0] + popt[5]), True
    except Exception:
        return _com_refine_patch(patch, xmin, ymin, qx, qy)


def refine_peak_centroid(
    image: np.ndarray,
    qx: float,
    qy: float,
    *,
    radius: int = 4,
) -> tuple[float, float, float, bool]:
    """Refine one peak with a fast local center-of-mass estimate."""

    arr = np.asarray(image, dtype=np.float64)
    r = max(2, int(radius))
    x0 = int(round(float(qx)))
    y0 = int(round(float(qy)))
    xmin = max(0, x0 - r)
    xmax = min(arr.shape[1], x0 + r + 1)
    ymin = max(0, y0 - r)
    ymax = min(arr.shape[0], y0 + r + 1)
    patch = arr[ymin:ymax, xmin:xmax]
    if patch.size < 9 or not np.isfinite(patch).any():
        return float(qx), float(qy), float(arr[y0, x0]), False
    patch = np.nan_to_num(patch, nan=float(np.nanmin(patch)))
    refined_x, refined_y, intensity, _refined = _com_refine_patch(patch, xmin, ymin, qx, qy)
    return refined_x, refined_y, intensity, True


def _gaussian2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    sx = np.maximum(np.abs(sigma_x), 1e-6)
    sy = np.maximum(np.abs(sigma_y), 1e-6)
    return offset + amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sx**2) + ((y - y0) ** 2) / (2 * sy**2))
    )


def _com_refine_patch(
    patch: np.ndarray,
    xmin: int,
    ymin: int,
    qx: float,
    qy: float,
) -> tuple[float, float, float, bool]:
    weights = patch - float(np.min(patch))
    total = float(weights.sum())
    if total <= 0:
        return float(qx), float(qy), float(np.max(patch)), False
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    refined_x = float((weights * (xx + xmin)).sum() / total)
    refined_y = float((weights * (yy + ymin)).sum() / total)
    return refined_x, refined_y, float(np.max(patch)), False


def _valid_mask(shape: tuple[int, int], edge: int) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    if edge <= 0 or edge * 2 >= min(shape):
        return valid
    valid[:edge, :] = False
    valid[-edge:, :] = False
    valid[:, :edge] = False
    valid[:, -edge:] = False
    return valid


def _empty_peak_list() -> PeakList:
    return PeakList(
        qx=np.zeros(0, dtype=np.float64),
        qy=np.zeros(0, dtype=np.float64),
        intensity=np.zeros(0, dtype=np.float64),
        refined=np.zeros(0, dtype=bool),
    )
