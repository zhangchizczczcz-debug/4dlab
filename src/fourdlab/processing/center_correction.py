"""Center-disk correction for nanobeam diffraction datacubes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import shift as ndi_shift

from fourdlab.io import DataCube


ProgressCallback = Callable[[int, int], None]
StopCallback = Callable[[], bool]


class CenterCorrectionCancelled(RuntimeError):
    """Raised when center correction is cancelled by the caller."""


@dataclass(frozen=True)
class CenterCorrectionConfig:
    """Options for aligning the center disk in every diffraction pattern."""

    mask: np.ndarray
    target_y: float
    target_x: float
    subpixel: bool = True
    order: int = 1


@dataclass
class CenterCorrectionResult:
    """Output of center correction."""

    cube: DataCube
    shifts_y: np.ndarray
    shifts_x: np.ndarray
    target_y: float
    target_x: float


def make_detector_mask(
    diffraction_shape: tuple[int, int],
    *,
    detector_shape: str,
    center_y: float,
    center_x: float,
    radius_inner: float,
    radius_outer: float,
) -> np.ndarray:
    """Create a boolean mask from the viewer's diffraction-space detector ROI."""

    qy, qx = diffraction_shape
    yy, xx = np.ogrid[:qy, :qx]
    distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    if detector_shape == "point":
        mask = np.zeros(diffraction_shape, dtype=bool)
        y = int(np.clip(round(center_y), 0, qy - 1))
        x = int(np.clip(round(center_x), 0, qx - 1))
        mask[max(0, y - 1) : min(qy, y + 2), max(0, x - 1) : min(qx, x + 2)] = True
        return mask
    if detector_shape == "circle":
        return distance <= radius_outer
    if detector_shape == "square":
        return (np.abs(yy - center_y) <= radius_outer) & (
            np.abs(xx - center_x) <= radius_outer
        )
    return (distance >= radius_inner) & (distance <= radius_outer)


def center_of_mass_from_mask(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return y/x COM inside a mask after local background subtraction."""

    arr = np.asarray(image, dtype=np.float64)
    masked = np.where(mask, arr, np.nan)
    finite = np.isfinite(masked)
    if not finite.any():
        return _mask_geometric_center(mask)

    baseline = np.nanmin(masked)
    weights = np.where(mask, arr - baseline, 0.0)
    weights = np.nan_to_num(weights, copy=False)
    total = float(weights.sum())
    if total <= 0:
        return _mask_geometric_center(mask)

    yy, xx = np.indices(arr.shape, dtype=np.float64)
    center_y = float((weights * yy).sum() / total)
    center_x = float((weights * xx).sum() / total)
    return center_y, center_x


def align_center_disks(
    cube: DataCube,
    config: CenterCorrectionConfig,
    *,
    progress: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> CenterCorrectionResult:
    """Align the selected center disk in every DP to the target position."""

    data = np.asarray(cube.data)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D datacube, got shape {data.shape!r}.")

    mask = np.asarray(config.mask, dtype=bool)
    if mask.shape != tuple(data.shape[2:]):
        raise ValueError(f"Mask shape {mask.shape!r} does not match DP shape {data.shape[2:]!r}.")
    if not mask.any():
        raise ValueError("Center correction mask is empty.")

    out_dtype = np.float32 if config.subpixel else data.dtype
    aligned = np.empty(data.shape, dtype=out_dtype)
    shifts_y = np.zeros(data.shape[:2], dtype=np.float32)
    shifts_x = np.zeros(data.shape[:2], dtype=np.float32)
    total = data.shape[0] * data.shape[1]
    done = 0

    for scan_y in range(data.shape[0]):
        for scan_x in range(data.shape[1]):
            if stop_requested is not None and stop_requested():
                raise CenterCorrectionCancelled(
                    f"Center correction cancelled at scan ({scan_y}, {scan_x})."
                )
            dp = np.asarray(data[scan_y, scan_x], dtype=np.float32)
            center_y, center_x = center_of_mass_from_mask(dp, mask)
            dy = float(config.target_y - center_y)
            dx = float(config.target_x - center_x)
            shifts_y[scan_y, scan_x] = dy
            shifts_x[scan_y, scan_x] = dx
            if config.subpixel:
                aligned[scan_y, scan_x] = ndi_shift(
                    dp,
                    shift=(dy, dx),
                    order=int(config.order),
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
            else:
                aligned[scan_y, scan_x] = _shift_integer_zero_fill(
                    dp,
                    shift_y=int(round(dy)),
                    shift_x=int(round(dx)),
                )
            done += 1
            if progress is not None:
                progress(done, total)
            if stop_requested is not None and stop_requested():
                raise CenterCorrectionCancelled(
                    f"Center correction cancelled after {done}/{total} patterns."
                )

    result_cube = DataCube(
        data=aligned,
        source_path=Path(cube.source_path),
        dataset_path="center_corrected",
    )
    return CenterCorrectionResult(
        cube=result_cube,
        shifts_y=shifts_y,
        shifts_x=shifts_x,
        target_y=float(config.target_y),
        target_x=float(config.target_x),
    )


def _mask_geometric_center(mask: np.ndarray) -> tuple[float, float]:
    points = np.argwhere(mask)
    if points.size == 0:
        raise ValueError("Cannot calculate center of an empty mask.")
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _shift_integer_zero_fill(image: np.ndarray, *, shift_y: int, shift_x: int) -> np.ndarray:
    """Integer-pixel shift with zero-filled exposed edges."""

    out = np.zeros_like(image)
    height, width = image.shape

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(height, height + shift_y)

    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(width, width + shift_x)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out
