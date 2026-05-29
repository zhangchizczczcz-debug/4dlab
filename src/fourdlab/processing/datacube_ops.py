"""Datacube cropping and binning utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fourdlab.io import DataCube


def crop_and_bin_datacube(
    cube: DataCube,
    *,
    scan_bounds: tuple[int, int, int, int] | None = None,
    diffraction_bounds: tuple[int, int, int, int] | None = None,
    scan_bin: tuple[int, int] = (1, 1),
    diffraction_bin: tuple[int, int] = (1, 1),
) -> DataCube:
    """Return a cropped/binned in-memory datacube."""

    sy, sx = cube.scan_shape
    qy, qx = cube.diffraction_shape
    scan_y0, scan_x0, scan_y1, scan_x1 = _bounds(scan_bounds, sy, sx)
    q_y0, q_x0, q_y1, q_x1 = _bounds(diffraction_bounds, qy, qx)

    data = np.asarray(cube.data[scan_y0:scan_y1, scan_x0:scan_x1, q_y0:q_y1, q_x0:q_x1])
    data = _bin_4d(data, scan_bin=scan_bin, diffraction_bin=diffraction_bin)
    dataset_path = (
        f"cropbin:scan[{scan_y0}:{scan_y1},{scan_x0}:{scan_x1}]"
        f":diff[{q_y0}:{q_y1},{q_x0}:{q_x1}]"
        f":bin{tuple(scan_bin)}x{tuple(diffraction_bin)}"
    )
    return DataCube(
        data=data,
        source_path=Path(cube.source_path),
        dataset_path=dataset_path,
    )


def _bounds(bounds: tuple[int, int, int, int] | None, y_size: int, x_size: int) -> tuple[int, int, int, int]:
    if bounds is None:
        return 0, 0, int(y_size), int(x_size)
    y0, x0, y1, x1 = (int(v) for v in bounds)
    y0 = int(np.clip(y0, 0, y_size - 1))
    x0 = int(np.clip(x0, 0, x_size - 1))
    y1 = int(np.clip(y1, y0 + 1, y_size))
    x1 = int(np.clip(x1, x0 + 1, x_size))
    return y0, x0, y1, x1


def _bin_4d(
    data: np.ndarray,
    *,
    scan_bin: tuple[int, int],
    diffraction_bin: tuple[int, int],
) -> np.ndarray:
    sy_bin = max(1, int(scan_bin[0]))
    sx_bin = max(1, int(scan_bin[1]))
    qy_bin = max(1, int(diffraction_bin[0]))
    qx_bin = max(1, int(diffraction_bin[1]))
    out = np.asarray(data)
    out = _bin_axis_pair(out, 0, 1, sy_bin, sx_bin)
    out = _bin_axis_pair(out, 2, 3, qy_bin, qx_bin)
    return np.asarray(out)


def _bin_axis_pair(data: np.ndarray, axis_y: int, axis_x: int, bin_y: int, bin_x: int) -> np.ndarray:
    if bin_y == 1 and bin_x == 1:
        return data
    shape = list(data.shape)
    new_y = shape[axis_y] // bin_y
    new_x = shape[axis_x] // bin_x
    if new_y < 1 or new_x < 1:
        raise ValueError("Bin size is larger than the selected region.")
    slices = [slice(None)] * data.ndim
    slices[axis_y] = slice(0, new_y * bin_y)
    slices[axis_x] = slice(0, new_x * bin_x)
    trimmed = data[tuple(slices)]
    moved = np.moveaxis(trimmed, (axis_y, axis_x), (0, 1))
    binned = moved.reshape(new_y, bin_y, new_x, bin_x, *moved.shape[2:]).mean(axis=(1, 3))
    return np.moveaxis(binned, (0, 1), (axis_y, axis_x))
