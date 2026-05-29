"""Small 4D-STEM data container used by the 4DLAB viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DataCube:
    """A portable wrapper around a 4D array with scan and diffraction axes."""

    data: Any
    source_path: Path
    dataset_path: str | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 4:
            raise ValueError(f"Expected a 4D dataset, got shape {self.shape!r}.")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def scan_shape(self) -> tuple[int, int]:
        return self.shape[:2]

    @property
    def diffraction_shape(self) -> tuple[int, int]:
        return self.shape[2:]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data.dtype)

    def diffraction_pattern(self, scan_y: int, scan_x: int) -> np.ndarray:
        """Return one diffraction pattern as a NumPy array."""

        scan_y = int(np.clip(scan_y, 0, self.scan_shape[0] - 1))
        scan_x = int(np.clip(scan_x, 0, self.scan_shape[1] - 1))
        return np.asarray(self.data[scan_y, scan_x, :, :])

    def diffraction_region(
        self,
        scan_y0: int,
        scan_x0: int,
        scan_y1: int,
        scan_x1: int,
        *,
        response: str = "mean",
    ) -> np.ndarray:
        """Return a diffraction image from a rectangular real-space region."""

        sy, sx = self.scan_shape
        y0 = int(np.clip(min(scan_y0, scan_y1), 0, sy - 1))
        x0 = int(np.clip(min(scan_x0, scan_x1), 0, sx - 1))
        y1 = int(np.clip(max(scan_y0, scan_y1), y0 + 1, sy))
        x1 = int(np.clip(max(scan_x0, scan_x1), x0 + 1, sx))
        values = np.asarray(self.data[y0:y1, x0:x1, :, :])
        if response == "max":
            return values.max(axis=(0, 1))
        if response == "sum":
            return values.sum(axis=(0, 1), dtype=np.float64)
        return values.mean(axis=(0, 1), dtype=np.float64)

    def virtual_image(
        self,
        center_y: float,
        center_x: float,
        radius_inner: float,
        radius_outer: float,
        *,
        detector_shape: str = "annulus",
        detector_response: str = "sum",
    ) -> np.ndarray:
        """Calculate a virtual image from a simple detector mask."""

        qy, qx = self.diffraction_shape
        yy, xx = np.ogrid[:qy, :qx]
        distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        if detector_shape == "point":
            y = int(np.clip(round(center_y), 0, qy - 1))
            x = int(np.clip(round(center_x), 0, qx - 1))
            return np.asarray(self.data[:, :, y, x])
        if detector_shape == "circle":
            mask = distance <= radius_outer
        elif detector_shape == "square":
            mask = (np.abs(yy - center_y) <= radius_outer) & (
                np.abs(xx - center_x) <= radius_outer
            )
        else:
            mask = (distance >= radius_inner) & (distance <= radius_outer)

        if not np.any(mask):
            return np.zeros(self.scan_shape, dtype=np.float32)

        values = np.asarray(self.data[:, :, mask])
        if detector_response == "mean":
            return values.mean(axis=-1, dtype=np.float64)
        if detector_response == "max":
            return values.max(axis=-1)
        return values.sum(axis=-1, dtype=np.float64)

    def virtual_image_chunked(
        self,
        center_y: float,
        center_x: float,
        radius_inner: float,
        radius_outer: float,
        *,
        detector_shape: str = "annulus",
        detector_response: str = "sum",
        scan_chunk_rows: int = 8,
    ) -> np.ndarray:
        """Calculate a virtual image in scan-row chunks.

        This avoids the large temporary allocation created by
        ``data[:, :, mask]`` on file-backed or memory-mapped datasets.
        """

        qy, qx = self.diffraction_shape
        yy, xx = np.ogrid[:qy, :qx]
        distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        if detector_shape == "point":
            y = int(np.clip(round(center_y), 0, qy - 1))
            x = int(np.clip(round(center_x), 0, qx - 1))
            return np.asarray(self.data[:, :, y, x])
        if detector_shape == "circle":
            mask = distance <= radius_outer
        elif detector_shape == "square":
            mask = (np.abs(yy - center_y) <= radius_outer) & (
                np.abs(xx - center_x) <= radius_outer
            )
        else:
            mask = (distance >= radius_inner) & (distance <= radius_outer)
        if not np.any(mask):
            return np.zeros(self.scan_shape, dtype=np.float32)

        ys, xs = np.nonzero(mask)
        qy0, qy1 = int(ys.min()), int(ys.max()) + 1
        qx0, qx1 = int(xs.min()), int(xs.max()) + 1
        mask_sub = mask[qy0:qy1, qx0:qx1]
        sy, sx = self.scan_shape
        out = np.zeros((sy, sx), dtype=np.float64)
        chunk = max(1, int(scan_chunk_rows))
        for y0 in range(0, sy, chunk):
            y1 = min(sy, y0 + chunk)
            values = np.asarray(self.data[y0:y1, :, qy0:qy1, qx0:qx1])
            selected = values[:, :, mask_sub]
            if detector_response == "mean":
                out[y0:y1] = selected.mean(axis=-1, dtype=np.float64)
            elif detector_response == "max":
                out[y0:y1] = selected.max(axis=-1)
            else:
                out[y0:y1] = selected.sum(axis=-1, dtype=np.float64)
        return out

    def quick_navigation_image(self, center_y: float | None = None, center_x: float | None = None) -> np.ndarray:
        """Return a cheap real-space navigation image from one detector pixel."""

        qy, qx = self.diffraction_shape
        y = int(np.clip(round((qy - 1) / 2.0 if center_y is None else center_y), 0, qy - 1))
        x = int(np.clip(round((qx - 1) / 2.0 if center_x is None else center_x), 0, qx - 1))
        return np.asarray(self.data[:, :, y, x])

    def close(self) -> None:
        """Release file-backed resources when the datacube is no longer needed."""

        mmap = getattr(self.data, "_mmap", None)
        if mmap is not None:
            mmap.close()
        file = getattr(self.data, "file", None)
        if file is not None:
            try:
                file.close()
            except Exception:
                pass
