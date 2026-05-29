"""File loaders for the first 4DLAB import workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from fourdlab.io.datacube import DataCube

HDF5_EXTENSIONS = {".h5", ".hdf5", ".emd", ".py4dstem"}
NUMPY_EXTENSIONS = {".npy"}
RAW_EXTENSIONS = {".raw"}
DEFAULT_RAW_DIFFRACTION_SHAPE = (130, 128)
DEFAULT_RAW_CROP_ROWS = 2
RAW_DTYPE = np.float32


class RawShapeError(ValueError):
    """Raised when a RAW file needs explicit import dimensions."""


@dataclass(frozen=True)
class RawLoadConfig:
    """Explicit dimensions and dtype for binary RAW datacube import."""

    scan_y: int
    scan_x: int
    diffraction_y: int
    diffraction_x: int
    dtype: str | np.dtype = "float32"
    crop_bottom_rows: int = 0

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def raw_shape(self) -> tuple[int, int, int, int]:
        return (
            int(self.scan_y),
            int(self.scan_x),
            int(self.diffraction_y),
            int(self.diffraction_x),
        )

    @property
    def expected_bytes(self) -> int:
        return int(np.prod(self.raw_shape)) * self.numpy_dtype.itemsize

    def validate(self, file_size: int) -> None:
        values = {
            "scan_y": self.scan_y,
            "scan_x": self.scan_x,
            "diffraction_y": self.diffraction_y,
            "diffraction_x": self.diffraction_x,
        }
        for name, value in values.items():
            if int(value) <= 0:
                raise RawShapeError(f"{name} must be positive.")
        if int(self.crop_bottom_rows) < 0:
            raise RawShapeError("crop_bottom_rows must be zero or positive.")
        if int(self.crop_bottom_rows) >= int(self.diffraction_y):
            raise RawShapeError("crop_bottom_rows must be smaller than diffraction_y.")
        if int(file_size) != self.expected_bytes:
            raise RawShapeError(
                "RAW dimensions do not match file size: "
                f"expected {self.expected_bytes} bytes, got {file_size} bytes."
            )


def load_datacube(
    path: str | Path,
    *,
    mmap: bool = True,
    raw_config: RawLoadConfig | None = None,
) -> DataCube:
    """Load a 4D-STEM datacube from a supported local file."""

    source_path = Path(path).expanduser().resolve()
    suffix = source_path.suffix.lower()
    if suffix in NUMPY_EXTENSIONS:
        mode = "r" if mmap else None
        data = np.load(source_path, mmap_mode=mode)
        return DataCube(data=data, source_path=source_path)

    if suffix in HDF5_EXTENSIONS:
        return _load_hdf5_datacube(source_path, mmap=mmap)

    if suffix in RAW_EXTENSIONS:
        return _load_raw_datacube(source_path, raw_config=raw_config)

    raise ValueError(
        "Unsupported file type "
        f"{suffix!r}. Supported: .npy, .raw, .h5, .hdf5, .emd, .py4dstem."
    )


def _load_hdf5_datacube(path: Path, *, mmap: bool) -> DataCube:
    file = h5py.File(path, "r")
    dataset = _find_best_4d_dataset(file)
    if dataset is None:
        file.close()
        paths = _summarize_datasets(file)
        suffix = "\nDatasets found:\n" + "\n".join(paths[:20]) if paths else ""
        raise ValueError(f"No 4D dataset found in {path}.{suffix}")

    data: Any = dataset if mmap else dataset[()]
    cube = DataCube(data=data, source_path=path, dataset_path=dataset.name)
    if not mmap:
        file.close()
    return cube


def _find_best_4d_dataset(group: h5py.Group) -> h5py.Dataset | None:
    candidates: list[h5py.Dataset] = []

    def visitor(_name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 4 and np.prod(obj.shape) > 0:
            candidates.append(obj)

    group.visititems(visitor)
    if not candidates:
        return None
    candidates.sort(key=_dataset_score, reverse=True)
    return candidates[0]


def _dataset_score(dataset: h5py.Dataset) -> tuple[int, int, int]:
    path = dataset.name.lower()
    score = 0
    if path.endswith("/data"):
        score += 20
    if "datacube" in path or "datacubes" in path:
        score += 20
    if "array" in path:
        score += 5
    if "calibration" in path or "metadata" in path:
        score -= 30
    if dataset.dtype.kind in "fiu":
        score += 10
    # Prefer scan axes first and detector axes last, common for py4DSTEM/4DLAB.
    sy, sx, qy, qx = (int(v) for v in dataset.shape)
    if sy <= qy * 4 and sx <= qx * 4:
        score += 5
    return score, int(np.prod(dataset.shape)), -len(path)


def _summarize_datasets(group: h5py.Group) -> list[str]:
    paths: list[str] = []

    def visitor(_name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(f"{obj.name} shape={tuple(obj.shape)} dtype={obj.dtype}")

    group.visititems(visitor)
    return paths


def _load_raw_datacube(path: Path, *, raw_config: RawLoadConfig | None = None) -> DataCube:
    file_size = path.stat().st_size
    if raw_config is None:
        qy, qx = DEFAULT_RAW_DIFFRACTION_SHAPE
        pixels_per_pattern = qy * qx
        itemsize = np.dtype(RAW_DTYPE).itemsize
        total_values = file_size // itemsize
        if total_values * itemsize != file_size:
            raise RawShapeError(f"RAW file size is not divisible by float32 item size: {path}.")
        if total_values % pixels_per_pattern != 0:
            raise RawShapeError(
                f"RAW file does not fit default diffraction shape {qy}x{qx}: {path}."
            )
        pattern_count = total_values // pixels_per_pattern
        scan_y, scan_x = _infer_scan_shape(pattern_count)
        raw_config = RawLoadConfig(
            scan_y=scan_y,
            scan_x=scan_x,
            diffraction_y=qy,
            diffraction_x=qx,
            dtype=np.dtype(RAW_DTYPE),
            crop_bottom_rows=DEFAULT_RAW_CROP_ROWS,
        )

    raw_config.validate(file_size)
    scan_y, scan_x, qy, qx = raw_config.raw_shape
    raw = np.memmap(path, dtype=raw_config.numpy_dtype, mode="r").reshape(scan_y, scan_x, qy, qx)
    crop_rows = int(raw_config.crop_bottom_rows)
    if crop_rows:
        data = raw[:, :, : qy - crop_rows, :]
        dataset_path = (
            f"raw:{scan_y}x{scan_x}x{qy}x{qx} dtype={raw_config.numpy_dtype}"
            f" cropped_to_{qy - crop_rows}x{qx}"
        )
    else:
        data = raw
        dataset_path = f"raw:{scan_y}x{scan_x}x{qy}x{qx} dtype={raw_config.numpy_dtype}"
    return DataCube(data=data, source_path=path, dataset_path=dataset_path)


def _infer_scan_shape(pattern_count: int) -> tuple[int, int]:
    side = int(np.sqrt(pattern_count))
    if side * side == pattern_count:
        return side, side
    return 1, pattern_count
