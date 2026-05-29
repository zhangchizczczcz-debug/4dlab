"""Export 4DLAB datacubes to portable file formats."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from fourdlab.io.datacube import DataCube


def export_datacube(cube: DataCube, path: str | Path) -> Path:
    """Export a datacube based on the output file extension."""

    out = Path(path).expanduser().resolve()
    suffix = out.suffix.lower()
    if suffix == ".npy":
        export_npy(cube, out)
    elif suffix == ".raw":
        export_raw(cube, out)
    elif suffix in {".h5", ".hdf5", ".emd", ".py4dstem"}:
        export_hdf5(cube, out)
    else:
        raise ValueError("Export path must end with .npy, .raw, .h5, .hdf5, .emd, or .py4dstem.")
    return out


def export_npy(cube: DataCube, path: Path, *, scan_chunk_rows: int = 8) -> None:
    """Write a datacube to `.npy` without forcing a full RAM copy."""

    arr = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=cube.dtype,
        shape=cube.shape,
    )
    _copy_scan_chunks(cube, arr, scan_chunk_rows=scan_chunk_rows)
    arr.flush()


def export_raw(cube: DataCube, path: Path, *, scan_chunk_rows: int = 8) -> None:
    """Write raw binary data plus a JSON sidecar with shape/dtype metadata."""

    with path.open("wb") as handle:
        for chunk in _iter_scan_chunks(cube, scan_chunk_rows=scan_chunk_rows):
            contiguous = np.ascontiguousarray(chunk)
            handle.write(contiguous.tobytes(order="C"))
    metadata = {
        "shape": list(cube.shape),
        "dtype": str(cube.dtype),
        "order": "C",
        "source_path": str(cube.source_path),
        "dataset_path": cube.dataset_path,
        "note": "RAW files have no embedded shape metadata; keep this sidecar with the .raw file.",
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def export_hdf5(cube: DataCube, path: Path, *, scan_chunk_rows: int = 8) -> None:
    """Write a simple HDF5/EMD-compatible datacube at `/datacube/data`."""

    with h5py.File(path, "w") as file:
        root = file.create_group("datacube")
        dataset = root.create_dataset(
            "data",
            shape=cube.shape,
            dtype=cube.dtype,
            chunks=_hdf5_chunks(cube.shape),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        dataset.attrs["fourdlab_role"] = "datacube"
        dataset.attrs["source_path"] = str(cube.source_path)
        if cube.dataset_path:
            dataset.attrs["source_dataset_path"] = cube.dataset_path
        root.attrs["emd_group_type"] = "array"
        root.attrs["python_class"] = "fourdlab.io.DataCube"
        _copy_scan_chunks(cube, dataset, scan_chunk_rows=scan_chunk_rows)


def _copy_scan_chunks(cube: DataCube, target, *, scan_chunk_rows: int) -> None:
    y0 = 0
    for chunk in _iter_scan_chunks(cube, scan_chunk_rows=scan_chunk_rows):
        y1 = y0 + chunk.shape[0]
        target[y0:y1, :, :, :] = chunk
        y0 = y1


def _iter_scan_chunks(cube: DataCube, *, scan_chunk_rows: int):
    sy, _sx, _qy, _qx = cube.shape
    rows = max(1, int(scan_chunk_rows))
    for y0 in range(0, sy, rows):
        y1 = min(sy, y0 + rows)
        yield np.asarray(cube.data[y0:y1, :, :, :])


def _hdf5_chunks(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    sy, sx, qy, qx = shape
    return min(sy, 4), min(sx, 16), min(qy, 64), min(qx, 64)
