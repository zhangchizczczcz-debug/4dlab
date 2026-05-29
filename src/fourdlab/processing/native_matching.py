"""Sparse native template matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SparseMatchResult:
    score: float
    matched_peak_count: int
    mean_residual_q: float
    median_residual_q: float
    matched_template_indices: np.ndarray
    matched_experiment_indices: np.ndarray
    residuals_q: np.ndarray


def match_sparse_template(
    template_qxy: np.ndarray,
    template_intensity: np.ndarray,
    exp_qxy: np.ndarray,
    exp_intensity: np.ndarray,
    *,
    tolerance_q: float,
    min_number_peaks: int,
    intensity_weight_mode: Literal["uniform", "raw", "sqrt", "log"] = "sqrt",
) -> SparseMatchResult:
    """Match one sparse template to one experimental peak set."""

    template_qxy = np.asarray(template_qxy, dtype=np.float64)
    exp_qxy = np.asarray(exp_qxy, dtype=np.float64)
    if template_qxy.size == 0 or exp_qxy.size == 0:
        return _empty_result()
    tree = cKDTree(exp_qxy)
    distances, indices = tree.query(template_qxy, k=1, distance_upper_bound=float(tolerance_q))
    valid = np.isfinite(distances) & (indices < exp_qxy.shape[0])
    if not np.any(valid):
        return _empty_result()
    residuals = distances[valid].astype(np.float64)
    template_indices = np.flatnonzero(valid).astype(np.int64)
    exp_indices = indices[valid].astype(np.int64)
    closeness = np.exp(-(residuals**2) / (2.0 * max(float(tolerance_q), 1.0e-12) ** 2))
    template_weights = _weights(template_intensity, intensity_weight_mode)
    exp_weights = _weights(exp_intensity, intensity_weight_mode)
    terms = closeness * template_weights[template_indices] * exp_weights[exp_indices]
    denom = max(float(np.sum(template_weights)), 1.0e-12)
    score = float(np.sum(terms) / denom)
    matched_count = int(template_indices.size)
    if matched_count < int(min_number_peaks):
        score *= matched_count / max(float(min_number_peaks), 1.0)
    return SparseMatchResult(
        score=score,
        matched_peak_count=matched_count,
        mean_residual_q=float(np.mean(residuals)),
        median_residual_q=float(np.median(residuals)),
        matched_template_indices=template_indices,
        matched_experiment_indices=exp_indices,
        residuals_q=residuals,
    )


def _weights(values: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    if mode == "uniform":
        out = np.ones_like(arr)
    elif mode == "raw":
        out = arr
    elif mode == "log":
        out = np.log1p(arr)
    else:
        out = np.sqrt(arr)
    if out.size == 0:
        return out
    total = float(np.sum(out))
    if total <= 0:
        return np.ones_like(out) / float(out.size)
    return out / total


def _empty_result() -> SparseMatchResult:
    return SparseMatchResult(
        score=0.0,
        matched_peak_count=0,
        mean_residual_q=float("nan"),
        median_residual_q=float("nan"),
        matched_template_indices=np.zeros(0, dtype=np.int64),
        matched_experiment_indices=np.zeros(0, dtype=np.int64),
        residuals_q=np.zeros(0, dtype=np.float64),
    )
