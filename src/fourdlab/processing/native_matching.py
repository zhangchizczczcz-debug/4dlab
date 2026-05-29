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


@dataclass
class PreparedNativeTemplateBank:
    """Flat template bank used for one-query-per-pixel CPU matching."""

    flat_qxy: np.ndarray
    flat_weight: np.ndarray
    flat_template_id: np.ndarray
    flat_template_peak_index: np.ndarray
    template_count: int
    template_peak_count: np.ndarray


@dataclass
class NativeFastMatch:
    """Top-N native template matches from a prepared template bank."""

    template_indices: np.ndarray
    scores: np.ndarray
    matched_peak_counts: np.ndarray
    mean_residual_q: np.ndarray
    median_residual_q: np.ndarray


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


def prepare_native_template_bank(
    templates,
    *,
    intensity_weight_mode: Literal["uniform", "raw", "sqrt", "log"] = "sqrt",
) -> PreparedNativeTemplateBank:
    """Flatten template peaks once for fast repeated matching."""

    qxy_parts = []
    weight_parts = []
    template_id_parts = []
    peak_index_parts = []
    peak_counts = np.zeros(len(templates), dtype=np.int32)
    for template_id, template in enumerate(templates):
        qxy = np.asarray(template.qxy, dtype=np.float64)
        weights = _weights(np.asarray(template.intensity, dtype=np.float64), intensity_weight_mode)
        count = int(qxy.shape[0])
        peak_counts[template_id] = count
        if count == 0:
            continue
        qxy_parts.append(qxy)
        weight_parts.append(weights)
        template_id_parts.append(np.full(count, template_id, dtype=np.int32))
        peak_index_parts.append(np.arange(count, dtype=np.int32))
    if not qxy_parts:
        return PreparedNativeTemplateBank(
            flat_qxy=np.zeros((0, 2), dtype=np.float64),
            flat_weight=np.zeros(0, dtype=np.float64),
            flat_template_id=np.zeros(0, dtype=np.int32),
            flat_template_peak_index=np.zeros(0, dtype=np.int32),
            template_count=len(templates),
            template_peak_count=peak_counts,
        )
    return PreparedNativeTemplateBank(
        flat_qxy=np.concatenate(qxy_parts, axis=0),
        flat_weight=np.concatenate(weight_parts, axis=0),
        flat_template_id=np.concatenate(template_id_parts, axis=0),
        flat_template_peak_index=np.concatenate(peak_index_parts, axis=0),
        template_count=len(templates),
        template_peak_count=peak_counts,
    )


def match_prepared_template_bank_cpu(
    bank: PreparedNativeTemplateBank,
    exp_qxy: np.ndarray,
    exp_intensity: np.ndarray,
    *,
    tolerance_q: float,
    min_number_peaks: int,
    num_matches: int,
    intensity_weight_mode: Literal["uniform", "raw", "sqrt", "log"] = "sqrt",
) -> NativeFastMatch:
    """Match all templates to one peak list with one KDTree query."""

    topn = max(1, int(num_matches))
    empty = _empty_fast(topn)
    exp_qxy = np.asarray(exp_qxy, dtype=np.float64)
    if bank.flat_qxy.size == 0 or exp_qxy.size == 0:
        return empty
    tree = cKDTree(exp_qxy)
    distances, indices = tree.query(bank.flat_qxy, k=1, distance_upper_bound=float(tolerance_q))
    valid = np.isfinite(distances) & (indices < exp_qxy.shape[0])
    if not np.any(valid):
        return empty

    valid_template_id = bank.flat_template_id[valid]
    valid_distances = distances[valid].astype(np.float64)
    exp_weights = _weights(exp_intensity, intensity_weight_mode)
    closeness = np.exp(-(valid_distances**2) / (2.0 * max(float(tolerance_q), 1.0e-12) ** 2))
    terms = closeness * bank.flat_weight[valid] * exp_weights[indices[valid]]
    scores = np.bincount(valid_template_id, weights=terms, minlength=bank.template_count)
    counts = np.bincount(valid_template_id, minlength=bank.template_count).astype(np.int32)
    residual_sum = np.bincount(valid_template_id, weights=valid_distances, minlength=bank.template_count)
    mean_residual = residual_sum / np.maximum(counts, 1)
    scores = scores.copy()
    too_few = counts < int(min_number_peaks)
    scores[too_few] *= counts[too_few] / max(float(min_number_peaks), 1.0)

    candidate_count = min(topn, bank.template_count)
    if candidate_count == bank.template_count:
        order = np.argsort(scores)[::-1]
    else:
        order = np.argpartition(scores, -candidate_count)[-candidate_count:]
        order = order[np.argsort(scores[order])[::-1]]

    template_indices = np.full(topn, -1, dtype=np.int32)
    top_scores = np.full(topn, np.nan, dtype=np.float64)
    top_counts = np.zeros(topn, dtype=np.int32)
    top_mean = np.full(topn, np.nan, dtype=np.float64)
    top_median = np.full(topn, np.nan, dtype=np.float64)
    for rank, template_id in enumerate(order[:topn]):
        template_indices[rank] = int(template_id)
        top_scores[rank] = float(scores[template_id])
        top_counts[rank] = int(counts[template_id])
        top_mean[rank] = float(mean_residual[template_id]) if counts[template_id] else float("nan")
        mask = valid_template_id == template_id
        top_median[rank] = float(np.median(valid_distances[mask])) if np.any(mask) else float("nan")
    return NativeFastMatch(
        template_indices=template_indices,
        scores=top_scores,
        matched_peak_counts=top_counts,
        mean_residual_q=top_mean,
        median_residual_q=top_median,
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


def _empty_fast(topn: int) -> NativeFastMatch:
    return NativeFastMatch(
        template_indices=np.full(topn, -1, dtype=np.int32),
        scores=np.full(topn, np.nan, dtype=np.float64),
        matched_peak_counts=np.zeros(topn, dtype=np.int32),
        mean_residual_q=np.full(topn, np.nan, dtype=np.float64),
        median_residual_q=np.full(topn, np.nan, dtype=np.float64),
    )


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
