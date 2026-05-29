"""Optional CuPy acceleration for native template matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fourdlab.processing.native_matching import NativeFastMatch, PreparedNativeTemplateBank


@dataclass
class PreparedNativeTemplateGpuBank:
    templates_qxy: object
    templates_weight: object
    templates_mask: object
    template_count: int
    max_template_peaks: int


def native_gpu_available() -> bool:
    """Return whether CuPy and a CUDA device are available."""

    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def prepare_native_template_bank_gpu(bank: PreparedNativeTemplateBank) -> PreparedNativeTemplateGpuBank:
    """Create padded GPU template arrays from the CPU flat bank."""

    import cupy as cp

    max_peaks = int(np.max(bank.template_peak_count)) if bank.template_peak_count.size else 0
    max_peaks = max(max_peaks, 1)
    qxy = np.zeros((bank.template_count, max_peaks, 2), dtype=np.float32)
    weight = np.zeros((bank.template_count, max_peaks), dtype=np.float32)
    mask = np.zeros((bank.template_count, max_peaks), dtype=bool)
    for flat_idx, template_id in enumerate(bank.flat_template_id):
        peak_idx = int(bank.flat_template_peak_index[flat_idx])
        qxy[int(template_id), peak_idx] = bank.flat_qxy[flat_idx]
        weight[int(template_id), peak_idx] = bank.flat_weight[flat_idx]
        mask[int(template_id), peak_idx] = True
    return PreparedNativeTemplateGpuBank(
        templates_qxy=cp.asarray(qxy),
        templates_weight=cp.asarray(weight),
        templates_mask=cp.asarray(mask),
        template_count=bank.template_count,
        max_template_peaks=max_peaks,
    )


def match_prepared_template_bank_gpu(
    gpu_bank: PreparedNativeTemplateGpuBank,
    exp_qxy: np.ndarray,
    exp_intensity: np.ndarray,
    *,
    tolerance_q: float,
    min_number_peaks: int,
    num_matches: int,
    chunk_size: int = 4096,
) -> NativeFastMatch:
    """Match templates on GPU in chunks and return CPU top-N arrays."""

    import cupy as cp

    topn = max(1, int(num_matches))
    if exp_qxy.size == 0:
        return _empty(topn)
    exp = cp.asarray(np.asarray(exp_qxy, dtype=np.float32))
    exp_weight = cp.asarray(_sqrt_weights(exp_intensity).astype(np.float32))
    scores = cp.zeros(gpu_bank.template_count, dtype=cp.float32)
    counts = cp.zeros(gpu_bank.template_count, dtype=cp.int32)
    residual_sum = cp.zeros(gpu_bank.template_count, dtype=cp.float32)
    tol = max(float(tolerance_q), 1.0e-12)
    chunk = max(1, int(chunk_size))

    for start in range(0, gpu_bank.template_count, chunk):
        end = min(gpu_bank.template_count, start + chunk)
        templates = gpu_bank.templates_qxy[start:end]
        weights = gpu_bank.templates_weight[start:end]
        mask = gpu_bank.templates_mask[start:end]
        diff = templates[:, :, None, :] - exp[None, None, :, :]
        dist2 = cp.sum(diff * diff, axis=3)
        best_idx = cp.argmin(dist2, axis=2)
        best_dist2 = cp.take_along_axis(dist2, best_idx[:, :, None], axis=2)[:, :, 0]
        valid = mask & (best_dist2 <= tol * tol)
        closeness = cp.exp(-best_dist2 / (2.0 * tol * tol)) * valid
        matched_exp_weight = exp_weight[best_idx]
        terms = closeness * weights * matched_exp_weight
        local_scores = cp.sum(terms, axis=1)
        local_counts = cp.sum(valid, axis=1).astype(cp.int32)
        local_residual = cp.sum(cp.sqrt(cp.maximum(best_dist2, 0.0)) * valid, axis=1)
        penalty = cp.minimum(local_counts / max(float(min_number_peaks), 1.0), 1.0)
        local_scores = local_scores * penalty
        scores[start:end] = local_scores
        counts[start:end] = local_counts
        residual_sum[start:end] = local_residual

    candidate_count = min(topn, gpu_bank.template_count)
    order = cp.argsort(scores)[::-1][:candidate_count]
    template_indices = np.full(topn, -1, dtype=np.int32)
    top_scores = np.full(topn, np.nan, dtype=np.float64)
    top_counts = np.zeros(topn, dtype=np.int32)
    top_mean = np.full(topn, np.nan, dtype=np.float64)
    order_cpu = cp.asnumpy(order).astype(np.int32)
    scores_cpu = cp.asnumpy(scores[order]).astype(np.float64)
    counts_cpu = cp.asnumpy(counts[order]).astype(np.int32)
    residual_cpu = cp.asnumpy(residual_sum[order]).astype(np.float64)
    for rank, template_id in enumerate(order_cpu):
        template_indices[rank] = int(template_id)
        top_scores[rank] = float(scores_cpu[rank])
        top_counts[rank] = int(counts_cpu[rank])
        top_mean[rank] = float(residual_cpu[rank] / max(counts_cpu[rank], 1)) if counts_cpu[rank] else float("nan")
    return NativeFastMatch(
        template_indices=template_indices,
        scores=top_scores,
        matched_peak_counts=top_counts,
        mean_residual_q=top_mean,
        median_residual_q=top_mean.copy(),
    )


def _sqrt_weights(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.sqrt(np.clip(np.nan_to_num(arr, nan=0.0), 0.0, None))
    total = float(np.sum(arr))
    if total <= 0:
        return np.ones_like(arr) / max(arr.size, 1)
    return arr / total


def _empty(topn: int) -> NativeFastMatch:
    return NativeFastMatch(
        template_indices=np.full(topn, -1, dtype=np.int32),
        scores=np.full(topn, np.nan, dtype=np.float64),
        matched_peak_counts=np.zeros(topn, dtype=np.int32),
        mean_residual_q=np.full(topn, np.nan, dtype=np.float64),
        median_residual_q=np.full(topn, np.nan, dtype=np.float64),
    )
