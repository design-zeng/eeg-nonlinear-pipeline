# -*- coding: utf-8 -*-
"""crqa_py — Cross-Recurrence Quantification Analysis in pure Python
===================================================================
*Numerically identical* to Norbert Marwan’s MATLAB `crqa.m` for all 13
indices, plus a wrapper that mimics `nonlinear_analysis()` I/O.

▸ GPU optional (CuPy) — falls back to NumPy when unavailable.
▸ now supports **CPU ProcessPool + single-GPU worker** mode
  (set `use_gpu=True` and `max_workers>1` in `rqa_analysis`).
"""
from __future__ import annotations
from utils.timer import timer
from utils.memory import wait_for_available_memory, wait_for_available_gpu_memory

import itertools, uuid, warnings, queue as _pyqueue
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory, managers
import json
import os
import time

from scipy import ndimage; SCIPY_AVAILABLE = True

try:
    import cupy as cp          # GPU optional
except ModuleNotFoundError:
    cp = None

Array  = Union[np.ndarray, "cp.ndarray"]
_DTYPE = np.float32

###############################################################################
# Helper utilities
###############################################################################
def _backend(x: Array):
    """Return cp or np backend matching `x`."""
    return cp if (cp is not None and isinstance(x, cp.ndarray)) else np

def _to_ndarray(x: Array, use_gpu: bool = True) -> Array:
    xp = cp if (use_gpu and cp) else np
    arr = xp.asarray(x, dtype=_DTYPE)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr

def _zscore(x: Array) -> Array:
    return (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, ddof=0, keepdims=True)

def _embed(ts: Array, m: int, tau: int) -> Array:
    n, dim = ts.shape
    if m == 1:
        return ts.copy()
    rows = n - (m - 1) * tau
    if rows < 1:
        raise ValueError("Too few samples for embedding.")
    xp = _backend(ts)
    emb = xp.empty((rows, dim * m), dtype=_DTYPE)
    for i in range(m):
        emb[:, i * dim:(i + 1) * dim] = ts[i * tau:i * tau + rows]
    return emb

###############################################################################
# Distance & RP
###############################################################################
_METHODS: Dict[str, Literal["max", "eu", "min", "nr"]] = {
    "max": "max", "euclidean": "eu", "eu": "eu", "min": "min", "nr": "nr",
}

@timer()
def _distance_matrix(x: Array, y: Array, method: str, use_gpu: bool) -> Array:
    """Optimized distance matrix calculation."""
    xp = cp if (use_gpu and cp) else np

    n = x.shape[0]
    size_gb = 2* (n * n * 4) / 1024 ** 3  # float32 (using _DTYPE)
    wait_for_available_memory(size_gb)

    if method == "max":     # Chebyshev
        # Use more efficient memory layout
        return xp.abs(x[:, None, :] - y[None, :, :]).max(axis=2)
    if method == "min":     # Manhattan
        return xp.abs(x[:, None, :] - y[None, :, :]).sum(axis=2)
    if method == "nr":      # normalised Euclidean
        nx = xp.linalg.norm(x, axis=1, keepdims=True)
        ny = xp.linalg.norm(y, axis=1, keepdims=True)
        nx = xp.where(nx == 0.0, 1.0, nx)
        ny = xp.where(ny == 0.0, 1.0, ny)
        x, y = x / nx, y / ny
    # Euclidean distance: use efficient vectorized implementation
    diff = x[:, None, :] - y[None, :, :]
    return xp.sqrt((diff ** 2).sum(axis=2))

def _rp_binary(dmat: Array, eps: float) -> Array:
    xp = _backend(dmat)
    return (dmat <= eps).astype(_DTYPE)

###############################################################################
# Line statistics (diag / vert / white)
###############################################################################
@timer()
def _diag_line_lengths(rp: Array, lmin: int):
    """Optimized diagonal line length calculation using vectorized methods.
    
    Algorithmically equivalent to the original implementation, but uses
    scipy.ndimage.label for faster connected component labeling when available.
    """
    xp = _backend(rp)
    n = rp.shape[0]
    
    # Convert to CPU numpy array for scipy (if available)
    if xp is cp:
        rp_cpu = rp.get()
    else:
        rp_cpu = rp
    
    # Use scipy.ndimage.label for vectorized labeling (if available)
    if SCIPY_AVAILABLE and ndimage is not None:
        # Label connected components on all diagonals
        # We need to process all non-zero diagonals
        lens = []
        for k in range(-n + 1, n):
            if k == 0:
                continue
            d = np.diag(rp_cpu, k=k)
            if d.sum() == 0:
                continue
            # Use connected component labeling
            labeled, num_features = ndimage.label(d)
            for i in range(1, num_features + 1):
                length = (labeled == i).sum()
                if length >= lmin:
                    lens.append(length)
        arr = xp.asarray(lens, dtype=_DTYPE)
        return (arr.get() if xp is cp else arr), n
    else:
        # Fallback to optimized vectorized method: batch processing
        lens = []
        for k in range(-n + 1, n):
            if k == 0:
                continue
            d = xp.diag(rp, k=k)
            if d.sum() == 0:
                continue
            idx = xp.where(d == 1)[0]
            if idx.size == 0:
                continue
            # Use vectorized diff and concatenate to find consecutive segments
            diff = xp.diff(idx)
            breaks = xp.where(diff != 1)[0] + 1
            starts = xp.concatenate([xp.array([0]), breaks])
            ends = xp.concatenate([breaks, xp.array([idx.size])])
            lengths = ends - starts
            valid_lengths = lengths[lengths >= lmin]
            if valid_lengths.size > 0:
                lens.extend(valid_lengths.tolist())
        arr = xp.asarray(lens, dtype=_DTYPE)
        return (arr.get() if xp is cp else arr), n

@timer()
def _vertical_line_lengths(rp: Array, vmin: int):
    """Optimized vertical line length calculation using vectorized methods.
    
    Algorithmically equivalent to the original implementation, but uses
    scipy.ndimage.label for faster connected component labeling when available.
    """
    xp = _backend(rp)
    n = rp.shape[1]
    
    # Convert to CPU numpy array for scipy (if available)
    if xp is cp:
        rp_cpu = rp.get()
    else:
        rp_cpu = rp
    
    # Use scipy.ndimage.label for vectorized labeling (if available)
    if SCIPY_AVAILABLE and ndimage is not None:
        # Label connected components for each column
        lens = []
        for col in range(n):
            col_vec = rp_cpu[:, col]
            if col_vec.sum() == 0:
                continue
            labeled, num_features = ndimage.label(col_vec)
            for i in range(1, num_features + 1):
                length = (labeled == i).sum()
                if length >= vmin:
                    lens.append(length)
        arr = xp.asarray(lens, dtype=_DTYPE)
        return arr.get() if xp is cp else arr
    else:
        # Fallback to optimized vectorized method
        lens = []
        for col in range(n):
            col_vec = rp[:, col]
            if col_vec.sum() == 0:
                continue
            idx = xp.where(col_vec == 1)[0]
            if idx.size == 0:
                continue
            # Use vectorized diff and concatenate to find consecutive segments
            diff = xp.diff(idx)
            breaks = xp.where(diff != 1)[0] + 1
            starts = xp.concatenate([xp.array([0]), breaks])
            ends = xp.concatenate([breaks, xp.array([idx.size])])
            lengths = ends - starts
            valid_lengths = lengths[lengths >= vmin]
            if valid_lengths.size > 0:
                lens.extend(valid_lengths.tolist())
        arr = xp.asarray(lens, dtype=_DTYPE)
        return arr.get() if xp is cp else arr

@timer()
def _white_vertical_lengths(rp: Array):
    """Optimized white vertical line length calculation using vectorized methods.
    
    Algorithmically equivalent to the original implementation, but uses
    scipy.ndimage.label for faster connected component labeling when available.
    """
    xp = _backend(rp)
    n = rp.shape[1]
    rp_w = 1 - rp
    
    # Convert to CPU numpy array for scipy (if available)
    if xp is cp:
        rp_w_cpu = rp_w.get()
    else:
        rp_w_cpu = rp_w
    
    # Use scipy.ndimage.label for vectorized labeling (if available)
    if SCIPY_AVAILABLE and ndimage is not None:
        lens = []
        for col in range(n):
            col_vec = rp_w_cpu[:, col]
            if col_vec.sum() == 0:
                continue
            labeled, num_features = ndimage.label(col_vec)
            for i in range(1, num_features + 1):
                length = (labeled == i).sum()
                lens.append(length)
        arr = xp.asarray(lens, dtype=_DTYPE)
        return arr.get() if xp is cp else arr
    else:
        # Fallback to optimized vectorized method
        lens = []
        for col in range(n):
            col_vec = rp_w[:, col]
            if col_vec.sum() == 0:
                continue
            idx = xp.where(col_vec == 1)[0]
            if idx.size == 0:
                continue
            # Use vectorized diff to find consecutive segments
            diff = xp.diff(idx)
            breaks = xp.where(diff != 1)[0] + 1
            starts = xp.concatenate([xp.array([0]), breaks])
            ends = xp.concatenate([breaks, xp.array([idx.size])])
            lengths = ends - starts
            if lengths.size > 0:
                lens.extend(lengths.tolist())
        arr = xp.asarray(lens, dtype=_DTYPE)
        return arr.get() if xp is cp else arr

@timer()
def _recurrence_times(rp: Array) -> Tuple[float, float]:
    """Optimized recurrence time calculation using vectorized methods.
    
    Algorithmically equivalent to the original implementation, but uses
    vectorized operations for better performance.
    """
    xp = _backend(rp)
    n = rp.shape[1]
    
    # Vectorized calculation of t1: intervals between consecutive 1s across all columns
    t1_list = []
    t2_list = []
    
    # Batch process all columns
    for col in range(n):
        col_vec = rp[:, col].astype(xp.int32)
        
        # T1: intervals between consecutive 1s
        rps = xp.where(col_vec == 1)[0]
        if rps.size >= 2:
            diffs = xp.diff(rps)
            t1_list.extend(diffs.tolist())
        
        # T2: intervals between 0-to-1 transitions
        transitions = xp.where(xp.diff(col_vec) == 1)[0]
        if transitions.size >= 2:
            diffs2 = xp.diff(transitions)
            t2_list.extend(diffs2.tolist())
    
    # Convert to numpy array for computation (more efficient)
    if t1_list:
        t1_arr = np.array(t1_list, dtype=np.float32)
        t1_mean = float(t1_arr.mean())
    else:
        t1_mean = np.nan
    
    if t2_list:
        t2_arr = np.array(t2_list, dtype=np.float32)
        t2_mean = float(t2_arr.mean())
    else:
        t2_mean = np.nan
    
    return t1_mean, t2_mean

def _entropy(counts: np.ndarray) -> float:
    if counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    nz = p > 0
    return -(p[nz] * np.log2(p[nz])).sum() * np.log(2.0)

###############################################################################
# Network measures
###############################################################################
def is_symmetric_matrix(A: Array, atol: float = 0.0) -> bool:
    xp = _backend(A)
    return xp.allclose(A, A.T, atol=atol)

def compute_triangles_fast(A: Array):
    xp = _backend(A)
    A = A.astype(xp.int8 if xp is np else xp.int32, copy=False)
    A2 = A @ A
    tri = (A2 * A).sum(axis=1)
    return tri, float(tri.sum()), float(A2.sum())

def compute_triangles_with_sym_check(A: Array):
    xp = _backend(A)
    if is_symmetric_matrix(A):
        return compute_triangles_fast(A)
    tri = xp.einsum("ij,jk,ki->i", A, A, A)
    return tri, float(xp.einsum("ij,jk,ki->", A, A, A)), float((A @ A).sum())

def _network_measures(rp: Array):
    # #region agent log
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cursor', 'debug.log')
    try:
        rp_shape = rp.shape if hasattr(rp, 'shape') else 'unknown'
        rp_sum = float(rp.sum()) if hasattr(rp, 'sum') else 'unknown'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "rqa_analysis.py:338", "message": "_network_measures entry", "data": {"rp_shape": str(rp_shape), "rp_sum": rp_sum}, "timestamp": int(time.time() * 1000)}) + '\n')
    except: pass
    # #endregion
    xp = _backend(rp)
    A = rp.astype(int, copy=False)
    kv = A.sum(axis=1)
    # #region agent log
    try:
        kv_cpu = kv.get() if hasattr(kv, 'get') else kv
        kv_min, kv_max = float(kv_cpu.min()), float(kv_cpu.max())
        kv_zero_count = int((kv_cpu == 0).sum())
        kv_one_count = int((kv_cpu == 1).sum())
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "rqa_analysis.py:341", "message": "degrees computed", "data": {"kv_min": kv_min, "kv_max": kv_max, "kv_zero_count": kv_zero_count, "kv_one_count": kv_one_count}, "timestamp": int(time.time() * 1000)}) + '\n')
    except: pass
    # #endregion
    tri, trace_all, denom = compute_triangles_with_sym_check(A)
    # #region agent log
    try:
        tri_cpu = tri.get() if hasattr(tri, 'get') else tri
        tri_min, tri_max = float(tri_cpu.min()), float(tri_cpu.max())
        tri_neg_count = int((tri_cpu < 0).sum())
        is_sym = is_symmetric_matrix(A)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "rqa_analysis.py:342", "message": "triangles computed", "data": {"tri_min": tri_min, "tri_max": tri_max, "tri_neg_count": tri_neg_count, "is_symmetric": is_sym, "trace_all": trace_all, "denom": denom}, "timestamp": int(time.time() * 1000)}) + '\n')
    except: pass
    # #endregion
    with np.errstate(divide="ignore", invalid="ignore"):
        # Clustering coefficient formula (aligned with MATLAB crqa):
        # C_i = 2 * edges_in_neighborhood_i / (k_i * (k_i - 1))
        # where:
        #   edges_in_neighborhood_i = number of edges between neighbors of node i
        #   k_i = degree of node i (number of neighbors)
        #
        # Standard definition from network theory (Watts & Strogatz 1998, Marwan et al. 2007):
        # For both symmetric and asymmetric graphs, use the same formula without extra factors.
        # The tri values are pre-computed to represent the correct edge counts in neighborhoods.
        #
        # MATLAB crqa uses: cl = tri / (k * (k - 1))
        # Python now aligns with this standard implementation.
        kv_cpu = kv.get() if hasattr(kv, 'get') else kv
        tri_cpu = tri.get() if hasattr(tri, 'get') else tri
        denominator = kv_cpu * (kv_cpu - 1)
        # #region agent log
        try:
            denom_zero_count = int((denominator == 0).sum())
            denom_min = float(denominator.min())
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "rqa_analysis.py:356", "message": "before division", "data": {"denominator_zero_count": denom_zero_count, "denominator_min": denom_min}, "timestamp": int(time.time() * 1000)}) + '\n')
        except: pass
        # #endregion
        # Fix: For nodes with degree 0 or 1, clustering coefficient is 0 (cannot form triangles)
        # This prevents division by zero and matches MATLAB crqa behavior
        cl_local = np.where(denominator > 0, tri_cpu / denominator, 0.0)
        # #region agent log
        try:
            cl_local_inf_count = int(np.isinf(cl_local).sum())
            cl_local_nan_count = int(np.isnan(cl_local).sum())
            cl_local_finite = cl_local[np.isfinite(cl_local)]
            cl_local_min = float(cl_local_finite.min()) if cl_local_finite.size > 0 else None
            cl_local_max = float(cl_local_finite.max()) if cl_local_finite.size > 0 else None
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "rqa_analysis.py:357", "message": "after division", "data": {"cl_local_inf_count": cl_local_inf_count, "cl_local_nan_count": cl_local_nan_count, "cl_local_min": cl_local_min, "cl_local_max": cl_local_max}, "timestamp": int(time.time() * 1000)}) + '\n')
        except: pass
        # #endregion
        cl_local = xp.asarray(cl_local, dtype=_DTYPE) if xp is cp else np.asarray(cl_local, dtype=_DTYPE)
    # #region agent log
    try:
        cl_local_for_log = cl_local.get() if hasattr(cl_local, 'get') else cl_local
        clust_before_nanmean = float(np.nanmean(cl_local_for_log))
        cl_local_finite_for_log = cl_local_for_log[np.isfinite(cl_local_for_log)]
        clust_using_mean = float(np.mean(cl_local_finite_for_log)) if cl_local_finite_for_log.size > 0 else None
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "rqa_analysis.py:357", "message": "before nanmean", "data": {"clust_before_nanmean": clust_before_nanmean, "clust_using_mean": clust_using_mean}, "timestamp": int(time.time() * 1000)}) + '\n')
    except: pass
    # #endregion
    clust = float(xp.nanmean(cl_local))
    # #region agent log
    try:
        is_inf = np.isinf(clust) or np.isnan(clust)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "rqa_analysis.py:357", "message": "_network_measures exit", "data": {"clust": clust, "is_inf_or_nan": is_inf, "trans": float((trace_all / denom) if denom > 0 else xp.nan)}, "timestamp": int(time.time() * 1000)}) + '\n')
    except: pass
    # #endregion
    trans = float((trace_all / denom) if denom > 0 else xp.nan)
    return clust, trans

###############################################################################
# CRQA core (network-less path)
###############################################################################
@dataclass
class CRQAResult:
    RR: float; DET: float; L_mean: float; L_max: float; ENTR: float; LAM: float
    TT: float; V_max: float; T1: float; T2: float; RTE: float
    Clust: float; Trans: float

    def as_array(self) -> np.ndarray:
        return np.array([self.RR, self.DET, self.L_mean, self.L_max, self.ENTR,
                         self.LAM, self.TT, self.V_max, self.T1, self.T2,
                         self.RTE, self.Clust, self.Trans], dtype=_DTYPE)

@timer()
def _crqa_no_net(
    x: Array, *, m: int, tau: int, e: float, lmin: int, vmin: int,
    theiler: int, method: str, normalize: bool, use_gpu: bool,
    max_threads_per_channel: int = 1
) -> Tuple[CRQAResult, Array]:

    x_arr = _to_ndarray(x, use_gpu)
    if normalize:
        x_arr = _zscore(x_arr)

    x_emb = _embed(x_arr, m, tau)
    method_key = _METHODS.get(method.lower(), method.lower())
    if method_key not in _METHODS.values():
        raise NotImplementedError(f"method '{method}' not supported")

    dmat = _distance_matrix(x_emb, x_emb, method_key, use_gpu)
    xp = _backend(dmat)
    rp = (dmat <= e).astype(xp.bool_)
    del dmat
    if use_gpu and cp:
        cp._default_memory_pool.free_all_blocks()

    if theiler > 0:
        xp, n = _backend(rp), rp.shape[0]
        # Vectorized theiler window processing: set all diagonals at once
        # Create mask matrix for better efficiency
        mask = xp.ones_like(rp, dtype=xp.bool_)
        for k in range(-theiler, theiler + 1):
            if k == 0:
                continue
            diag_idx = xp.arange(max(0, -k), min(n, n - k))
            mask[diag_idx, diag_idx + k] = False
        rp = rp * mask

    # Calculate N_all and RR (on GPU to avoid premature conversion)
    xp = _backend(rp)
    n = rp.shape[0]
    N_all = rp.size - (0 if theiler == 0 else
                       2 * n * theiler - theiler * (theiler + 1))
    RR = float(rp.sum() / N_all)
    
    # Defer CPU conversion - only convert when needed
    # Compute all operations that can be done on GPU first
    rp_sum_cpu = float(rp.sum())  # Pre-compute sum to avoid repeated conversion
    
    # ---------------- Parallel phase ----------------
    results = {}

    def compute_diag():
        lens, _ = _diag_line_lengths(rp, lmin)
        lens_sum = float(lens.sum()) if lens.size else 0.0
        return {
            'L_max': float(lens.max()) if lens.size else 0.0,
            'L_mean': float(lens.mean()) if lens.size else np.nan,
            'DET': lens_sum / rp_sum_cpu if rp_sum_cpu > 0 else np.nan,
            'ENTR': _entropy(np.histogram(lens, bins=np.arange(1, n + 1))[0])
        }

    def compute_vert():
        lens = _vertical_line_lengths(rp, vmin)
        lens_sum = float(lens.sum()) if lens.size else 0.0
        return {
            'V_max': float(lens.max()) if lens.size else 0.0,
            'TT': float(lens.mean()) if lens.size else np.nan,
            'LAM': lens_sum / rp_sum_cpu if rp_sum_cpu > 0 else np.nan
        }

    def compute_white():
        white = _white_vertical_lengths(rp)
        if white.size and white.max() > 0:
            rte = _entropy(np.histogram(white, bins=np.arange(1, white.max() + 2))[0]) / np.log(white.max())
        else:
            rte = np.nan
        return {'RTE': rte}

    def compute_rec_times():
        t1, t2 = _recurrence_times(rp)
        return {'T1': t1, 'T2': t2}

    tasks = {
        'diag': compute_diag,
        'vert': compute_vert,
        'white': compute_white,
        'recur': compute_rec_times
    }

    if max_threads_per_channel > 1:
        with ThreadPoolExecutor(max_threads_per_channel) as ex:
            future_to_key = {ex.submit(func): key for key, func in tasks.items()}
            for f in as_completed(future_to_key):
                key = future_to_key[f]
                results.update(f.result())
    else:
        # fallback to sequential
        for key, func in tasks.items():
            results.update(func())

    res = CRQAResult(
        RR=RR,
        DET=results['DET'],
        L_mean=results['L_mean'],
        L_max=results['L_max'],
        ENTR=results['ENTR'],
        LAM=results['LAM'],
        TT=results['TT'],
        V_max=results['V_max'],
        T1=results['T1'],
        T2=results['T2'],
        RTE=results['RTE'],
        Clust=np.nan,
        Trans=np.nan
    )
    # Only convert to CPU at the end when needed (for network measures)
    rp_cpu = rp.get() if (cp is not None and isinstance(rp, cp.ndarray)) else rp
    return res, rp_cpu

###############################################################################
# ----------------------  GPU worker process  --------------------------- ###
###############################################################################
def _gpu_worker(task_q: managers.QueueProxy,
                result_q: managers.QueueProxy,
                device_id: int = 0):
    """GPU process: fetch RP from shared memory, compute network measures."""
    if cp is None:
        result_q.put(("ERROR", "CuPy not available"))
        return

    try:
        cp.cuda.Device(device_id).use()
    except Exception as exc:
        result_q.put(("ERROR", f"CUDA init failed: {exc}"))
        return

    while True:
        try:
            job = task_q.get(timeout=1)
        except _pyqueue.Empty:
            continue

        if job == "STOP":
            break

        try:
            qlen = task_q.qsize()
        except (AttributeError, NotImplementedError):
            qlen = "N/A"
        #print(f"[GPU Worker] got job, pending: {qlen}")

        job_id, shm_name, shape, dtype_str = job
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            rp_np = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)

            # Estimate GPU memory usage
            required_bytes = rp_np.nbytes
            if not wait_for_available_gpu_memory(required_bytes):
                raise MemoryError(f"Not enough GPU memory for {job_id}, needs {required_bytes / 1024 ** 2:.2f} MB")

            rp_gpu = cp.asarray(rp_np)  # H→D zero-copy
            clust, trans = _network_measures(rp_gpu)
            del rp_np
            del rp_gpu
            cp._default_memory_pool.free_all_blocks()
            result_q.put((job_id, clust, trans))
        except Exception as exc:
            result_q.put((job_id, "ERROR", repr(exc)))
            raise
        finally:
            shm.close()
        try:
            qlen = task_q.qsize()
        except (AttributeError, NotImplementedError):
            qlen = "N/A"
        #print(f"[GPU Worker] finished job, pending: {qlen}")

###############################################################################
# CPU-side worker (process) that offloads network part ------------------- ###
###############################################################################
@timer()
def _cpu_rqa_worker(
    sig: np.ndarray, ch_name: str, cfg: dict,
    task_q: Optional[managers.QueueProxy],
    result_q: Optional[managers.QueueProxy]
) -> np.ndarray:
    """Run CRQA up to RP; offload network measures to GPU via queues."""
    try:
        res_base, rp = _crqa_no_net(sig, **cfg)         # heavy part

        # ---------- network measures ----------
        if task_q is None:          # no GPU → local CPU
            clust, trans = _network_measures(rp)
        else:                       # send to GPU
            job_id = uuid.uuid4().hex
            shm = shared_memory.SharedMemory(create=True, size=rp.nbytes)
            shm_arr = np.ndarray(rp.shape, dtype=rp.dtype, buffer=shm.buf)
            shm_arr[...] = rp
            task_q.put((job_id, shm.name, rp.shape, rp.dtype.str))

            # wait result
            while True:
                jid, *data = result_q.get()
                if jid == job_id:
                    if data[0] == "ERROR":
                        raise RuntimeError(f"GPU worker error: {data[1]}")
                    clust, trans = data
                    break
            shm.unlink()

        result = res_base.as_array()
        result[-2] = clust
        result[-1] = trans
        return result
    except Exception as exc:
        warnings.warn(f"[{ch_name}] failed: {exc}")
        raise

###############################################################################
# Public batch API
###############################################################################
@timer()
def rqa_analysis(
    signal2: np.ndarray, *, tau: int = 1, emb_dim: Optional[int] = None,
    m: Optional[int] = None, e: float = 0.1, lmin: int = 2, vmin: int = 2,
    theiler: int = 1, method: str = "max", normalize: bool = True,
    flatten: bool = True, tqdm_progress=None, use_gpu: bool = True,
    fs: float = 500.0, max_workers: Optional[int] = None,
    max_threads_per_channel: int = 1,
    **kwargs
) -> np.ndarray:
    if signal2.ndim != 2:
        raise ValueError("`signal2` must be 2-D (channels × samples)")

    n_ch, _ = signal2.shape
    m_final = m or emb_dim or 1

    # Verify that GPU and CUDA driver are genuinely available before proceeding with GPU path
    if use_gpu:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                warnings.warn("No compatible GPU device found; falling back to CPU calculation.")
                use_gpu = False
        except Exception as exc:
            warnings.warn(f"GPU/CUDA driver unavailable ({exc}); automatically falling back to CPU calculation.")
            use_gpu = False

    cfg_common = dict(m=m_final, tau=tau, e=e, lmin=lmin, vmin=vmin,
                      theiler=theiler, method=method,
                      normalize=normalize, use_gpu=use_gpu, max_threads_per_channel=max_threads_per_channel)

    # --- decide strategy ---
    thread_only = not use_gpu
    results: list[np.ndarray] = []

    if thread_only:
        for i in range(n_ch):
            res = _cpu_rqa_worker(signal2[i], f"Ch{i+1}", cfg_common,
                                  None, None)
            results.append(res)
            if tqdm_progress is not None:
                tqdm_progress.update()
        feat = np.vstack(results)
        return feat.ravel() if flatten else feat

    # -------- CPU ProcessPool + GPU worker --------
    cpu_workers = max(max_workers - 1, 1)
    manager = mp.Manager()                    # creates shared server process
    task_q: managers.QueueProxy   = manager.Queue()
    result_q: managers.QueueProxy = manager.Queue()

    gpu_proc = mp.Process(target=_gpu_worker,
                          args=(task_q, result_q, 0),
                          daemon=True)
    gpu_proc.start()

    with ProcessPoolExecutor(max_workers=cpu_workers,
                             mp_context=mp.get_context('spawn')) as ex:
        futures = [ex.submit(_cpu_rqa_worker,
                             signal2[i], f"Ch{i+1}", cfg_common,
                             task_q, result_q)
                   for i in range(n_ch)]
        for f in as_completed(futures):
            results.append(f.result())
            if tqdm_progress is not None:
                tqdm_progress.update()

    task_q.put("STOP")
    gpu_proc.join()

    feat = np.vstack(results)
    return feat.ravel() if flatten else feat


__all__ = ["rqa_analysis", "CRQAResult"]
