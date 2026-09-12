"""nonlinear_analysis.py
================================
Python re‑implementation of MATLAB `Nonlinear_Analysis.m` **with full feature parity**
(17 non‑linear features per EEG channel) + optional diagnostic plots.

The module exposes a single public function:

    nonlinear_analysis(signal2, fs=500, tau=10, emb_dim=2,
                       save_plots=True, plot_dir="plots", channel_names=None)

and an `__all__` list so that `from nonlinear_analysis import *` only exports
that API.

Dependencies (install with pip):
    numpy, scipy, nolds, antropy, matplotlib, pyts, pywavelets (optional), pyrqa (optional)
"""

from __future__ import annotations
from collections import deque

import warnings
import os, shutil, tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

# Non‑linear dynamics & entropy libraries
import nolds  # corr_dim, lyap_r, etc.
import antropy as ant  # sample_entropy, permutation_entropy
from features.exact_entropy import sample_entropy, permutation_entropy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, Future
from threading import Lock

# from tqdm import tqdm
# Optional: wavelet entropy (PyWavelets)
try:
    import pywt  # nosec B402 – only used for entropy calculation
except ImportError:  # pragma: no cover
    pywt = None
    warnings.warn(
        "PyWavelets not installed — Wavelet entropy will be returned as NaN. Run `pip install PyWavelets`.")

# Optional: Recurrence Quantification Analysis (pyrqa)
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.analysis_type import Classic
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.computation import RQAComputation
    from pyrqa.metric import EuclideanMetric

    PYRQA_OK = True
except ImportError:  # pragma: no cover
    PYRQA_OK = False
    warnings.warn(
        "pyrqa not installed — RQA features will be NaN. Run `pip install pyrqa` to enable.")

# Optional diagnostic plots (delegated to a helper module)
try:
    from .plots import generate_channel_plots  # when part of a package
except ImportError:  # standalone script fallback
    try:
        from plots import generate_channel_plots  # type: ignore
    except ImportError:  # pragma: no cover
        def generate_channel_plots(*_args, **_kwargs):  # type: ignore
            return  # plotting silently disabled if helper not found

__all__ = ["nonlinear_analysis"]


class SignalPreloader:
    def __init__(self, load_fn):
        self.load_fn = load_fn
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures: dict[int, Future] = {}
        self.lock = Lock()

    def preload(self, ch_idx: int) -> None:
        with self.lock:
            if ch_idx not in self.futures:
                print(f"[Preloader] ⏳ Starting preload for channel {ch_idx}")
                self.futures[ch_idx] = self.executor.submit(self.load_fn, ch_idx)

    def get(self, ch_idx: int):
        with self.lock:
            if ch_idx not in self.futures:
                print(f"[Preloader] 🔄 Loading on-demand for channel {ch_idx}")
                self.futures[ch_idx] = self.executor.submit(self.load_fn, ch_idx)
            else:
                print(f"[Preloader] ✅ Reusing preload for channel {ch_idx}")
            future = self.futures[ch_idx]
        result = future.result()
        print(f"[Preloader] 📦 Load complete for channel {ch_idx}")
        return result


# -----------------------------------------------------------------------------
# Internal helper functions
# -----------------------------------------------------------------------------

def _wavelet_entropy(sig: np.ndarray, wavelet: str = "db4", level: int | None = None) -> float:
    """Shannon entropy of wavelet energy distribution (as in MATLAB `wentropy`)."""
    if pywt is None:
        return float("nan")
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    if energies.sum() == 0:
        return 0.0
    probs = energies / energies.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _bandpower(sig: np.ndarray, fs: float, fmin: float, fmax: float, nperseg: int = 1024) -> float:
    """Relative bandpower via Welch PSD integration (equivalent to MATLAB `bandpower`)."""
    f, Pxx = welch(sig, fs=fs, nperseg=min(nperseg, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(Pxx[idx], f[idx]))


def _power_signal(sig: np.ndarray) -> float:
    """Mean signal power (µV²) — MATLAB `powersignal`."""
    return float(np.mean(sig ** 2))


def _rqa_features(x: np.ndarray, m: int, tau: int):
    """
    Return RR, DET, ENTR, L for a 1-D signal `x`.
    Falls back to (np.nan, …) if PyRQA is absent.
    """
    if not PYRQA_OK:
        return (np.nan,) * 4
    ts = TimeSeries(x.reshape(-1, 1), embedding_dimension=m, time_delay=tau)
    settings = Settings(
        ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(0.1),  # equal to MATLAB RPplot_FAN(...,10,0)
        similarity_measure=EuclideanMetric(),
    )
    result = RQAComputation.create(settings).run()
    return (
        result.recurrence_rate,
        result.determinism,
        result.entropy_diagonal_lines,
        result.average_diagonal_line,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _features_per_channel(
        x: np.ndarray,
        *,
        fs: float,
        tau,
        lag,
        emb_dim,
        save_plots: bool,
        plot_dir: Path,
        ch_label: str,
        max_threads_for_features_per_channel: int = 5
) -> list[float]:
    """
        Compute a subset of non-linear features for a single EEG channel.

        Notes
        -----
        • This function already runs inside a *process* spawned by a
          ProcessPoolExecutor.  We therefore exploit **threads** here to overlap
          Python-level latency (and C extensions that release the GIL) without
          spawning extra processes.
        • Only f1, f2, f3, f7, f8 are active.  All other legacy features remain
          commented out for future use.
        """

    # ------------------------------------------------------------------
    # 0. Pre-compute shared values
    # ------------------------------------------------------------------
    if not emb_dim:
        emb_dim = nolds.embedding_dim(x, tau=tau, dims=range(2, 15))[0]

    r_vals = nolds.logarithmic_r(0.1 * np.std(x),
                                 0.5 * np.std(x),
                                 factor=1.08)[:25]

    # ------------------------------------------------------------------
    # 1. Define feature lambdas (must be picklable inside the same process)
    # ------------------------------------------------------------------
    def calc_f1():
        # Correlation Dimension
        return nolds.corr_dim(x, emb_dim=2, lag=lag, rvals=r_vals, fit="poly")

    def calc_f2():
        # Higuchi Fractal Dimension
        return ant.higuchi_fd(x)

    def calc_f3():
        # Largest Lyapunov Exponent (scaled by fs)
        lle_per_step = nolds.lyap_r(
            x, tau=lag, emb_dim=2,
            trajectory_len=5, fit="poly", min_tsep=tau
        )
        return lle_per_step * fs  # fs=500 → 1 / s

    def calc_f7():
        # Sample Entropy
        return sample_entropy(x, m=emb_dim, tau=tau)

    def calc_f8():
        # Permutation Entropy
        return permutation_entropy(x, m=emb_dim, tau=tau, normalize=True)

    # ── 4. Wavelet Shannon Entropy ──────────────────────────────────
    # f4 = _wavelet_entropy(x)
    #
    # # ── 5. Kurtosis (Fisher flag off to match MATLAB) ───────────────
    # f5 = kurtosis(x, fisher=False, bias=False)
    #
    # # ── 6. Mean signal power ────────────────────────────────────────
    # f6 = _power_signal(x)

    # ── 9-13. Band powers δ, θ, α, β, γ ─────────────────────────────
    # f9 = _bandpower(x, fs, 0.1, 4)
    # f10 = _bandpower(x, fs, 4, 8)
    # f11 = _bandpower(x, fs, 8, 13)
    # f12 = _bandpower(x, fs, 13, 30)
    # f13 = _bandpower(x, fs, 30, 100)

    # ── 14-17. Recurrence Quantification Analysis ──────────────────
    # f14, f15, f16, f17 = _rqa_features(x, emb_dim, tau)

    feature_funcs = {
        "f1": calc_f1,
        "f2": calc_f2,
        "f3": calc_f3,
        "f7": calc_f7,
        "f8": calc_f8,
    }

    # ------------------------------------------------------------------
    # 2. Parallel execution (threads)
    # ------------------------------------------------------------------
    results: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_threads_for_features_per_channel) as executor:
        future_map = {executor.submit(func): name
                      for name, func in feature_funcs.items()}

        for fut in as_completed(future_map):
            name = future_map[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:
                warnings.warn(f"{name} failed on {ch_label}: {exc}")
                # Re-raise so the parent process can react (logging / retry)
                raise

    # ------------------------------------------------------------------
    # 3. Optional diagnostic plots
    # ------------------------------------------------------------------
    if save_plots:
        generate_channel_plots(x, fs, tau, emb_dim, plot_dir, ch_label)

    # ------------------------------------------------------------------
    # 4. Assemble output in fixed order
    # ------------------------------------------------------------------
    return [
        results["f1"],
        results["f2"],
        results["f3"],
        results["f7"],
        results["f8"],
    ]


def nonlinear_analysis(
        signal2: np.ndarray,
        *,
        fs: float = 500.0,
        tau: int = 10,
        lag: int = 1,
        emb_dim: int = None,
        save_plots: bool = True,
        plot_dir: str | Path = "plots",
        channel_names: list[str] | None = None,
        flatten: bool = True,
        tqdm_progress=None,
        max_threads_per_channel: int | None = 1,
        max_workers: int | None = 1,
        use_cache: bool = False,
        cache_dir=None,
        on_almost_done_channels: callable | None = None,
        channel_threshold: int = 1,
) -> np.ndarray:
    if signal2.ndim != 2:
        raise ValueError("`signal2` must be 2-D (channels × samples)")

    n_channels, _ = signal2.shape
    channel_names = channel_names or [f"ch{c:02d}" for c in range(n_channels)]
    plot_dir = Path(plot_dir)

    temp_dir = None
    if use_cache:
        # print(f"use cache -- cpu count: {os.cpu_count()}, max workers: {max_workers }")
        import uuid
        if cache_dir:
            temp_dir = cache_dir / "__nonlinear_cache" / f"nl_cache_{uuid.uuid4().hex}"
        else:
            temp_dir = Path(tempfile.gettempdir()) / f"nl_cache_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)  # ensure uniqueness
        for ch_idx in range(n_channels):
            np.save(temp_dir / f"ch_{ch_idx:02d}.npy", signal2[ch_idx])
        del signal2

    def load_channel(ch_idx):
        if use_cache:
            # Avoid mmap on Windows: open memmaps keep handles and block shutil.rmtree
            # (WinError 32) after channel processing finishes.
            return np.load(temp_dir / f"ch_{ch_idx:02d}.npy")
        else:
            return signal2[ch_idx]

    # ─── Main execution ─────────────────────────────────────────────
    feat_rows = [None] * n_channels
    if not max_workers or max_workers <= 1:
        for ch_idx in range(n_channels):
            x = load_channel(ch_idx)
            res = _features_per_channel(
                x,
                fs=fs, tau=tau, lag=lag, emb_dim=emb_dim,
                save_plots=save_plots, plot_dir=plot_dir,
                ch_label=channel_names[ch_idx],
            )
            feat_rows[ch_idx] = res
            if tqdm_progress is not None:
                tqdm_progress.update()
    # ── Multiprocessing branch ------------------------------------
    else:
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as exe:
            fut_to_idx = {
                exe.submit(
                    _features_per_channel,
                    load_channel(ch_idx),
                    fs=fs, tau=tau, lag=lag, emb_dim=emb_dim,
                    save_plots=save_plots, plot_dir=plot_dir,
                    ch_label=channel_names[ch_idx],
                    max_threads_for_features_per_channel=max_threads_per_channel
                ): ch_idx
                for ch_idx in range(n_channels)
            }
            total_futs = len(fut_to_idx)
            processed = 0
            callback_fired = False
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    feat_rows[idx] = fut.result()
                    processed += 1
                    remaining = total_futs - processed
                    if (not callback_fired
                            and on_almost_done_channels
                            and remaining <= channel_threshold):
                        on_almost_done_channels()
                        callback_fired = True
                    if tqdm_progress is not None:
                        tqdm_progress.update()
                except Exception as exc:
                    warnings.warn(f"Worker on {channel_names[idx]} failed: {exc}")
                    raise RuntimeError(f"Worker on {channel_names[idx]} failed: {exc}")


    # ─── Clean up temp cache ────────────────────────────────────────
    if temp_dir and temp_dir.exists():
        import gc
        import time as _time

        gc.collect()
        last_err = None
        for _ in range(5):
            try:
                shutil.rmtree(temp_dir)
                last_err = None
                break
            except PermissionError as exc:  # Windows file-lock race
                last_err = exc
                _time.sleep(0.2)
                gc.collect()
        if last_err is not None:
            warnings.warn(f"Could not remove nonlinear cache {temp_dir}: {last_err}")

    # ─── Return result ──────────────────────────────────────────────
    feat_arr = np.asarray(feat_rows, dtype=float)
    return feat_arr.ravel() if flatten else feat_arr
