"""Diagnostic figures (debug mode only; never used by the pipeline logic)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402


def _envelope(x: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min/max of x in n_bins equal blocks (bin index, min, max): a whole file drawn without plotting every sample."""
    n = len(x) // n_bins * n_bins
    b = x[:n].reshape(n_bins, -1)
    return np.arange(n_bins), b.min(1), b.max(1)


def plot_waveform(path: Path, s1, s2, zoom_s: float | None = None, zoom_ms: float = 2.0, title: str = "") -> dict:
    """S1 waveform of a whole file, a zoom with the S1 and S2 samples overlaid, and the S1 vs S2 power spectra
    (Welch, equal frequency resolution). Returns the numbers shown, for the caller to print."""
    fs1, fs2 = s1.fs, s2.fs
    x1, x2 = np.asarray(s1.samples, np.float64), np.asarray(s2.samples, np.float64)
    if zoom_s is None:
        zoom_s = s1.duration_s / 2
    z1 = slice(int(zoom_s * fs1), int((zoom_s + zoom_ms / 1e3) * fs1))
    z2 = slice(int(np.ceil(zoom_s * fs2)), int((zoom_s + zoom_ms / 1e3) * fs2))
    bin_hz = 62.5     # divides both rates: identical bin frequencies and segment lengths (8000 / 3072 samples).
    f1, p1 = signal.welch(x1, fs1, nperseg=int(fs1 / bin_hz))   # Unequal segments (e.g. 8192 vs 3145.7) bias the
    f2, p2 = signal.welch(x2, fs2, nperseg=int(fs2 / bin_hz))   # S2/S1 ratio by +0.3 dB on real, peaky spectra.
    band = (f2 >= 2e3) & (f2 <= 80e3)
    diff_db = 10 * np.log10(p2[band] / np.interp(f2[band], f1, p1))

    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    k, lo, hi = _envelope(x1, 3000)
    t = (k + 0.5) * (len(x1) // 3000) / fs1
    ax[0].fill_between(t, lo, hi, color="tab:blue", lw=0)
    ax[0].axvspan(zoom_s, zoom_s + zoom_ms / 1e3, color="tab:red", alpha=0.4)
    ax[0].set_title(f"S1: channel {s1.meta.get('channel_index', '?')} at {fs1 / 1e3:.0f} kHz, whole file "
                    f"({s1.duration_s:.1f} s; min/max per {len(x1) // 3000} samples; red = zoom below)")
    ax[0].set_xlabel("time in file (s)")
    ax[1].plot((np.arange(z1.start, z1.stop) / fs1 - zoom_s) * 1e3, x1[z1], "-", lw=0.8, color="tab:blue",
               label=f"S1 {fs1 / 1e3:.0f} kHz ({z1.stop - z1.start} samples)")
    ax[1].plot((np.arange(z2.start, z2.stop) / fs2 - zoom_s) * 1e3, x2[z2], "o", ms=2.5, color="tab:orange",
               label=f"S2 {fs2 / 1e3:.0f} kHz ({z2.stop - z2.start} samples)")
    ax[1].set_title(f"Zoom {zoom_ms:g} ms at {zoom_s:.3f} s: original and resampled samples")
    ax[1].set_xlabel(f"time after {zoom_s:.3f} s (ms)")
    ax[1].legend(loc="upper right")
    ax[2].plot(f1 / 1e3, 10 * np.log10(p1), lw=0.8, color="tab:blue", label=f"S1 {fs1 / 1e3:.0f} kHz")
    ax[2].plot(f2 / 1e3, 10 * np.log10(p2), lw=0.8, color="tab:orange", label=f"S2 {fs2 / 1e3:.0f} kHz")
    ax[2].axvline(fs2 / 2e3, ls="--", color="gray")
    ax[2].text(fs2 / 2e3, ax[2].get_ylim()[1], f" S2 Nyquist {fs2 / 2e3:.0f} kHz", va="top", color="gray")
    ax[2].set_title(f"Power spectrum (Welch, {bin_hz:g} Hz bins): median S2 - S1 over 2-80 kHz "
                    f"= {np.median(diff_db):+.2f} dB")
    ax[2].set_xlabel("frequency (kHz)")
    ax[2].set_ylabel("dB / Hz")
    ax[2].legend(loc="upper right")
    for a in ax[:2]:
        a.set_ylabel("amplitude (full scale = 1)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

    def stats(x, fs):
        return {"fs_hz": int(fs), "n_samples": len(x), "duration_s": round(len(x) / fs, 4),
                "peak": round(float(np.abs(x).max()), 5), "rms": round(float(np.sqrt(np.mean(x ** 2))), 6)}

    return {"file": s1.name, "channel": s1.meta.get("channel_index"), "s1": stats(x1, fs1), "s2": stats(x2, fs2),
            "zoom": {"start_s": round(zoom_s, 4), "ms": zoom_ms},
            "psd_median_diff_db_2_80khz": round(float(np.median(diff_db)), 3)}


def plot_overview(path: Path, spec, dm, tr, cfg, title: str = "", seconds: float = 12.0) -> None:
    """Two excerpts (file start, and the busiest stretch by track count): S4, S5 confidence, tracks over S4."""
    hop_s = spec.hop_s
    n_win = int(seconds / hop_s)
    T = spec.n_frames
    starts = [0]
    if tr.tracks and T > 2 * n_win:
        cnt = np.zeros(T)
        for t in tr.tracks:
            cnt[t.frames[0]:t.frames[-1] + 1] += 1
        csum = np.convolve(cnt, np.ones(n_win), mode="valid")
        starts.append(int(np.argmax(csum)))
    k_lo, k_hi = dm.bin_lo, dm.bin_hi
    f_ext = [k_lo * spec.bin_hz / 1e3, (k_hi + 1) * spec.bin_hz / 1e3]
    fig, ax = plt.subplots(3, len(starts), figsize=(9 * len(starts), 9), squeeze=False)
    for j, a in enumerate(starts):
        b = min(T, a + n_win)
        t0, t1 = a * hop_s, b * hop_s
        sub = spec.db[a:b, k_lo:k_hi + 1]
        ext = [t0, t1, f_ext[0], f_ext[1]]
        ax[0, j].imshow(sub.T, origin="lower", aspect="auto", extent=ext, cmap="viridis",
                        vmin=np.percentile(sub, 5), vmax=np.percentile(sub, 99.8))
        ax[0, j].set_title(f"S4 log-magnitude (band)  [{t0:.0f}-{t1:.0f} s]")
        ax[1, j].imshow(dm.conf[a:b].T, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=1)
        ax[1, j].set_title(f"S5 confidence ({dm.detector_version})")
        ax[2, j].imshow(sub.T, origin="lower", aspect="auto", extent=ext, cmap="gray",
                        vmin=np.percentile(sub, 5), vmax=np.percentile(sub, 99.8))
        for t in tr.tracks:
            if t.frames[-1] < a or t.frames[0] >= b:
                continue
            m = (t.frames >= a) & (t.frames < b)
            ax[2, j].plot(t.frames[m] * hop_s + 0.5 * hop_s, t.freq_hz[m] / 1e3, lw=1.6, color=plt.cm.tab20(t.id % 20))
        ax[2, j].set_title(f"S6 tracks over S4 ({len(tr.tracks)} kept in file, {len(tr.rejected)} rejected)")
        for r in range(3):
            ax[r, j].set_ylabel("kHz")
        ax[2, j].set_xlabel("time in file (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
