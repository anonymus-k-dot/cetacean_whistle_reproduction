"""DCLDE 2011 benchmark for the 5-24 kHz detector: run it on the Li et al. (2023) test files and score it their way.

    python mod_detector_5_24/dclde2011.py detect   [--tag NAME] [--override KEY=VALUE ...] [--device cpu|cuda]
    python mod_detector_5_24/dclde2011.py score    --tag NAME [--gt 2011|2025] [--min-det-ms 150] [--min-overlap 0.3] [--full-band]
    python mod_detector_5_24/dclde2011.py selftest [--gt 2011|2025]

Data (colab_download_dclde2011.py layout, default mod_detector_5_24/data/dclde2011, --data to change):
    development/<species>/<stem>.wav and .ann (2011 annotations), annotations2025/development/<species>/<stem>.ann
Detections go to <data>_detections/<tag>/<species>/<stem>.det in Silbido's tonal format, so Silbido's own scoreall
can score the same files.

Scoring re-implements Silbido's dtPerformance.m / dtAnalyzeResults.m (github.com/MarineBioAcousticsRC/silbido) with the
criteria of Li et al. (2023, JASA, doi:10.1121/10.0020274, Sec. II E):
  * a ground-truth (GT) whistle is EXPECTED when it lasts >= 150 ms and has > 10 dB SNR over at least one third of its
    frames (Silbido's SNR: 8 ms Hamming frames every 2 ms, 5-50 kHz, 3x3 median filter, per-bin mean subtraction over
    non-click frames of ~8 s blocks, strongest bin within +/-500 Hz of the annotated path);
  * a detection matches a GT whistle when, over the part of the detection inside the whistle's time span, the mean
    absolute frequency difference is <= 350 Hz and (Li et al.'s added rule) that part covers > 30 % of the whistle;
  * precision = matches to expected GT / (those + detections matching no GT at all); detections that match only
    unexpected GT count neither way. recall = expected GT matched / expected GT. Counts are pooled per species, then
    precision, recall and F1 are averaged over the four species.
Band limit (this detector only sees 5-24 kHz): GT whistles not entirely inside [band_lo_hz, band_hi_hz] of
pipeline.toml are treated like low-SNR ones (not expected, not penalised). --full-band switches this off.
Minimum detection length: Silbido's graph search only outputs tonals >= 150 ms (odontocete.xml minlen_ms), while S6
keeps tracks from ~64 ms. --min-det-ms 150 drops shorter detections before scoring, so the output follows the same
convention as the benchmark's detectors (default 0 = all tracks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import whistlepipe.config as C  # noqa: E402

DEFAULT_DATA = HERE / "data" / "dclde2011"

# Li et al. (2023) test set: 9 development files, 4 species (all mono, 192 kHz).
TEST_FILES = [
    ("bottlenose", "Qx-Tt-SCI0608-N1-060814-121518"),
    ("bottlenose", "palmyra092007FS192-070924-205305"),
    ("bottlenose", "palmyra092007FS192-070924-205730"),
    ("common", "Qx-Dc-CC0411-TAT11-CH2-041114-154040-s"),
    ("common", "QX-Dc-FLIP0610-VLA-061015-165000"),
    ("melon-headed", "palmyra092007FS192-071004-032342"),
    ("melon-headed", "palmyra102006-061020-204327_4"),
    ("spinner", "palmyra092007FS192-070927-224737"),
    ("spinner", "palmyra102006-061103-213127_4"),
]

# Silbido scoring constants (odontocete.xml defaults; dtPerformance.m sets blocklen_s = 5, PeakTolerance 500, Match 350)
SIL_LEN_S, SIL_ADV_S = 0.008, 0.002
SIL_LO_HZ, SIL_HI_HZ = 5000.0, 50000.0
SIL_CLICK_DB, SIL_BROADBAND = 10.0, 0.01
SIL_BLOCK_S, SIL_PAD_S = 5.0, 1.5
PEAK_TOL_HZ, MATCH_TOL_HZ = 500.0, 350.0
GT_SNR_DB, GT_RATIO_ABOVE, GT_MIN_LEN_S = 10.0, 1 / 3, 0.150


# ----------------------------------------------------------------------------- files
def gt_path(data: Path, species: str, stem: str, gt: str) -> Path:
    sub = "development" if gt == "2011" else "annotations2025/development"
    return data / sub / species / f"{stem}.ann"


def read_tonals(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Silbido tonal file (.ann/.bin/.det): headerless legacy format or 'silbido!' header (as silbidopy reads it)."""
    b = Path(path).read_bytes()
    pos, mask, version = 0, 0b11, -1
    if b[:8] == b"silbido!":
        version, mask, _ = struct.unpack(">HHH", b[8:14])
        pos = struct.unpack(">I", b[14:18])[0]           # header size: skip comment/timestamp
    per_tonal = 8 * ((mask & 16) > 0) + 8 * ((mask & 32) > 0)
    per_point = 8 * sum((mask & bit) > 0 for bit in (1, 2, 4, 8, 64))
    if mask & (512 | 1024) or (mask & 3) != 3:
        raise ValueError(f"{path}: unsupported tonal fields (mask {mask:#x})")
    out = []
    while pos < len(b):
        pos += per_tonal + (8 if version > 2 else 0)
        n = struct.unpack(">i", b[pos:pos + 4])[0]
        pos += 4
        pts = np.frombuffer(b, dtype=">f8", count=n * per_point // 8, offset=pos).reshape(n, per_point // 8)
        pos += n * per_point
        out.append((pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)))
    return out


def write_tonals(path: Path, tonals) -> None:
    """Headerless legacy Silbido format (time, frequency per point); Silbido and read_tonals both read it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for t, fr in tonals:
            f.write(struct.pack(">i", len(t)))
            f.write(np.column_stack([t, fr]).astype(">f8").tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Mono float64 in [-1, 1) (int16/32768 as in S1; 24-bit wavs come back from scipy as left-justified int32)."""
    fs, x = wavfile.read(str(path))
    if x.ndim != 1:
        raise ValueError(f"{path}: expected mono, got {x.shape[1]} channels")
    scale = {np.dtype("int16"): 32768.0, np.dtype("int32"): 2.0 ** 31, np.dtype("uint8"): 128.0}.get(x.dtype, 1.0)
    x = x.astype(np.float64)
    if x.dtype == np.uint8:
        x -= 128.0
    return x / scale, int(fs)


def _round(x):
    """MATLAB round (half away from zero)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _unique_first(t, f):
    t2, i = np.unique(t, return_index=True)
    return t2, np.asarray(f)[i]


# ----------------------------------------------------------------------------- Silbido SNR of each GT whistle
def _silbido_block(x: np.ndarray, fs: int, start_s: float, stop_s: float):
    """dtSpecAnal(..., 'linear') + dtSpectrogramNoiseComp('median') on one block: (snr_dB (bins, frames), frame times)."""
    L, A = int(round(fs * SIL_LEN_S)), int(round(fs * SIL_ADV_S))
    seg = x[int(round(start_s * fs)):int(round(stop_s * fs))]
    n_fr = max(0, (len(seg) - L + A) // A)
    bin_hz = fs / L
    k_lo = int(np.ceil(SIL_LO_HZ / bin_hz)) + 1                          # 1-based, as in MATLAB
    k_hi = min(int(np.ceil(SIL_HI_HZ / bin_hz)) + 1, L // 2)
    if n_fr == 0:
        return np.zeros((k_hi - k_lo + 1, 0)), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(seg, L)[::A][:n_fr] * signal.get_window("hamming", L, fftbins=False)
    mag = np.abs(np.fft.fft(frames, axis=1)[:, k_lo - 1:k_hi]).T
    power_db = 20 * np.log10(np.maximum(mag, 1e-300))
    meanf = power_db.mean(axis=1, keepdims=True)
    click = ((power_db - meanf) > SIL_CLICK_DB).sum(axis=0) > SIL_BROADBAND * power_db.shape[0]
    use = ~click if not click.all() else np.ones_like(click)
    med = ndimage.median_filter(power_db, size=3, mode="constant", cval=0.0)   # MATLAB medfilt2 pads with zeros
    snr = med - med[:, use].mean(axis=1, keepdims=True)
    return snr, np.arange(n_fr) * A / fs + start_s


def gt_snr_db(x: np.ndarray, fs: int, gts, ratio_above: float = GT_RATIO_ABOVE) -> tuple[np.ndarray, np.ndarray]:
    """Per GT whistle: SNR exceeded by `ratio_above` of its frames, and a 'skipped' flag (shorter than one frame
    advance: Silbido ignores these entirely). Follows dtPerformance.m block by block; gts sorted by start time."""
    L, A = int(round(fs * SIL_LEN_S)), int(round(fs * SIL_ADV_S))
    bin_hz = fs / L
    k_lo = int(np.ceil(SIL_LO_HZ / bin_hz)) + 1
    n_bins = min(int(np.ceil(SIL_HI_HZ / bin_hz)) + 1, L // 2) - k_lo + 1
    search = int(_round(PEAK_TOL_HZ / bin_hz))
    eof_s = len(x) / fs
    last_s = np.floor((eof_s - SIL_LEN_S) / SIL_ADV_S) * SIL_ADV_S
    times = np.array([[t[0], min(t[-1], last_s)] for t, _ in gts]).reshape(-1, 2)
    snr_out = np.full(len(gts), -np.inf)
    skipped = np.zeros(len(gts), bool)
    done = np.zeros(len(gts), bool)
    while not done.all():
        g0 = int(np.argmin(done))
        start = times[g0, 0]
        stop = max(times[g0, 1], start + SIL_BLOCK_S)
        proc = np.flatnonzero((times[:, 0] >= start) & (times[:, 1] <= stop) & ~done)
        spec, t_idx = _silbido_block(x, fs, max(0.0, start - SIL_PAD_S), min(stop + SIL_PAD_S, eof_s))
        for g in proc:
            done[g] = True
            t, f = gts[g]
            if t[-1] - t[0] < A / fs:
                skipped[g] = True
                continue
            idx = np.flatnonzero((t_idx >= times[g, 0]) & (t_idx <= times[g, 1]))
            if not len(idx):
                continue
            t2, f2 = _unique_first(t, f)
            f_q = (_round(np.interp(t_idx[idx], t2, f2) / bin_hz) - k_lo + 1).astype(int)   # 1-based bin in range
            lo = np.clip(f_q - search, 1, n_bins) - 1
            hi = np.clip(f_q + search, 1, n_bins)
            s = np.sort([spec[a:b, j].max() for a, b, j in zip(lo, hi, idx)])
            snr_out[g] = s[max(1, int(_round((1 - ratio_above) * len(s)))) - 1]
    return snr_out, skipped


# ----------------------------------------------------------------------------- matching (dtPerformance.m)
def score_file(dets, gts, expected: np.ndarray, skipped: np.ndarray, min_overlap: float = 0.3) -> dict:
    """Silbido matching for one file. gts sorted by start time; `expected` = GT that counts toward recall."""
    det_t0 = np.array([t[0] for t, _ in dets]) if dets else np.zeros(0)
    det_t1 = np.array([t[-1] for t, _ in dets]) if dets else np.zeros(0)
    false_pos = np.ones(len(dets), bool)
    det_matches_expected = 0
    gt_match = gt_miss = 0
    deviations, coverage = [], []
    for g, (t, f) in enumerate(gts):
        if skipped[g]:
            continue
        t, f = _unique_first(t, f)
        g0, g1, dur = t[0], t[-1], t[-1] - t[0]
        cand = np.flatnonzero(((det_t0 <= g0) & (det_t1 >= g0)) | ((det_t0 >= g0) & (det_t0 <= g1)))
        devs, covered, matched = [], 0.0, False
        for d in cand:
            dt, df = dets[d]
            m = (dt >= g0) & (dt <= g1)
            if not m.any():
                continue
            t_ov = dt[m]
            dev = np.abs(df[m] - np.interp(t_ov, t, f))
            span = t_ov[-1] - t_ov[0]
            if dev.mean() <= MATCH_TOL_HZ and (min_overlap <= 0 or span > min_overlap * dur):
                matched = True
                false_pos[d] = False
                det_matches_expected += int(expected[g])
                devs.append(dev)
                covered += span
        if expected[g]:
            if matched:
                gt_match += 1
                deviations.append(float(np.concatenate(devs).mean()))
                coverage.append(min(covered / dur, 1.0) * 100 if dur > 0 else 100.0)
            else:
                gt_miss += 1
    return {"n_detections": len(dets), "detections_matched": det_matches_expected, "false_pos": int(false_pos.sum()),
            "gt_expected": gt_match + gt_miss, "gt_matched": gt_match, "gt_missed": gt_miss,
            "deviations_hz": deviations, "coverage_pct": coverage}


def summarise(rows: list[dict]) -> dict:
    """dtAnalyzeResults 'overall': pool counts, then precision/recall/F1 (percent), deviation, coverage, fragmentation."""
    det = sum(r["detections_matched"] for r in rows)
    fp = sum(r["false_pos"] for r in rows)
    match = sum(r["gt_matched"] for r in rows)
    exp = sum(r["gt_expected"] for r in rows)
    dev = [v for r in rows for v in r["deviations_hz"]]
    cov = [v for r in rows for v in r["coverage_pct"]]
    p = 100 * det / (det + fp) if det + fp else float("nan")
    r = 100 * match / exp if exp else float("nan")
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "deviation_hz": float(np.mean(dev)) if dev else float("nan"),
            "coverage_pct": float(np.mean(cov)) if cov else float("nan"),
            "fragmentation": det / match if match else float("nan"),
            "gt_expected": exp, "gt_matched": match, "detections_matched": det, "false_pos": fp,
            "n_detections": sum(r["n_detections"] for r in rows)}


def expected_mask(gts, snr, skipped, band, full_band: bool) -> tuple[np.ndarray, dict]:
    dur = np.array([t[-1] - t[0] for t, _ in gts])
    in_band = np.array([full_band or (f.min() >= band[0] and f.max() <= band[1]) for _, f in gts], bool)
    ok_len, ok_snr = dur >= GT_MIN_LEN_S, snr > GT_SNR_DB
    exp = ok_len & ok_snr & in_band & ~skipped
    return exp, {"gt_total": len(gts), "too_short": int((~ok_len).sum()), "low_snr": int((ok_len & ~ok_snr).sum()),
                 "outside_band": int((ok_len & ok_snr & ~in_band).sum()), "skipped": int(skipped.sum()),
                 "expected": int(exp.sum())}


def sorted_gts(path: Path):
    gts = [(t, f) for t, f in read_tonals(path) if len(t) > 0]
    return [gts[i] for i in np.argsort([t[0] for t, _ in gts], kind="stable")]


# ----------------------------------------------------------------------------- commands
def load_cfg(overrides: list[str]):
    return C.load_pipeline_config(HERE / "pipeline.toml", overrides)


def band_hz(cfg) -> tuple[float, float]:
    s5 = cfg["decisions"]["s5"]
    return float(s5["band_lo_hz"]), float(s5["band_hi_hz"])


def detect_file(wav: Path, cfg, det) -> tuple[list, dict]:
    """S2-S6 of the pipeline on one whole recording; returns tracks as (frame-centre time s, frequency Hz)."""
    from whistlepipe import dsp
    from whistlepipe.audio import AudioSignal
    from whistlepipe.tracking import run_s6
    t0 = time.perf_counter()
    x, fs = read_wav(wav)
    s3 = dsp.s3_highpass(dsp.s2_resample(AudioSignal(x, float(fs), wav.name), cfg), cfg)
    s4 = dsp.s4_stft(s3, cfg)
    tr = run_s6(det.predict(s4), cfg)
    t_c = s4.frame_center_times_s()
    tracks = [(t_c[k.frames], k.freq_hz.astype(np.float64)) for k in tr.tracks]
    return tracks, {"duration_s": round(len(x) / fs, 3), "fs": fs, "n_tracks": len(tracks),
                    "s6_stats": {k: v for k, v in tr.stats.items() if k == "rejected_by_reason" or np.isscalar(v)},
                    "seconds": round(time.perf_counter() - t0, 1)}


def cmd_detect(a) -> None:
    import torch
    from whistlepipe.detector import RidgeDetector
    if a.override and not a.tag:
        sys.exit("give a --tag when using --override, so the detections do not mix with the default run")
    cfg = load_cfg(a.override)
    ck = Path(cfg["decisions"]["s5"]["checkpoint"])
    ck = ck if ck.is_absolute() else C.PROJECT_ROOT / ck
    det = RidgeDetector.from_checkpoint(ck, cfg, a.device)
    tag = a.tag or det.version
    out = Path(f"{a.data}_detections") / tag
    info = {"tag": tag, "checkpoint": str(ck.relative_to(C.PROJECT_ROOT)) if ck.is_relative_to(C.PROJECT_ROOT) else str(ck),
            "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(), "overrides": a.override,
            "pipeline_config_hash": cfg["_meta"]["effective_hash"], "band_hz": band_hz(cfg),
            "peak_threshold": cfg["decisions"]["s6"]["peak_threshold"], "torch": torch.__version__, "files": {}}
    for sp, stem in TEST_FILES:
        wav = a.data / "development" / sp / f"{stem}.wav"
        if not wav.exists():
            print(f"  {stem}: no audio, skipped")
            continue
        tracks, meta = detect_file(wav, cfg, det)
        write_tonals(out / sp / f"{stem}.det", tracks)
        info["files"][stem] = meta
        print(f"  {sp:13s} {stem}: {meta['n_tracks']} tracks from {meta['duration_s']:.0f} s ({meta['seconds']:.0f} s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "detect_info.json").write_text(json.dumps(info, indent=2))
    print(f"detections: {out}  ({len(info['files'])} of {len(TEST_FILES)} files)")


_GT_CACHE: dict[str, tuple] = {}   # annotation file -> (sorted GT, SNR, skipped); the SNR pass is the slow part


def load_gt(wav: Path, ann: Path) -> tuple:
    if str(ann) not in _GT_CACHE:
        gts = sorted_gts(ann)
        x, fs = read_wav(wav)
        _GT_CACHE[str(ann)] = (gts, *gt_snr_db(x, fs, gts))
    return _GT_CACHE[str(ann)]


def score_set(data: Path, det_dir: Path | None, gt: str, band, full_band: bool, min_overlap: float, dets_from_gt=None,
              min_det_s: float = 0.0) -> dict:
    per_file, by_species = {}, {}
    for sp, stem in TEST_FILES:
        wav, ann = data / "development" / sp / f"{stem}.wav", gt_path(data, sp, stem, gt)
        det_p = det_dir / sp / f"{stem}.det" if det_dir else None
        if not wav.exists() or (det_p is not None and not det_p.exists()):
            continue
        gts, snr, skipped = load_gt(wav, ann)
        exp, counts = expected_mask(gts, snr, skipped, band, full_band)
        dets = dets_from_gt(gts) if dets_from_gt else read_tonals(det_p)
        dets = [(t, f) for t, f in dets if t[-1] - t[0] >= min_det_s]
        r = score_file(dets, gts, exp, skipped, min_overlap)
        per_file[stem] = {"species": sp, **counts, **summarise([r]), "_raw": r}
        by_species.setdefault(sp, []).append(r)
    species = {sp: summarise(rows) for sp, rows in by_species.items()}
    avg = {k: float(np.mean([s[k] for s in species.values()])) for k in ("precision", "recall", "f1", "deviation_hz",
                                                                          "coverage_pct", "fragmentation")} if species else {}
    for v in per_file.values():
        v.pop("_raw")
    return {"gt": gt, "band_hz": None if full_band else list(band), "min_overlap": min_overlap, "min_detection_s": min_det_s,
            "n_files": len(per_file), "files": per_file, "species": species, "species_average": avg}


def print_table(res: dict) -> None:
    hdr = f"{'':40s} {'P':>6s} {'R':>6s} {'F1':>6s} {'dev Hz':>7s} {'cov %':>6s} {'frag':>5s}  GT exp/total"
    print(hdr)
    for stem, v in res["files"].items():
        print(f"  {stem[:38]:38s} {v['precision']:6.1f} {v['recall']:6.1f} {v['f1']:6.1f} {v['deviation_hz']:7.0f} "
              f"{v['coverage_pct']:6.1f} {v['fragmentation']:5.2f}  {v['expected']}/{v['gt_total']} "
              f"(short {v['too_short']}, low SNR {v['low_snr']}, out of band {v['outside_band']})")
    for sp, v in res["species"].items():
        print(f"{sp:40s} {v['precision']:6.1f} {v['recall']:6.1f} {v['f1']:6.1f} {v['deviation_hz']:7.0f} "
              f"{v['coverage_pct']:6.1f} {v['fragmentation']:5.2f}  {v['gt_expected']}")
    a = res["species_average"]
    if a:
        print(f"{'SPECIES AVERAGE':40s} {a['precision']:6.1f} {a['recall']:6.1f} {a['f1']:6.1f} {a['deviation_hz']:7.0f} "
              f"{a['coverage_pct']:6.1f} {a['fragmentation']:5.2f}")
    print(f"({res['n_files']} of {len(TEST_FILES)} files; GT {res['gt']}; band {res['band_hz'] or 'full'}; "
          f"min overlap {res['min_overlap']}; detections >= {res['min_detection_s'] * 1000:g} ms)")


def cmd_score(a) -> None:
    det_dir = Path(f"{a.data}_detections") / a.tag
    if not det_dir.is_dir():
        sys.exit(f"no detections in {det_dir} (run detect first)")
    res = score_set(a.data, det_dir, a.gt, band_hz(load_cfg([])), a.full_band, a.min_overlap, min_det_s=a.min_det_ms / 1000)
    print_table(res)
    name = (f"score_gt{a.gt}{f'_min{a.min_det_ms:g}ms' if a.min_det_ms else ''}{'_fullband' if a.full_band else ''}"
            f"{'' if a.min_overlap == 0.3 else f'_ov{a.min_overlap:g}'}.json")
    (det_dir / name).write_text(json.dumps(res, indent=2))
    print(f"saved {det_dir / name}")


def cmd_selftest(a) -> None:
    """The scorer must give 100 % for the GT scored against itself, and reject shifted or truncated copies."""
    band = band_hz(load_cfg([]))

    def shifted(df):
        return lambda gts: [(t, f + df) for t, f in gts]

    def first_part(frac):
        return lambda gts: [(t[t <= t[0] + frac * (t[-1] - t[0])], f[t <= t[0] + frac * (t[-1] - t[0])]) for t, f in gts]

    cases = [("identity", lambda g: g, 0.3), ("shift +300 Hz", shifted(300.0), 0.3), ("shift +400 Hz", shifted(400.0), 0.3),
             ("first 25 %", first_part(0.25), 0.3), ("first 25 %, no overlap rule", first_part(0.25), 0.0),
             ("first 40 %", first_part(0.40), 0.3)]
    for name, fn, ov in cases:
        res = score_set(a.data, None, a.gt, band, False, ov, dets_from_gt=fn)
        tot ={k: sum(v[k] for v in res["files"].values()) for k in ("detections_matched", "false_pos", "gt_matched", "gt_expected")}
        p = 100 * tot["detections_matched"] / max(1, tot["detections_matched"] + tot["false_pos"])
        r = 100 * tot["gt_matched"] / max(1, tot["gt_expected"])
        print(f"{name:30s} files {res['n_files']}  recall {r:6.2f}  precision {p:6.2f}  (GT expected {tot['gt_expected']}, "
              f"false pos {tot['false_pos']})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="run the detector on the test recordings")
    d.add_argument("--tag", help="name of the detection set (default: the checkpoint version)")
    d.add_argument("--override", action="append", default=[], help="pipeline.toml override, e.g. decisions.s6.suppress_harmonics=false")
    d.add_argument("--device", default=None)
    s = sub.add_parser("score", help="score a detection set (Li et al. 2023 protocol)")
    s.add_argument("--tag", required=True)
    for p in (s, sub.add_parser("selftest", help="check the scorer on the annotations themselves")):
        p.add_argument("--gt", choices=["2011", "2025"], default="2011", help="2011: as Li et al.; 2025: revised annotations")
    s.add_argument("--min-overlap", type=float, default=0.3, help="Li et al.: detection must cover > 30%% of the whistle (0 = Silbido default)")
    s.add_argument("--min-det-ms", type=float, default=0.0, help="drop detections shorter than this (Silbido outputs >= 150 ms)")
    s.add_argument("--full-band", action="store_true", help="do not exclude GT whistles outside the detector band")
    a = ap.parse_args(argv)
    {"detect": cmd_detect, "score": cmd_score, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
