# Detector paper: what is left

Status 2026-10-31. Short paper (2–4 pages) on the replacement detector (RidgeNet S5 + S6 tracker), two configurations:
2–24 kHz (`ridgenet-v2c`, used for the VAE-GMM dataset) and 5–24 kHz (`ridgenet-5-24`, this folder, for DCLDE 2011).

Done so far: 5–24 kHz model trained and calibrated (tau 0.4); DCLDE 2011 benchmark (`dclde2011.py`) built, self-tested
and run on the 9 Li et al. (2023) test files. Main result (harmonics kept, detections >= 150 ms, 2011 annotations,
5–24 kHz, species average): **P 53.6, R 84.9, F1 65.2** (Li et al. 2023 best: 88.8 / 85.8 / 87.2, over 5–50 kHz).
Score files: Google Drive `MyDrive/whistle_detector_5_24/dclde2011_detections/<tag>/score_*.json`.

## 1. Silbido baseline on the same 9 files (next)
The fair comparison: Silbido's graph search scored by `dclde2011.py` with the same 5–24 kHz rules.
- [ ] Find MATLAB access (university licence / MATLAB Online). If none, decide on an alternative.
- [ ] Silbido 3.0 is in the NOAA bucket (`dclde/2011/dclde_2011/software/silbido-3.0.zip`) and on GitHub.
- [ ] Run the graph search with its defaults (`odontocete.xml`) on the 9 recordings; save the detections as
      `<data>_detections/silbido-graph/<species>/<stem>.det`.
- [ ] `python mod_detector_5_24/dclde2011.py score --tag silbido-graph` (no `--min-det-ms`: Silbido already outputs
      >= 150 ms), plus the `--full-band` and `--gt 2025` variants.

## 2. Keep the results in the repository
- [ ] Copy the `score_*.json` files from Drive into `mod_detector_5_24/results/dclde2011/<tag>/` and commit them.
- [ ] The Colab runs used a bundle flagged "uncommitted changes" (built 2026-10-29T18:45:17Z); its file hashes match
      the code committed with this file. Rebuild the bundle from a clean commit before any new run you will report.

## 3. Real-data test of the 2–24 kHz detector on 2017 recordings
- [ ] Choose 5–10 files from the 2017 encounters, **excluding the 56 background files** in `background_files.csv`.
- [ ] Choose an annotation tool (e.g. Silbido's annotator, Raven) and hand-label the whistles. Decide who labels.
- [ ] Extend the scorer to 500 kHz 6-channel FLAC input and the 2–24 kHz band (scoring itself works on any
      Silbido-format files).
- [ ] Only this in-domain number is quoted in the VAE-GMM paper.

## 4. Precision (optional, strengthens the paper)
- [ ] Look at false positives on `Qx-Tt-SCI0608-N1-060814-121518` (P 29 %) and `Qx-Dc-CC0411-TAT11-CH2-041114-154040-s`
      (P 32 %, CalCOFI: echosounder pings per the corpus README).
- [ ] Any tuning only on DCLDE development files **outside** the 9 test files, never on the test files.

## 5. Figures and writing
- [ ] Figure: spectrogram excerpt with annotations vs detections (one per species or one representative).
- [ ] Table: per-species P / R / F1 / deviation / coverage / fragmentation, plus the variant rows.
- [ ] Draft the paper. Framing: tiny model (~14k parameters), trained only on synthetic whistles over 2017 false
      killer whale backgrounds, never saw DCLDE data; recall on par with Li et al., precision is the limitation.

## Must be stated in the paper
- Band 5–24 kHz: annotated whistles not entirely inside it are excluded (about 25 % of them); `--full-band` result
  reported too (F1 62.1).
- Harmonics kept for the benchmark (DCLDE annotates harmonics); the VAE-GMM tracker suppresses them.
- 150 ms minimum detection length = Silbido's default, not tuned; the no-minimum score (F1 46.5) is reported too.
- Li et al.'s numbers are over 5–50 kHz with their own setup, so they are context, not a direct comparison.
