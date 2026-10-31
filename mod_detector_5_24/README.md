# mod_detector_5_24: RidgeNet with a 5–24 kHz band

A variant of the adopted detector (`ridgenet-v2c`) whose analysis band starts at **5 kHz instead of 2 kHz**, matching the
lower edge of the DCLDE 2011 whistle annotations. It is for the detector paper's benchmark only. The main pipeline,
`configs/`, `src/`, `models/detector/`, `data/` and `preprocessed/` are not touched, and the VAE-GMM dataset keeps
using `ridgenet-v2c` (2–24 kHz).

## What differs from ridgenet-v2c
| Setting | ridgenet-v2c | ridgenet-5-24 | File |
|---|---|---|---|
| S5 band (`decisions.s5.band_lo_hz`) | 2000 Hz → bins 22–256 (235 bins) | 5000 Hz → bins 54–256 (5062.5–24000 Hz, 203 bins) | `pipeline.toml` |
| Synthetic whistle range (`synth.whistles.f_range_hz`) | 2300–23700 Hz | 5300–23700 Hz | `detector.toml` |
| Output paths | `models/`, `data/`, `preprocessed/`, `preprocessing_checks/` | the same names inside `mod_detector_5_24/` | `pipeline.toml` |

Everything else is identical to v2c: architecture, conditioning, label fix, SNR mix, time reversal, scenario mix, seed,
6000 steps (checked: the effective detector config differs from `models/detector/ridgenet-v2c/detector_config.json`
only in `f_range_hz` and the version label). The upper edge stays at 24 kHz; whistles above it in DCLDE 2011 have to be
excluded at scoring time.

Changed lines in the two TOML files are marked `MOD 5-24`.

## How to use
Always go through `whistle_5_24.py`. It runs the normal CLI with this folder's configs (`--config` is refused).
```bash
python mod_detector_5_24/whistle_5_24.py build-backgrounds        # background banks screened in the 5-24 kHz band
python mod_detector_5_24/whistle_5_24.py build-ridge-masks        # label-fix masks for those banks
python mod_detector_5_24/whistle_5_24.py train --version ridgenet-5-24
python mod_detector_5_24/whistle_5_24.py eval-synth --version ridgenet-5-24 --split val --calibrate
python mod_detector_5_24/whistle_5_24.py eval-synth --version ridgenet-5-24 --split test
```
Needs `encounter_191_audio/` (the background source) and the project's Python environment. v2c took about 65 min
on the RTX 5060 laptop GPU. `train --resume` saves the training state every 250 steps and continues after an interruption.

## On Google Colab (free T4)
1. Get the 56 background files (`background_files.csv`) into Drive: paste `colab_download_backgrounds.py` into one Colab
   cell (it downloads them from NOAA's public bucket and checks each SHA-256).
2. `python mod_detector_5_24/make_colab_bundle_5_24.py` (after committing), and upload `dist/whistle_code_5_24.zip` to
   `MyDrive/whistle_detector_5_24/code/`.
3. Open `train_5_24_colab.ipynb` in Colab and follow its first cell: backgrounds, a 300-step timing test, resumable
   training, then calibration on validation and one test evaluation. Outputs land in `MyDrive/whistle_detector_5_24/models/`.

The background picker lists the whole encounter-191 folder, so the notebook creates the 307 files it never opens as
empty name-only placeholders and checks that the picker chose exactly the 56 downloaded files.

## DCLDE 2011 benchmark (Li et al. 2023 test set)
`dclde2011.py` runs the detector on the 9 test recordings and scores it with the Li et al. (2023) protocol,
re-implemented from Silbido's scoring code (details in the file's docstring).
```bash
python mod_detector_5_24/dclde2011.py selftest                       # scorer check: GT vs itself = 100 %, shifted/truncated copies rejected
python mod_detector_5_24/dclde2011.py detect                         # -> data/dclde2011_detections/ridgenet-5-24/
python mod_detector_5_24/dclde2011.py detect --tag ridgenet-5-24_keep-harmonics --override decisions.s6.suppress_harmonics=false
python mod_detector_5_24/dclde2011.py score --tag ridgenet-5-24_keep-harmonics --min-det-ms 150 [--gt 2025] [--min-overlap 0] [--full-band]
```
Data: `colab_download_dclde2011.py` (Drive) or the same files under `mod_detector_5_24/data/dclde2011/` (git-ignored).
About 3 s of CPU per minute of audio. Detections are written in Silbido's tonal format (`.det`), so Silbido's
`scoreall` can cross-check the scores.

* **Harmonics.** The DCLDE annotations include harmonics; S6 suppresses them by default (the VAE-GMM dataset wants
  fundamentals only). The benchmark run therefore turns harmonic suppression off.
* **Minimum detection length.** Silbido's graph search (the benchmark's baseline, and the extraction step in Li et al.)
  only outputs tonals of at least 150 ms; S6 keeps tracks from ~64 ms. The main score uses `--min-det-ms 150`; the
  score without it is reported too. 150 ms is Silbido's default, not a value tuned on these files.
* **Colab:** `colab_dclde2011_benchmark.py` (one cell) runs the self-test, both detection sets and all score variants
  from Drive; `DETECT = False` re-scores existing detections only.
* **Band.** Ground-truth whistles not entirely inside 5–24 kHz are excluded like low-SNR ones (not expected, not
  penalised), so the scores are not directly comparable with Li et al.'s 5–50 kHz numbers. `--full-band` scores against
  all whistles.

## Things to check after building
* **Background banks are rebuilt, not copied.** Window screening runs inside the band, so the 5–24 kHz banks may keep
  different windows than v2c's. That is consistent but means the two detectors were not trained on identical backgrounds.
* **Label-fix threshold.** `bg_ridge_db = 4.0` was calibrated on the 2–24 kHz band (2.19 % of real-background nodes
  marked). `build-ridge-masks` prints the marked fraction; if it is far from 2 %, the threshold needs recalibrating.
* **S6 threshold.** `decisions.s6.peak_threshold` is still v2c's 0.5. `eval-synth --calibrate` reports the calibrated
  value; copy it into `pipeline.toml` once, before any real-data scoring.
* **Not comparable by `compare-synth`.** The paired comparison against v2c needs both models on one band and one
  synthetic set, so v2c vs 5-24 numbers are separate synthetic results, not a paired test.
