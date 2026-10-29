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

## Things to check after building
* **Background banks are rebuilt, not copied.** Window screening runs inside the band, so the 5–24 kHz banks may keep
  different windows than v2c's. That is consistent but means the two detectors were not trained on identical backgrounds.
* **Label-fix threshold.** `bg_ridge_db = 4.0` was calibrated on the 2–24 kHz band (2.19 % of real-background nodes
  marked). `build-ridge-masks` prints the marked fraction; if it is far from 2 %, the threshold needs recalibrating.
* **S6 threshold.** `decisions.s6.peak_threshold` is still v2c's 0.5. `eval-synth --calibrate` reports the calibrated
  value; copy it into `pipeline.toml` once, before any real-data scoring.
* **Not comparable by `compare-synth`.** The paired comparison against v2c needs both models on one band and one
  synthetic set, so v2c vs 5-24 numbers are separate synthetic results, not a paired test.
