# Colab cell: DCLDE 2011 benchmark of the 5-24 kHz detector (Li et al. 2023 test set and scoring rules).
# Paste this whole file into ONE Colab cell and run it. A CPU runtime is enough (~15-25 min, mostly the scorer's SNR pass).
# Needs in Google Drive:
#   MyDrive/whistle_detector_5_24/code/whistle_code_5_24.zip   <- python mod_detector_5_24/make_colab_bundle_5_24.py
#   MyDrive/whistle_detector_5_24/dclde2011/                   <- colab_download_dclde2011.py (29 files)
# Writes detections and score_*.json files to MyDrive/whistle_detector_5_24/dclde2011_detections/<tag>/.
# Safe to re-run (detections are recomputed, scores overwritten).
import importlib, importlib.util, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from google.colab import drive

DETECT = False   # False: only re-score the detections already in Drive (skips step 2)
DRIVE = "/content/drive/MyDrive/whistle_detector_5_24"
CODE = "/content/whistle_code"
DATA = f"{DRIVE}/dclde2011"

drive.mount("/content/drive")
assert os.path.exists(f"{DRIVE}/code/whistle_code_5_24.zip"), "upload whistle_code_5_24.zip to MyDrive/whistle_detector_5_24/code/"
n_wav = len(list(Path(DATA).glob("development/*/*.wav")))
assert n_wav == 9, f"{DATA} has {n_wav} of 9 recordings: run colab_download_dclde2011.py first"

shutil.rmtree(CODE, ignore_errors=True)
with zipfile.ZipFile(f"{DRIVE}/code/whistle_code_5_24.zip") as z:
    z.extractall("/content")
OLD = "old code bundle in Drive: delete MyDrive/whistle_detector_5_24/code/whistle_code_5_24.zip and upload the new one"
assert os.path.exists(f"{CODE}/mod_detector_5_24/dclde2011.py"), OLD
assert "--min-det-ms" in open(f"{CODE}/mod_detector_5_24/dclde2011.py").read(), OLD
assert os.path.exists(f"{CODE}/mod_detector_5_24/models/detector/ridgenet-5-24/checkpoint.pt"), "bundle has no trained checkpoint"
if importlib.util.find_spec("soundfile") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=True)
sys.path.insert(0, f"{CODE}/mod_detector_5_24")
version = json.load(open(f"{CODE}/CODE_VERSION.json"))
print({k: v for k, v in version.items() if k != "files_sha256"})

import dclde2011 as D
importlib.reload(D)
data = ["--data", DATA]
KEEP = "ridgenet-5-24_keep-harmonics"

print("\n===== 1. scorer self-test (answer key against itself: expect 100 / 100) =====")
D.main(data + ["selftest"])
if DETECT:
    print("\n===== 2. detection =====")
    D.main(data + ["detect", "--tag", KEEP, "--override", "decisions.s6.suppress_harmonics=false"])   # benchmark setting
    D.main(data + ["detect"])                                                                          # tracker as used for VAE-GMM
print("\n===== 3. scores =====")
M150 = ["--min-det-ms", "150"]   # Silbido's graph search only outputs tonals >= 150 ms
for tag, extra, title in [
        (KEEP, M150, "MAIN: harmonics kept, detections >= 150 ms, 2011 answer key (as Li et al.), 5-24 kHz"),
        (KEEP, [], "harmonics kept, all detections (no 150 ms minimum)"),
        (KEEP, M150 + ["--gt", "2025"], "harmonics kept, >= 150 ms, revised 2025 answer key"),
        (KEEP, M150 + ["--min-overlap", "0"], "harmonics kept, >= 150 ms, without Li et al.'s 30 % overlap rule (Silbido default)"),
        (KEEP, M150 + ["--full-band"], "harmonics kept, >= 150 ms, all whistles counted, also those above 24 kHz"),
        ("ridgenet-5-24", M150, "harmonics suppressed (VAE-GMM tracker setting), >= 150 ms, 2011 answer key")]:
    print(f"\n--- {title}")
    D.main(data + ["score", "--tag", tag] + extra)

drive.flush_and_unmount()   # make sure Drive has finished writing
print("\nDONE: copy everything from '===== 1.' down and send it.")
