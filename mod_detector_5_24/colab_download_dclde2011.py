# Colab cell: download the DCLDE 2011 test set of Li et al. (2023, JASA, doi:10.1121/10.0020274) from NOAA's public
# bucket into Google Drive: 9 recordings from the development folder, the 2011 annotations (what Li et al. scored
# against), the revised 2025 annotations, the corpus README and the silbidopy reader for the .ann format (~0.82 GB).
# Paste this whole file into ONE Colab cell and run it. Nothing touches your own computer.
# Safe to re-run: files already in Drive with the right size and MD5 are skipped; a broken download is retried.
# Data: Roch et al. (2025) DCLDE 2011 Conference Data, doi:10.25921/wsnc-js16 (research and educational use).
import hashlib, os, shutil, time, urllib.request
from google.colab import drive

DRIVE_DIR = "/content/drive/MyDrive/whistle_detector_5_24/dclde2011"   # where the files end up (bucket layout kept)
URL = "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2011/dclde_2011/"
TMP = "/content/_dl_dclde2011"

# path relative to the bucket folder, size in bytes, MD5 (from the bucket listing).
# development/<species>/X.ann = 2011 annotations; annotations2025/development/<species>/X.ann = revised 2025 annotations.
# All 9 recordings are 192 kHz wav; none has a reliability footnote in the corpus README.
FILES = """
development/bottlenose/Qx-Tt-SCI0608-N1-060814-121518.wav,145859024,016bde125be1e6a9c31bbf8731ba234b
development/bottlenose/Qx-Tt-SCI0608-N1-060814-121518.ann,1138112,9a8a2d4893289df52f3d5b222ded7179
annotations2025/development/bottlenose/Qx-Tt-SCI0608-N1-060814-121518.ann,1049118,992cdeee704b222fb4b62347c09c6670
development/bottlenose/palmyra092007FS192-070924-205305.wav,95821868,c7de7f784e6e97d6683a97634914ba61
development/bottlenose/palmyra092007FS192-070924-205305.ann,4247980,6f5d000a72e9262ceb1bf6a2d01c09d4
annotations2025/development/bottlenose/palmyra092007FS192-070924-205305.ann,4108178,2ec45954de0117c23042bcab431939c0
development/bottlenose/palmyra092007FS192-070924-205730.wav,57385004,5913fa93ade7ffd494de84c9efde6d8b
development/bottlenose/palmyra092007FS192-070924-205730.ann,2568672,54ab9b5614bbf935c86c9b1fcba1c9c3
annotations2025/development/bottlenose/palmyra092007FS192-070924-205730.ann,2569666,29b7b9f58466256068e14d32c1780b17
development/common/Qx-Dc-CC0411-TAT11-CH2-041114-154040-s.wav,115200044,e4380e4e3809a42743800d6bb89456a8
development/common/Qx-Dc-CC0411-TAT11-CH2-041114-154040-s.ann,5249596,ed229e8241197508d36ac4973b3141dd
annotations2025/development/common/Qx-Dc-CC0411-TAT11-CH2-041114-154040-s.ann,5217398,2b325b2d09502e4001ffed240750096f
development/common/QX-Dc-FLIP0610-VLA-061015-165000.wav,115200044,1cf400d3e834f8e9bbadf1e9afbc2b06
development/common/QX-Dc-FLIP0610-VLA-061015-165000.ann,3670027,91a0d97ebe82ff023f7b6375decf5f4d
annotations2025/development/common/QX-Dc-FLIP0610-VLA-061015-165000.ann,3761771,fa636e332b0abaa64454d7588c545a03
development/melon-headed/palmyra092007FS192-071004-032342.wav,76161068,002bb860f8bbe07dab66fa2ffcb2c6da
development/melon-headed/palmyra092007FS192-071004-032342.ann,1811336,60739fe84a64051af77f91beb6fb1255
annotations2025/development/melon-headed/palmyra092007FS192-071004-032342.ann,1811336,60739fe84a64051af77f91beb6fb1255
development/melon-headed/palmyra102006-061020-204327_4.wav,25034796,1d18c137211b6bbb450747bdef77443f
development/melon-headed/palmyra102006-061020-204327_4.ann,5040324,d690d3b6c710e1989171c004f0fa6ebc
annotations2025/development/melon-headed/palmyra102006-061020-204327_4.ann,5040324,d690d3b6c710e1989171c004f0fa6ebc
development/spinner/palmyra092007FS192-070927-224737.wav,54794540,1785cc925537ad78fb10bc6cf8018cfd
development/spinner/palmyra092007FS192-070927-224737.ann,8118024,70266825517e7e87cd268b1f77c31d66
annotations2025/development/spinner/palmyra092007FS192-070927-224737.ann,8118024,70266825517e7e87cd268b1f77c31d66
development/spinner/palmyra102006-061103-213127_4.wav,60522540,50595518e6a39e42d05a68809d5edb16
development/spinner/palmyra102006-061103-213127_4.ann,4942180,b5a733b4be8bfa6b23eed52b25e29116
annotations2025/development/spinner/palmyra102006-061103-213127_4.ann,4942180,b5a733b4be8bfa6b23eed52b25e29116
DCLDE2011-README.md,46860,11450fd6f5d64fb5ed492371f79f2008
software/silbidopy-commit-64da6e0fbca86a35725dea4cec9fab9d049bd5ec.zip,20384,8cb08348b53b04ce7852a033b5e0dab6
"""
files = [(p, int(s), m) for p, s, m in (l.split(",") for l in FILES.strip().splitlines())]
assert len(files) == 29 and sum(p.endswith(".wav") for p, _, _ in files) == 9

drive.mount("/content/drive")
os.makedirs(TMP, exist_ok=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


todo = [(p, s, m) for p, s, m in files if not (os.path.exists(f"{DRIVE_DIR}/{p}") and os.path.getsize(f"{DRIVE_DIR}/{p}") == s)]
need_gb = sum(s for _, s, _ in todo) / 1e9
free_gb = shutil.disk_usage("/content/drive").free / 1e9
print(f"{len(files) - len(todo)} already in Drive, {len(todo)} to download ({need_gb:.2f} GB); Drive reports {free_gb:.1f} GB free")
if free_gb < need_gb + 0.5:
    print("WARNING: Drive may not have enough space; check drive.google.com/settings/storage before continuing")

t0 = time.time()
for i, (path, size, want) in enumerate(todo, 1):
    tmp = f"{TMP}/{os.path.basename(path)}"
    for attempt in range(1, 4):
        try:
            urllib.request.urlretrieve(URL + path, tmp)
            if os.path.getsize(tmp) == size and md5(tmp) == want:
                break
            print(f"  {path}: checksum mismatch (attempt {attempt}), retrying")
        except Exception as e:
            print(f"  {path}: {e} (attempt {attempt}), retrying")
        time.sleep(5)
    else:
        raise RuntimeError(f"{path}: failed 3 times; re-run the cell later")
    os.makedirs(os.path.dirname(f"{DRIVE_DIR}/{path}"), exist_ok=True)
    shutil.copyfile(tmp, f"{DRIVE_DIR}/{path}")     # verified copy goes to Drive
    os.remove(tmp)
    print(f"[{i}/{len(todo)}] {path} OK  ({(time.time() - t0) / 60:.1f} min)")

bad = [p for p, s, _ in files if not os.path.exists(f"{DRIVE_DIR}/{p}") or os.path.getsize(f"{DRIVE_DIR}/{p}") != s]
print(f"ALL {len(files)} FILES IN DRIVE: {DRIVE_DIR}" if not bad else f"missing or incomplete: {bad}")
drive.flush_and_unmount()   # make sure Drive has finished writing before the session ends
