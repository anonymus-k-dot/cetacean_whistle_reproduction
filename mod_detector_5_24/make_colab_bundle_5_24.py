"""Pack what the 5-24 kHz detector notebook needs on Colab into one zip (upload it to Google Drive).

    python mod_detector_5_24/make_colab_bundle_5_24.py [--out PATH]

Contents: src/whistlepipe/ (without __pycache__), configs/, whistle.py, this folder's configs, launcher, file list and
DCLDE 2011 benchmark, the trained ridgenet-5-24 checkpoint once it exists, metadata/manifest_363.csv (names of all encounter-191 files; the background picker needs the full list) and
CODE_VERSION.json (git commit, uncommitted-change flag, SHA-256 of every packed file), as colab/make_colab_bundle.py does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "dist" / "whistle_code_5_24.zip"))
    args = ap.parse_args()
    files = sorted(p for p in (ROOT / "src" / "whistlepipe").rglob("*.py") if "__pycache__" not in p.parts)
    files += sorted((ROOT / "configs").glob("*.toml")) + [ROOT / "whistle.py", ROOT / "metadata" / "manifest_363.csv"]
    files += [HERE / n for n in ("pipeline.toml", "detector.toml", "whistle_5_24.py", "background_files.csv", "dclde2011.py")]
    files += [p for p in [HERE / "models" / "detector" / "ridgenet-5-24" / "checkpoint.pt"] if p.exists()]   # after training
    version = {"git_commit": git("rev-parse", "HEAD"),
               "git_uncommitted_code_changes": bool(git("status", "--porcelain", "--", "src", "configs", "whistle.py", "mod_detector_5_24")),
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "files_sha256": {str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, "whistle_code/" + str(f.relative_to(ROOT)).replace("\\", "/"))
        z.writestr("whistle_code/CODE_VERSION.json", json.dumps(version, indent=2))
    print(f"{out}  ({out.stat().st_size / 1e6:.2f} MB, {len(files)} files, commit {version['git_commit'][:7]}, "
          f"uncommitted changes: {version['git_uncommitted_code_changes']})")


if __name__ == "__main__":
    main()
