"""Colab cell 11 on the laptop: check the finished V1 run, then encode the latents (the GMM input).

Needs post_training/finalize_run.py first (it writes checkpoints/final.pt). Then, as the Colab cell:
  * prints the run status and the final checkpoint's SHA-256,
  * check_final_epochs: the last epoch must not be a loss spike (spec §2.1),
  * encode(): encoder means mu(x) of all 63,490 encounter-191 images, unrotated (spec §3.5) ->
    models/vaegmm/<run_id>/latents_mu.npy (63,490 x 64 float32) + latents_mu.json.
Added on the laptop: the local images must be the ones the run trained on (SHA-256 of dataset_info.json and images.npy
against train_info.json), and a summary of the latents goes to post_training/out/cell11_report.json.

    python post_training/cell11_check_encode.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from whistlepipe.detector.ridgenet import resolve_device, sha256_file  # noqa: E402
from whistlepipe.vaegmm.cli import EXPERIMENTS  # noqa: E402
from whistlepipe.vaegmm.config import load_vaegmm_config, resolve  # noqa: E402
from whistlepipe.vaegmm.data import training_images  # noqa: E402
from whistlepipe.vaegmm.evaluate import encode  # noqa: E402
from whistlepipe.vaegmm.train import check_final_epochs  # noqa: E402

RUN_ID = EXPERIMENTS["V1"]["run_id"]
OUT = ROOT / "post_training" / "out"


def main() -> int:
    cfg = load_vaegmm_config()                     # local defaults: preprocessed/, models/vaegmm/, device auto
    run = resolve(cfg["paths"]["runs_dir"]) / RUN_ID
    info = json.loads((run / "train_info.json").read_text(encoding="utf-8"))
    print(info["status"], "| final checkpoint SHA-256:", info.get("final_checkpoint_sha256"))
    final = run / "checkpoints" / "final.pt"
    if not final.exists():
        raise SystemExit("no checkpoints/final.pt: run post_training/finalize_run.py first")
    if sha256_file(final) != info.get("final_checkpoint_sha256"):
        raise SystemExit("checkpoints/final.pt does not match final_checkpoint_sha256 in train_info.json")

    # the data must be the run's training data (lineage recorded on Colab in train_info.json)
    t0 = time.time()
    lin = training_images(cfg).lineage()           # no hash cache: hashes images.npy (4.2 GB, ~15 s)
    want = info["dataset"]
    same = {"dataset_info_sha256": lin["dataset_info_sha256"] == want["dataset_info_sha256"],
            "encounters": lin["encounters"] == want["encounters"], "n_images": lin["n_images"] == want["n_images"],
            **{f"{e}/images.npy": lin["arrays"][e]["sha256"] == want["arrays"][e]["sha256"] for e in want["arrays"]}}
    print(f"data check ({time.time() - t0:.0f} s):", same)
    if not all(same.values()):
        raise SystemExit("local preprocessed/ data differ from the run's training data; not encoded")

    health = check_final_epochs(run)
    print(health)
    report = {"run_id": RUN_ID, "status": info["status"], "final_epoch": info.get("final_epoch"),
              "final_checkpoint_sha256": info.get("final_checkpoint_sha256"), "data_check": same, "final_epoch_check": health}
    if not health["healthy"]:
        print("NOT ENCODED: the last epoch looks like a loss spike. Compare it with the earlier epochs in cell 10 before fitting the GMM.")
    else:
        dev = resolve_device(cfg["train"]["device"])
        existed = (run / "latents_mu.npy").exists()
        t0 = time.time()
        out = encode(cfg, RUN_ID)                  # latents_mu.npy (63,490 x 64): the GMM input
        wall = time.time() - t0
        print(out, "(already present, kept)" if existed else f"({wall:.0f} s on {dev}"
              + (f", {torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ")"))
        mu = np.load(out)
        meta = json.loads((run / "latents_mu.json").read_text(encoding="utf-8"))
        if meta["checkpoint_sha256"] != info["final_checkpoint_sha256"]:
            raise SystemExit("latents_mu.json was written from a different checkpoint than final.pt")
        sd = mu.std(0)
        summ = {"path": str(out.relative_to(ROOT)).replace("\\", "/"), "shape": list(mu.shape), "dtype": str(mu.dtype),
                "all_finite": bool(np.isfinite(mu).all()), "sha256": sha256_file(out),
                "mean_abs_of_dim_means": float(np.abs(mu.mean(0)).mean()),
                "dim_std_min": float(sd.min()), "dim_std_median": float(np.median(sd)), "dim_std_max": float(sd.max()),
                "n_dims_std_below_0.01": int((sd < 0.01).sum()), "overall_std": float(mu.std()),
                "encode_wall_s": None if existed else round(wall, 1), "device": str(dev)}
        print(json.dumps(summ, indent=1))
        report["latents"] = summ
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cell11_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", (OUT / "cell11_report.json").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
