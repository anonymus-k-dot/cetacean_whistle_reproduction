"""Table 6 supplement: generation latency with the GPU kept warm (no 0.5 s pause between trials).

`python whistle.py vae-eval cost` follows the paper's protocol (250 trials, first 50 excluded, 0.5 s between trials). On the
laptop's RTX 5060 the pause lets the GPU drop to an idle power state, so each trial includes waking it: B = 16 took 47.7 ms per
batch against 1.9 ms back to back. [DECISION] This script repeats the measurement without the pause (same trials, same
exclusion, same two modes) and reports it next to the protocol result; it is a supplement, not a replacement.
Writes post_training/out/table6_warm.json.

    python post_training/table6_warm.py
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

from whistlepipe.detector.ridgenet import resolve_device  # noqa: E402
from whistlepipe.storage import software_versions  # noqa: E402
from whistlepipe.vaegmm.cli import EXPERIMENTS  # noqa: E402
from whistlepipe.vaegmm.config import load_vaegmm_config, main_k, resolve  # noqa: E402
from whistlepipe.vaegmm.evaluate import load_gmm  # noqa: E402
from whistlepipe.vaegmm.train import load_trained_vae  # noqa: E402

RUN_ID = EXPERIMENTS["V1"]["run_id"]
OUT = ROOT / "post_training" / "out"


def main() -> int:
    cfg = load_vaegmm_config()
    K = main_k(cfg)
    dev = resolve_device(cfg["train"]["device"])
    vae, _ = load_trained_vae(resolve(cfg["paths"]["runs_dir"]) / RUN_ID, dev)
    gmm = load_gmm(cfg, RUN_ID, K)
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    res = {"run_id": RUN_ID, "K": K, "device": str(dev),
           "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
           "protocol": "250 back-to-back trials (no pause), first 50 excluded; per-image latency = batch time / B",
           "software": software_versions()}
    for B in (1, 16, 32):
        for what in ("decoder_only", "gmm_sampling_plus_decoder"):
            z0, _ = gmm.sample(B, g)
            ts = []
            for _ in range(250):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    z = gmm.sample(B, g)[0] if what != "decoder_only" else z0
                    vae.decoder(z)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            kept = np.array(ts[50:])
            res[f"B{B}_{what}_batch_s"] = float(kept.mean())
            res[f"B{B}_{what}_batch_s_median"] = float(np.median(kept))
            res[f"B{B}_{what}_per_image_s"] = float(kept.mean()) / B
            print(f"B {B:2d} {what:26s} {1e3 * kept.mean():7.3f} ms per batch, {1e3 * kept.mean() / B:6.3f} ms per image", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table6_warm.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", (OUT / "table6_warm.json").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
