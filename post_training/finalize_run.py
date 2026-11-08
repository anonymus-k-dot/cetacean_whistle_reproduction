"""Step 0 (no Colab cell): finalise the V1 run, which the owner stopped after epoch 101 of the planned 150.

train_vae() writes checkpoints/final.pt and closes train_info.json only when the last planned epoch finishes, so a stopped
run needs this step before anything downstream (encode, fit-gmm, vae-eval) reads it. The script
  1. checks that checkpoints/last.pt holds the last epoch with a summary in train_log.jsonl and the run spec of train_info.json,
  2. copies last.pt to final.pt byte for byte (final model = last finished epoch, no model selection, spec §2.1),
  3. records the early stop in train_info.json (status "stopped_early"; every field written on Colab is kept).
Running it again changes nothing once final.pt matches last.pt.

    python post_training/finalize_run.py          (from the project root, or from anywhere)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from whistlepipe.vaegmm.cli import EXPERIMENTS  # noqa: E402
from whistlepipe.vaegmm.config import load_vaegmm_config, resolve  # noqa: E402
from whistlepipe.vaegmm.train import check_final_epochs  # noqa: E402

RUN_ID = EXPERIMENTS["V1"]["run_id"]
DECISION_DATE = "2026-11-08"          # owner: "the training has stopped, the last.pt is the final.pt"
STOP_NOTE = ("Colab cell 9 stopped during epoch 102 on 2026-11-03 (last write to the Drive run folder 13:55 IST). The owner "
             "chose not to resume to the planned 150 epochs; final.pt is the last finished epoch (no model selection). "
             "The Drive copy was checked byte-identical to this folder on 2026-11-08.")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    cfg = load_vaegmm_config()
    run = resolve(cfg["paths"]["runs_dir"]) / RUN_ID
    ck, info_path = run / "checkpoints", run / "train_info.json"
    last, final = ck / "last.pt", ck / "final.pt"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["status"] == "complete":
        print(f"{RUN_ID} finished all planned epochs; nothing to finalise")
        return 0

    st = torch.load(last, map_location="cpu", weights_only=False)
    epoch, step, spec = int(st["epoch"]), int(st["global_step"]), st["spec"]
    summaries, tail = {}, []
    for line in (run / "train_log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("epoch_summary"):
            summaries[r["epoch"]] = r
    last_summary = max(summaries)
    for line in (run / "train_log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("epoch_summary") and r["epoch"] > last_summary:
            tail.append(r)
    if epoch != last_summary:
        raise SystemExit(f"last.pt is epoch {epoch}, but the last epoch summary in train_log.jsonl is {last_summary}")
    if spec != info["spec"]:
        raise SystemExit(f"last.pt run spec differs from train_info.json:\n{spec}\nvs\n{info['spec']}")
    if epoch >= spec["epochs"]:
        raise SystemExit(f"last.pt is the last planned epoch ({epoch}); train_vae --resume would finish the run normally")

    sha_last = sha256(last)
    if final.exists():
        if sha256(final) != sha_last:
            raise SystemExit(f"{final} exists and differs from last.pt; not overwritten")
        print("final.pt already present, identical to last.pt")
    else:
        tmp = final.with_suffix(".tmp")
        shutil.copyfile(last, tmp)
        os.replace(tmp, final)
        print(f"wrote {final.relative_to(ROOT)} (byte copy of last.pt)")
    if sha256(final) != sha_last:
        raise SystemExit("final.pt does not match last.pt after the copy")

    if info["status"] == "stopped_early" and info.get("final_checkpoint_sha256") == sha_last:
        print("train_info.json already records the early stop")
    else:
        health = check_final_epochs(run)
        info.update({
            "status": "stopped_early",
            "epochs_planned": int(spec["epochs"]), "epochs_trained": epoch, "final_epoch": epoch, "final_global_step": step,
            "final_checkpoint": "checkpoints/final.pt = byte copy of checkpoints/last.pt",
            "final_checkpoint_sha256": sha_last,
            "stopped": {"by": "owner", "decision_date": DECISION_DATE, "status_before": info["status"], "note": STOP_NOTE,
                        "log_tail": (f"train_log.jsonl ends with {len(tail)} step line(s) of the unfinished epoch "
                                     f"{last_summary + 1} (no summary, no checkpoint); readers use epoch summaries only")},
            "final_epoch_check": health,
            "finalized_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finalized_by": "post_training/finalize_run.py"})
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"updated {info_path.relative_to(ROOT)}: status stopped_early")
    print(f"{RUN_ID}: final model = epoch {epoch} of {spec['epochs']} planned (step {step}), SHA-256 {sha_last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
