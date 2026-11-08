"""Colab cell 10 on the laptop: loss curves, training length, latest sample grid of the V1 run.

Reads models/vaegmm/<run_id>/train_log.jsonl (epoch summaries only) and writes post_training/out/cell10_loss_curves.png.
Differences from the Colab cell: the figure is saved instead of shown; rec and KL use a log scale (rec falls from about 5,600
to 171), and the rec panel marks the encounter-191 reference levels and the epoch-40 check; the "time left" line reports
the early stop instead, because the run will not be resumed.

    python post_training/cell10_progress.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from whistlepipe.vaegmm.cli import EXPERIMENTS  # noqa: E402
from whistlepipe.vaegmm.config import load_vaegmm_config, resolve  # noqa: E402

RUN_ID = EXPERIMENTS["V1"]["run_id"]
OUT = ROOT / "post_training" / "out"
REC_AVERAGE_IMAGE, REC_PERFECT, REC_EPOCH40_CHECK = 421.9, 141.6, 395.0   # encounter 191, spec §2.1
LOG_SCALE = ("rec", "kl")


def main() -> int:
    cfg = load_vaegmm_config()
    run = resolve(cfg["paths"]["runs_dir"]) / RUN_ID
    info = json.loads((run / "train_info.json").read_text(encoding="utf-8"))
    spec = info["spec"]
    recs = [json.loads(line) for line in open(run / "train_log.jsonl", encoding="utf-8")]
    ep = pd.DataFrame([r for r in recs if r.get("epoch_summary")]).drop_duplicates("epoch", keep="last")
    cols = [c for c in ("rec", "kl", "adv", "loss_d", "d_real", "d_fake", "gp") if c in ep]

    rows = (len(cols) + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(12, 2.4 * rows), sharex=True)
    for a, c in zip(axs.flat, cols):
        a.plot(ep.epoch, ep[c], lw=1.2)
        a.set_title(c, fontsize=10)
        a.grid(alpha=0.3)
        if c in LOG_SCALE:
            a.set_yscale("log")
        if c == "rec":
            for y, lab, ls in ((REC_AVERAGE_IMAGE, "average image 421.9", "--"), (REC_PERFECT, "perfect 141.6", ":"),
                               (REC_EPOCH40_CHECK, "epoch-40 check 395", "-.")):
                a.axhline(y, color="grey", lw=0.8, ls=ls, label=lab)
            a.legend(fontsize=7, loc="upper right")
    for a in list(axs.flat)[len(cols):]:
        a.axis("off")
    for a in axs[-1]:
        a.set_xlabel("epoch")
    fig.suptitle(f"{RUN_ID}: epoch summaries (status {info['status']})", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cell10_loss_curves.png", dpi=100)
    plt.close(fig)
    print("wrote", (OUT / "cell10_loss_curves.png").relative_to(ROOT))

    done, per = int(ep.epoch.max()), float(ep.epoch_s.tail(5).mean())
    if info["status"] == "stopped_early":
        print(f"epoch {done}/{spec['epochs']} | {per / 60:.1f} min/epoch on {info['device']} | stopped early by the owner "
              f"({info['stopped']['decision_date']}); final model = epoch {info['final_epoch']}")
    else:
        left_h = (spec["epochs"] - done) * per / 3600
        print(f"epoch {done}/{spec['epochs']} | {per / 60:.1f} min/epoch | ~{left_h:.1f} h left")
    last = ep.iloc[-1]
    best = ep.loc[ep.rec.idxmin()]
    print(f"last epoch {int(last.epoch)}: rec {last.rec:.2f}, kl {last.kl:.1f}, adv {last.adv:.3f}, loss_d {last.loss_d:.3f}, "
          f"gp {last.gp:.4f} | lowest rec {best.rec:.2f} at epoch {int(best.epoch)}")
    tail = ep[ep.epoch > done - 10]
    print(f"rec over the last 10 epochs: {tail.rec.iloc[0]:.2f} -> {tail.rec.iloc[-1]:.2f} "
          f"({(tail.rec.iloc[-1] - tail.rec.iloc[0]) / (len(tail) - 1):+.2f} per epoch)")

    pngs = sorted(run.glob("samples_epoch_*.png"))
    if pngs:
        shutil.copyfile(pngs[-1], OUT / "cell10_latest_samples.png")
        print(f"latest sample grid: {pngs[-1].relative_to(ROOT)} (copied to post_training/out/cell10_latest_samples.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
