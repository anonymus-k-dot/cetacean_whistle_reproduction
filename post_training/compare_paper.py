"""Our V1 results next to the paper's (Table 1 row L = 64, Table 3; transcribed from the paper PDF, pages 10 and 12).

Reads models/vaegmm/<run_id>/eval/table1_recon.json and table3_fig12_K<K>.json; writes post_training/out/paper_comparison.csv
and prints both tables. The paper's Table 3 uses K = 10,000 and 228,673 images (all encounters); ours K = 3,000 and the
63,490 images of encounter 191, so N differs by design. Our primary Table 3 weights are eq. 10 (configs/vaegmm.toml).

    python post_training/compare_paper.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whistlepipe.vaegmm.cli import EXPERIMENTS  # noqa: E402
from whistlepipe.vaegmm.config import load_vaegmm_config, main_k, resolve  # noqa: E402

RUN_ID = EXPERIMENTS["V1"]["run_id"]
OUT = ROOT / "post_training" / "out"

PAPER_T1_L64 = {"MSE": 0.5778, "PSNR_dB": 44.5274, "SSIM": 0.9861, "KL_latent": 43.1929}
# no: (N, W1 [ms for L, Hz otherwise], FFD, recall %, precision %, F1 %)
PAPER_T3 = {1: (30593, 1.0075, 0.0899, 98.94, 83.95, 90.83), 2: (8202, 1.7744, 1.4866, 99.33, 83.19, 90.54),
            3: (21409, 1.2985, 0.7703, 99.18, 87.31, 92.87), 4: (33972, 5.9156, 1.1677, 98.71, 88.07, 93.09),
            5: (21056, 3.4688, 1.2994, 98.81, 89.03, 93.66), 6: (28503, 3.3562, 0.2531, 99.07, 85.37, 91.71),
            7: (58918, 1.4625, 0.0599, 99.01, 85.08, 91.51), 8: (59624, 2.8969, 0.1105, 99.04, 84.96, 91.46),
            9: (22851, 2.9156, 0.3384, 98.69, 84.93, 91.29), 10: (19354, 3.9938, 0.6993, 98.71, 84.55, 91.08),
            11: (27062, 3.5906, 0.2902, 98.74, 86.04, 91.95), 12: (10712, 5.6063, 1.1084, 98.17, 90.84, 94.36),
            13: (48642, 3.5156, 0.1313, 98.93, 82.66, 90.07)}
# Table 6, VAE-GMM L = 64 (page 14): params, FLOPs, inference latency (B = 1), per-image latency at B = 16 and B = 32
PAPER_T6 = {"decoder_params": 368e3, "decoder_FLOPs": 557e6, "B1_latency_ms": 1.7, "B16_per_image_ms": 0.11,
            "B32_per_image_ms": 0.054}


def main() -> int:
    cfg = load_vaegmm_config()
    K = main_k(cfg)
    ev = resolve(cfg["paths"]["runs_dir"]) / RUN_ID / "eval"
    rows = []
    t1 = json.loads((ev / "table1_recon.json").read_text(encoding="utf-8"))
    print("Table 1, L = 64          ours        paper")
    for k, ours in (("MSE", t1["MSE_per_image_sse"]), ("PSNR_dB", t1["PSNR_dB"]), ("SSIM", t1["SSIM"]), ("KL_latent", t1["KL_latent"])):
        print(f"  {k:10s} {ours:14.4f} {PAPER_T1_L64[k]:12.4f}")
        rows.append({"table": "1", "row": "L=64", "metric": k, "ours": ours, "paper": PAPER_T1_L64[k]})
    t3 = json.loads((ev / f"table3_fig12_K{K}.json").read_text(encoding="utf-8"))
    w = t3["weights_primary"]
    print(f"\nTable 3 (ours: K = {K}, weights {w}; paper: K = 10,000)")
    print(" no feature  range    N ours/paper    W1 ours/paper        FFD ours/paper   recall %       precision %    F1 %")
    for r in t3["rows"]:
        p = PAPER_T3[r["no"]]
        m = r[w]
        ours = (r["N"], m["W1"], m["FFD"], 100 * m["recall"], 100 * m["precision"], 100 * m["f1"])
        print(f" {r['no']:2d} {r['feature']:7s} {r['c1']:3}-{r['c2']:<3} {ours[0]:6d}/{p[0]:<6d} {ours[1]:7.3f}/{p[1]:<7.3f} {m['W1_unit']:2s} "
              f"{ours[2]:7.3f}/{p[2]:<7.3f} {ours[3]:6.2f}/{p[3]:<6.2f} {ours[4]:6.2f}/{p[4]:<6.2f} {ours[5]:6.2f}/{p[5]:<6.2f}")
        for name, o, pv in zip(("N", f"W1_{m['W1_unit']}", "FFD", "recall_pct", "precision_pct", "F1_pct"), ours, p):
            rows.append({"table": "3", "row": f"condition {r['no']} ({r['feature']} {r['c1']}-{r['c2']})", "metric": name,
                         "ours": o, "paper": pv})
    z = t3["fig12_lnFFD"]["Z"]
    diag = [z[i][i] for i in range(len(z))]
    off = [z[i][j] for i in range(len(z)) for j in range(len(z)) if i != j]
    print(f"\nFig. 12: diagonal ln FFD {min(diag):.2f} .. {max(diag):.2f}; off-diagonal {min(off):.2f} .. {max(off):.2f}; "
          f"rows whose diagonal is the row minimum: {sum(z[i][i] == min(z[i]) for i in range(len(z)))} of {len(z)}")
    t6p = ev / "table6_cost.json"
    if t6p.exists():
        t6 = json.loads(t6p.read_text(encoding="utf-8"))
        print(f"\nTable 6 (ours on {t6.get('gpu_name') or t6['device']}; the paper does not name its timing GPU)")
        print("  metric                 decoder only   GMM sampling + decoder   paper")
        ours = {"decoder_params": (t6["decoder_params"], None), "decoder_FLOPs": (t6["decoder_table6_flops"], None)}
        for B, key in ((1, "B1_latency_ms"), (16, "B16_per_image_ms"), (32, "B32_per_image_ms")):
            what = "batch_s" if B == 1 else "per_image_s"
            ours[key] = tuple(1e3 * t6[f"B{B}_{m}_{what}"] for m in ("decoder_only", "gmm_sampling_plus_decoder"))
        for key, (a, b) in ours.items():
            print(f"  {key:22s} {a:12.4g}   {'' if b is None else f'{b:12.4g}':>22s}   {PAPER_T6[key]:.4g}")
            rows.append({"table": "6", "row": "VAE-GMM L=64 decoder only", "metric": key, "ours": a, "paper": PAPER_T6[key]})
            if b is not None:
                rows.append({"table": "6", "row": "VAE-GMM L=64 GMM sampling + decoder", "metric": key, "ours": b,
                             "paper": PAPER_T6[key]})
        wp = OUT / "table6_warm.json"
        if wp.exists():
            w6 = json.loads(wp.read_text(encoding="utf-8"))
            print("  warm GPU, no pause (post_training/table6_warm.py):")
            for B, key in ((1, "B1_latency_ms"), (16, "B16_per_image_ms"), (32, "B32_per_image_ms")):
                what = "batch_s" if B == 1 else "per_image_s"
                a, b = (1e3 * w6[f"B{B}_{m}_{what}"] for m in ("decoder_only", "gmm_sampling_plus_decoder"))
                print(f"  {key:22s} {a:12.4g}   {b:22.4g}   {PAPER_T6[key]:.4g}")
                for row, v in (("decoder only, warm", a), ("GMM sampling + decoder, warm", b)):
                    rows.append({"table": "6", "row": f"VAE-GMM L=64 {row}", "metric": key, "ours": v, "paper": PAPER_T6[key]})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "paper_comparison.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["table", "row", "metric", "ours", "paper"])
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", (OUT / "paper_comparison.csv").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
