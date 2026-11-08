"""Post-fit check of a V1 GMM (default K = 3,000): the diagnostics of spec §3.5 plus three of this project's own.

Run after `python whistle.py fit-gmm --experiment V1 --K 3000`. Reads models/vaegmm/<run_id>/gmm_K<K>[_tag]/ (gmm.npz,
gmm.json, fit_log.jsonl) and the latents; writes post_training/out/gmm_check_K<K>[_tag].json and three PNGs.

Pass criteria, declared before the main fit finished:
  * the fit files exist and gmm.json names the latents in use (SHA-256 = latents_mu.npy = cell11_report.json),
  * the mean log-likelihood is finite at every iteration and never drops by more than 1e-6 of its value (EM ascends; the
    ridge added after each M-step and float32 responsibilities allow tiny drops),
  * every covariance is positive definite,
  * [DECISION, proxy for "GMM samples look like whistles"] the 7 whistle features of decoded GMM samples are closer to the
    real ones than those of decoded N(0, I) samples (Fréchet distance of the features, FFD).
Reported only, not pass/fail: convergence and iteration count, N_k counts and histogram, eigenvalue range, how many
components are the MAP component of at least one training whistle, and a memorisation measure [DECISION]: the distance
from each GMM sample to its nearest training latent, against the leave-one-out nearest-neighbour distance among the
training latents themselves.

    python post_training/gmm_check.py [--K 3000] [--tag ""]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from whistlepipe.detector.ridgenet import resolve_device, set_deterministic, sha256_file  # noqa: E402
from whistlepipe.vaegmm.cli import EXPERIMENTS  # noqa: E402
from whistlepipe.vaegmm.config import load_vaegmm_config, main_k, resolve  # noqa: E402
from whistlepipe.vaegmm.data import training_images  # noqa: E402
from whistlepipe.vaegmm.evaluate import generated_batches, gmm_dir, load_gmm, real_features  # noqa: E402
from whistlepipe.vaegmm.features import contour_features  # noqa: E402
from whistlepipe.vaegmm.metrics import frechet_from_samples  # noqa: E402
from whistlepipe.vaegmm.train import load_trained_vae  # noqa: E402

RUN_ID = EXPERIMENTS["V1"]["run_id"]
OUT = ROOT / "post_training" / "out"
LL_DROP_TOL = 1e-6
SEED_OFFSET = 5            # [DECISION] seed + 5 for this check (evaluate.py uses + 1, + 2, + 3; figures use 7)


@torch.no_grad()
def nn_distance(Q: torch.Tensor, R: torch.Tensor, exclude_self: bool = False, chunk: int = 2048) -> np.ndarray:
    """Euclidean distance from every row of Q to its nearest row of R. The nearest row is found in float32 (expanded
    form), then its distance is recomputed exactly in float64 from the explicit difference."""
    rn = (R * R).sum(1)
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d2 = (q * q).sum(1, keepdim=True) - 2.0 * q @ R.T + rn[None]
        if exclude_self:
            ar = torch.arange(len(q), device=q.device)
            d2[ar, i + ar] = float("inf")
        j = d2.argmin(1)
        out[i:i + len(q)] = (q.double() - R[j].double()).norm(dim=1).cpu().numpy()
    return out


@torch.no_grad()
def decoded_features(decoder, source: str, n: int, seed: int, dev, gmm, latent_dim: int, thr: float) -> tuple[np.ndarray, float]:
    feats, valid = [], []
    for x in generated_batches(decoder, source, n, seed, dev, gmm, latent_dim):
        f, v = contour_features(x.to(dev), thr)
        feats.append(f.cpu().numpy())
        valid.append(v.cpu().numpy())
    f, v = np.concatenate(feats), np.concatenate(valid)
    return f[v], float(v.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    set_deterministic()
    cfg = load_vaegmm_config()
    K = args.K or main_k(cfg)
    name = f"K{K}{('_' + args.tag) if args.tag else ''}"
    run = resolve(cfg["paths"]["runs_dir"]) / RUN_ID
    gd = gmm_dir(cfg, RUN_ID, K, args.tag)
    dev = resolve_device(cfg["train"]["device"])
    seed = int(cfg["decisions"]["train"]["seed"]) + SEED_OFFSET
    thr = float(cfg["paper"]["contour_threshold"])
    checks, rep = {}, {"run_id": RUN_ID, "gmm_dir": str(gd.relative_to(ROOT)).replace("\\", "/"), "K": K, "tag": args.tag}

    # ---- fit record and lineage
    checks["files_present"] = all((gd / f).exists() for f in ("gmm.npz", "gmm.json", "fit_log.jsonl"))
    if not checks["files_present"]:
        raise SystemExit(f"{gd}: gmm.npz / gmm.json / fit_log.jsonl missing (fit not finished?)")
    meta = json.loads((gd / "gmm.json").read_text(encoding="utf-8"))
    lat = run / "latents_mu.npy"
    sha_lat = sha256_file(lat)
    cell11 = json.loads((OUT / "cell11_report.json").read_text(encoding="utf-8"))
    checks["latents_match"] = meta["latents_sha256"] == sha_lat == cell11["latents"]["sha256"]
    rep["fit"] = {k: meta.get(k) for k in ("n_components", "reg_rel", "reg_abs", "max_iter", "tol", "tol_mode", "empty_nk", "seed",
                                           "converged", "n_iter", "final_mean_loglik", "fit_wall_s")}
    rep["fit"]["gmm_npz_sha256"] = sha256_file(gd / "gmm.npz")
    rep["fit"]["gmm_npz_mb"] = round((gd / "gmm.npz").stat().st_size / 1e6, 1)
    checks["n_components"] = meta["n_components"] == K

    # ---- log-likelihood curve
    hist = [json.loads(line) for line in (gd / "fit_log.jsonl").read_text(encoding="utf-8").splitlines()]
    it = np.array([h["iter"] for h in hist])
    ll = np.array([h["mean_loglik"] for h in hist], dtype=np.float64)
    frozen = np.array([h["n_frozen"] for h in hist])
    step = np.diff(ll)
    rel_drop = -step / np.abs(ll[1:])
    checks["loglik_finite"] = bool(np.isfinite(ll).all())
    checks["loglik_no_drop_gt_1e-6_rel"] = bool((rel_drop <= LL_DROP_TOL).all())
    rep["loglik"] = {"first": float(ll[0]), "last": float(ll[-1]), "n_iterations_logged": int(len(ll)),
                     "n_decreasing_steps": int((step < 0).sum()), "largest_relative_drop": float(max(rel_drop.max(), 0.0)),
                     "last_relative_change": float(abs(step[-1]) / abs(ll[-1])) if len(step) else None,
                     "gain_last_100_iterations": float(ll[-1] - ll[max(0, len(ll) - 101)]),
                     "max_n_frozen": int(frozen.max()), "n_frozen_last": int(frozen[-1])}

    # ---- the GMM itself
    g = load_gmm(cfg, RUN_ID, K, args.tag)      # load() runs the Cholesky factorisation of every covariance
    mu = np.load(lat)
    diag = g.diagnostics(len(mu))
    checks["covariances_positive_definite"] = diag["eig_min"] > 0
    rep["diagnostics"] = diag
    nk = g.weights_.cpu().numpy() * len(mu)
    lab, resp = g.predict(mu)
    occ = np.bincount(lab, minlength=K)
    rep["map_assignment"] = {"n_components_map_of_at_least_one_whistle": int((occ > 0).sum()),
                             "map_count_quantiles_0_50_95_100": np.quantile(occ, [0, .5, .95, 1]).tolist(),
                             "responsibility_of_map_median": float(np.median(resp)),
                             "frac_whistles_responsibility_above_0.99": float((resp > 0.99).mean())}

    # ---- memorisation measure (latent space)
    Xt = torch.as_tensor(mu, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    zg = torch.cat([g.sample(min(4096, len(mu) - i), gen)[0] for i in range(0, len(mu), 4096)])
    zp = torch.randn(len(mu), mu.shape[1], generator=gen, device=dev)
    d_loo = nn_distance(Xt, Xt, exclude_self=True)
    d_gmm = nn_distance(zg, Xt)
    d_pri = nn_distance(zp, Xt)
    q5 = float(np.quantile(d_loo, 0.05))
    rep["memorisation_latent"] = {
        "nn_distance_median_training_leave_one_out": float(np.median(d_loo)),
        "nn_distance_median_gmm_samples": float(np.median(d_gmm)),
        "nn_distance_median_prior_samples": float(np.median(d_pri)),
        "ratio_median_gmm_to_training": float(np.median(d_gmm) / np.median(d_loo)),
        "frac_gmm_samples_nearer_than_5pct_of_training_nn": float((d_gmm < q5).mean()),
        "note": "Euclidean, latent means; n samples = n training latents; ratio << 1 means samples sit on training latents"}

    # ---- decoded samples: whistle features vs the real ones
    images = training_images(cfg)
    vae, spec = load_trained_vae(run, dev)
    rf = real_features(cfg, images, dev)
    rf = rf[np.isfinite(rf).all(1)]
    fg, vg = decoded_features(vae.decoder, "gmm", len(mu), seed + 100, dev, g, spec["latent_dim"], thr)
    fp, vp = decoded_features(vae.decoder, "prior", len(mu), seed + 200, dev, None, spec["latent_dim"], thr)
    ffd_g, ffd_p = frechet_from_samples(rf, fg), frechet_from_samples(rf, fp)
    checks["gmm_features_closer_to_real_than_prior"] = bool(ffd_g < ffd_p)
    rep["decoded_features"] = {"FFD_real_vs_gmm": ffd_g, "FFD_real_vs_prior": ffd_p,
                               "frac_with_contour_real": float(len(rf) / len(images)), "frac_with_contour_gmm": vg,
                               "frac_with_contour_prior": vp,
                               "median_L_real_gmm_prior": [float(np.median(a[:, 0])) for a in (rf, fg, fp)],
                               "note": "7 whistle features (threshold 0.05) of n decoded samples; FFD = Fréchet distance"}

    # ---- figures
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs[0, 0].plot(it, ll, lw=1)
    axs[0, 0].set(title="mean log-likelihood per whistle", xlabel="EM iteration")
    if len(step):
        axs[0, 1].semilogy(it[1:], np.maximum(np.abs(step), 1e-16) / np.abs(ll[1:]), lw=0.8)
        axs[0, 1].axhline(float(cfg["paper"]["em_tol"]), color="grey", ls="--", lw=0.8, label="tolerance 1e-8")
        axs[0, 1].legend(fontsize=8)
    axs[0, 1].set(title="|relative change| per iteration", xlabel="EM iteration")
    axs[1, 0].plot(it, frozen, lw=1)
    axs[1, 0].set(title="components frozen (N_k < 1)", xlabel="EM iteration")
    axs[1, 1].hist(np.clip(nk, 0.5, None), bins=np.logspace(np.log10(0.5), np.log10(nk.max()), 50))   # N_k < 0.5 in the first bin
    axs[1, 1].set_xscale("log")
    axs[1, 1].axvline(mu.shape[1] + 1, color="grey", ls="--", lw=0.8, label=f"N_k = {mu.shape[1] + 1}")
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].set(title=f"N_k = pi_k N over the {K} components", xlabel="N_k")
    for a in axs.flat:
        a.grid(alpha=0.3)
    fig.suptitle(f"{RUN_ID} GMM {name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"gmm_check_{name}_fit.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(np.log10(max(min(d_gmm.min(), d_loo.min()), 1e-3)), np.log10(d_pri.max()), 80)
    for d, lab_ in ((d_loo, "training latent -> nearest other training latent"), (d_gmm, "GMM sample -> nearest training latent"),
                    (d_pri, "N(0, I) sample -> nearest training latent")):
        ax.hist(d, bins=bins, histtype="step", lw=1.2, label=lab_)
    ax.set(xscale="log", xlabel="Euclidean distance in latent space", ylabel="count", title=f"memorisation measure, GMM {name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"gmm_check_{name}_nn.png", dpi=100)
    plt.close(fig)

    with torch.no_grad():
        idx = np.random.default_rng(0).choice(len(images), 12, replace=False)     # the training sample-grid whistles
        real = torch.from_numpy(images.gather(idx))[:, None].to(dev)
        k_map, _ = g.predict(mu[idx])
        rows = [("real whistle", real),
                ("sample of its\nMAP component", vae.decoder(g.sample(12, gen, components=k_map)[0])),
                ("GMM sample", vae.decoder(g.sample(12, gen)[0])), ("GMM sample", vae.decoder(g.sample(12, gen)[0])),
                ("N(0, I) sample", vae.decoder(torch.randn(12, spec["latent_dim"], generator=gen, device=dev)))]
    fig, axs = plt.subplots(len(rows), 12, figsize=(13.2, 1.25 * len(rows) + 0.4))
    for r, (title, x) in enumerate(rows):
        for c in range(12):
            axs[r, c].imshow(x[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
        axs[r, 0].set_ylabel(title, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"GMM {name}: decoded samples (final.pt, epoch {torch.load(run / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False)['epoch']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"gmm_check_{name}_samples.png", dpi=90)
    plt.close(fig)

    rep["checks"] = checks
    rep["passed"] = all(checks.values())
    (OUT / f"gmm_check_{name}.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    print("PASSED" if rep["passed"] else "FAILED: " + ", ".join(k for k, v in checks.items() if not v))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
