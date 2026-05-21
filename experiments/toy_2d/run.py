#!/usr/bin/env python3
"""
Toy 2D experiment: train a score model at beta_M, then compare four methods
at a colder target beta_H.

Methods compared:
  1. Direct diffusion samples at beta_M (no correction)
  2. Classical annealed SMC with ULA mutation (baseline)
  3. Diffusion-informed SMC without SMC weight correction (ablation)
  4. Diffusion-informed SMC with SMC weight correction (proposed method)

Usage:
    cd /Users/siddhantgadkari/Desktop/JMC/YEAR_IV/FYP
    uv run python amortised_annealing/experiments/toy_2d/run.py [--energy TYPE]
                                                                [--beta-train FLOAT]
                                                                [--beta-final FLOAT]
                                                                [--n-particles INT]
                                                                [--n-steps INT]
                                                                [--seed INT]
                                                                [--no-train]
                                                                [--checkpoint PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Allow running as script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from amortised_annealing.baselines import ClassicalAnnealedSMC
from amortised_annealing.diagnostics.metrics import compute_metrics, mode_coverage_gmm
from amortised_annealing.diagnostics.plotting import (
    compare_methods,
    plot_energy_histogram,
    plot_loss_curve,
    plot_particles_2d,
    plot_energy_contour_2d,
    plot_smc_diagnostics,
)
from amortised_annealing.diffusion.reverse_sde import ReverseSDE
from amortised_annealing.diffusion.samplers import euler_maruyama_sample
from amortised_annealing.distributions import DoubleWell, GaussianMixture, Rastrigin, Ackley
from amortised_annealing.distributions.base import BoltzmannDistribution
from amortised_annealing.experiments.toy_2d.config import ExperimentConfig
from amortised_annealing.fkc.annealed_correctors import AnnealedDiffusionProposal
from amortised_annealing.fkc.fkc_annealed_proposal import FKCAnnealedDiffusionProposal
from amortised_annealing.fkc.weight_updates import annealing_weight_update
from amortised_annealing.schedules import VPSchedule, VESchedule, make_ladder
from amortised_annealing.score_models import MLPScore, train_score_model
from amortised_annealing.smc import ParticleCloud, SMCSampler


def get_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def build_energy(cfg: ExperimentConfig):
    kind = cfg.energy.type
    dim = cfg.energy.dim
    if kind == "double_well":
        return DoubleWell(dim=dim)
    elif kind == "gmm":
        return GaussianMixture.random_2d(n_components=8, radius=5.0, std=0.5)
    elif kind == "rastrigin":
        return Rastrigin(dim=dim)
    elif kind == "ackley":
        return Ackley(dim=dim)
    else:
        raise ValueError(f"Unknown energy type: {kind!r}")


def build_schedule(cfg: ExperimentConfig):
    if cfg.diffusion.schedule == "vp":
        return VPSchedule(beta_min=cfg.diffusion.beta_min, beta_max=cfg.diffusion.beta_max)
    elif cfg.diffusion.schedule == "ve":
        return VESchedule()
    else:
        raise ValueError(f"Unknown schedule: {cfg.diffusion.schedule!r}")


def make_sampler_fn(energy, beta_train: float, device: torch.device):
    """Return a function (n) -> [n, d] samples from pi_{beta_train}."""
    boltz = BoltzmannDistribution(energy, beta_train)

    def sample_fn(n: int) -> torch.Tensor:
        # Try analytic samples first
        s = boltz.sample(n, device=device)
        if s is not None:
            return s
        # Otherwise use Langevin on pi_{beta_train}
        from amortised_annealing.baselines.langevin import ULA
        ula = ULA(energy.energy, step_size=5e-3)
        x = torch.randn(n, energy.dim, device=device)
        return ula.run(x, beta_train, n_steps=50)

    return sample_fn


def build_no_correction_smc(energy, reverse_sde, beta_train, cfg, device):
    """Diffusion-informed SMC WITHOUT weight correction (ablation baseline).

    Uses diffusion mutation but weights are never updated — particles are not
    corrected for the proposal/target mismatch.
    """
    proposal = AnnealedDiffusionProposal(
        reverse_sde=reverse_sde,
        energy_fn=energy.energy,
        beta_train=beta_train,
        n_diffusion_steps=cfg.diffusion_smc.n_diffusion_steps,
        t_start=cfg.diffusion_smc.t_start,
        t_end=cfg.diffusion_smc.t_end,
        use_score_scaling=cfg.diffusion_smc.use_score_scaling,
        langevin_steps=cfg.diffusion_smc.langevin_steps,
        langevin_step_size=cfg.diffusion_smc.langevin_step_size,
    )

    def mutation_kernel(x, beta):
        return proposal.mutation_kernel(x, beta)

    def weight_update_no_correction(x, beta_prev, beta_curr):
        return torch.zeros(x.shape[0], device=x.device)  # no correction

    return SMCSampler(
        mutation_kernel=mutation_kernel,
        weight_update=weight_update_no_correction,
        energy_fn=energy.energy,
        ess_threshold=cfg.smc.ess_threshold,
        resampling_method=cfg.smc.resampling,
    )


def build_diffusion_smc(energy, reverse_sde, beta_train, cfg, device):
    """Diffusion-informed SMC WITH AIS weight correction (proposed method)."""
    proposal = AnnealedDiffusionProposal(
        reverse_sde=reverse_sde,
        energy_fn=energy.energy,
        beta_train=beta_train,
        n_diffusion_steps=cfg.diffusion_smc.n_diffusion_steps,
        t_start=cfg.diffusion_smc.t_start,
        t_end=cfg.diffusion_smc.t_end,
        use_score_scaling=cfg.diffusion_smc.use_score_scaling,
        langevin_steps=cfg.diffusion_smc.langevin_steps,
        langevin_step_size=cfg.diffusion_smc.langevin_step_size,
    )
    return SMCSampler(
        mutation_kernel=proposal.mutation_kernel,
        weight_update=proposal.weight_update,
        energy_fn=energy.energy,
        ess_threshold=cfg.smc.ess_threshold,
        resampling_method=cfg.smc.resampling,
    )


def build_fkc_smc(energy, reverse_sde, beta_train, cfg, device):
    """Diffusion SMC with path-space FKC correction (Feynman-Kac method).

    Accumulates the FKC log-weight ∫g_β(t, X_t) dt along the reverse diffusion
    trajectory. By Girsanov's theorem this IS the complete path-measure IS weight —
    no additional AIS endpoint correction is applied.
    """
    proposal = FKCAnnealedDiffusionProposal(
        reverse_sde=reverse_sde,
        energy_fn=energy.energy,
        beta_train=beta_train,
        n_diffusion_steps=cfg.diffusion_smc.n_diffusion_steps,
        t_start=cfg.diffusion_smc.t_start,
        t_end=cfg.diffusion_smc.t_end,
        use_score_scaling=cfg.diffusion_smc.use_score_scaling,
        langevin_steps=cfg.diffusion_smc.langevin_steps,
        langevin_step_size=cfg.diffusion_smc.langevin_step_size,
    )
    return SMCSampler(
        mutation_kernel=proposal.mutation_kernel,
        weight_update=proposal.weight_update,
        energy_fn=energy.energy,
        ess_threshold=cfg.smc.ess_threshold,
        resampling_method=cfg.smc.resampling,
    )


def print_metrics(name: str, metrics: dict):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")


def run(cfg: ExperimentConfig, args):
    device = get_device(cfg.device)
    print(f"\n{'='*60}")
    print(f"  Experiment: {cfg.name}")
    print(f"  Device: {device}  |  Seed: {cfg.seed}")
    print(f"{'='*60}")

    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build energy ──────────────────────────────────────────────────────
    energy = build_energy(cfg)
    beta_train = cfg.energy.beta_train
    beta_final = cfg.smc.beta_final
    print(f"\nEnergy: {cfg.energy.type} ({energy.dim}D)")
    print(f"Training beta: {beta_train}  →  Target beta: {beta_final}")

    # ── 2. Train score model ─────────────────────────────────────────────────
    schedule = build_schedule(cfg)
    model = MLPScore(
        dim=energy.dim,
        hidden_dims=cfg.model.hidden_dims,
        time_embed_dim=cfg.model.time_embed_dim,
        activation=cfg.model.activation,
    )

    checkpoint_path = out_dir / "score_model.pt"
    if args.no_train and checkpoint_path.exists():
        print(f"\nLoading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        loss_history = []
    else:
        print(f"\n── Training score model at β_M = {beta_train} ──")
        sample_fn = make_sampler_fn(energy, beta_train, device)
        t0 = time.time()
        model, loss_history = train_score_model(
            model, schedule, sample_fn, cfg.training, device
        )
        print(f"Training done in {time.time()-t0:.1f}s. Final loss: {loss_history[-1]:.4f}")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        fig = plot_loss_curve(loss_history, title=f"DSM loss (β_M={beta_train})")
        fig.savefig(out_dir / "loss_curve.png", dpi=150)
        import matplotlib.pyplot as plt; plt.close(fig)

    # ── 3. Generate beta_M samples via reverse diffusion ────────────────────
    print(f"\n── Generating {cfg.smc.n_particles} samples via reverse diffusion ──")
    reverse_sde = ReverseSDE(model, schedule)
    N = cfg.smc.n_particles

    t0 = time.time()
    x_diffusion_betaM = euler_maruyama_sample(
        reverse_sde, N, cfg.diffusion.n_reverse_steps, device,
        t_start=cfg.diffusion.t_start, t_end=cfg.diffusion.t_end,
        show_progress=True,
    )
    print(f"Diffusion sampling done in {time.time()-t0:.1f}s")

    # ── 4. Build beta ladder ──────────────────────────────────────────────────
    beta_ladder = make_ladder(
        beta_train, beta_final, cfg.smc.n_beta_steps, kind=cfg.smc.ladder
    ).to(device)
    print(f"\nBeta ladder: {beta_train:.2f} → {beta_final:.2f} ({cfg.smc.n_beta_steps} steps, {cfg.smc.ladder})")

    # ── 5. Baseline 1: Direct diffusion at beta_M (no correction) ───────────
    print("\n── Baseline: Direct diffusion samples (β_M only, no correction) ──")
    metrics_diffusion = compute_metrics(x_diffusion_betaM, energy.energy)
    print_metrics("Direct diffusion (β_M)", metrics_diffusion)

    # ── 6. Baseline 2: Classical annealed SMC ────────────────────────────────
    print("\n── Classical annealed SMC (ULA mutation) ──")
    x_high_temp = make_sampler_fn(energy, beta_train, device)(N)
    classical_smc = ClassicalAnnealedSMC(
        energy.energy,
        langevin_step_size=cfg.langevin.step_size,
        n_langevin_steps=cfg.langevin.n_steps_per_beta,
        ess_threshold=cfg.smc.ess_threshold,
        resampling_method=cfg.smc.resampling,
    )
    t0 = time.time()
    cloud_classical, diag_classical = classical_smc.run(
        x_high_temp.clone(), beta_ladder, show_progress=True
    )
    print(f"Classical SMC done in {time.time()-t0:.1f}s. Resamples: {diag_classical.n_resamples}")
    metrics_classical = compute_metrics(cloud_classical.x, energy.energy, cloud_classical.log_weights)
    print_metrics("Classical annealed SMC", metrics_classical)

    # ── 7. Ablation: Diffusion SMC without weight correction ─────────────────
    print("\n── Diffusion SMC WITHOUT weight correction (ablation) ──")
    sampler_no_corr = build_no_correction_smc(
        energy, reverse_sde, beta_train, cfg, device
    )
    init_cloud_1 = ParticleCloud(
        x=x_diffusion_betaM.clone(),
        log_weights=torch.zeros(N, device=device),
    )
    t0 = time.time()
    cloud_no_corr, diag_no_corr = sampler_no_corr.run(
        init_cloud_1, beta_ladder, show_progress=True
    )
    print(f"Done in {time.time()-t0:.1f}s")
    metrics_no_corr = compute_metrics(cloud_no_corr.x, energy.energy)
    print_metrics("Diffusion SMC (no correction)", metrics_no_corr)

    # ── 8. Proposed: Diffusion SMC with weight correction ────────────────────
    print("\n── Diffusion-informed SMC WITH weight correction (proposed) ──")
    sampler_diffusion_smc = build_diffusion_smc(
        energy, reverse_sde, beta_train, cfg, device
    )
    init_cloud_2 = ParticleCloud(
        x=x_diffusion_betaM.clone(),
        log_weights=torch.zeros(N, device=device),
    )
    t0 = time.time()
    cloud_diffusion_smc, diag_diffusion_smc = sampler_diffusion_smc.run(
        init_cloud_2, beta_ladder, show_progress=True
    )
    print(f"Done in {time.time()-t0:.1f}s. Resamples: {diag_diffusion_smc.n_resamples}")
    metrics_diffusion_smc = compute_metrics(
        cloud_diffusion_smc.x, energy.energy, cloud_diffusion_smc.log_weights
    )
    print_metrics("Diffusion-informed SMC (proposed)", metrics_diffusion_smc)

    # ── 9. FKC: Diffusion SMC with path-space FKC correction ────────────
    print("\n── Diffusion SMC WITH FKC path correction ──")
    sampler_fkc = build_fkc_smc(energy, reverse_sde, beta_train, cfg, device)
    init_cloud_3 = ParticleCloud(
        x=x_diffusion_betaM.clone(),
        log_weights=torch.zeros(N, device=device),
    )
    t0 = time.time()
    cloud_fkc, diag_fkc = sampler_fkc.run(
        init_cloud_3, beta_ladder, show_progress=True
    )
    print(f"Done in {time.time()-t0:.1f}s. Resamples: {diag_fkc.n_resamples}")
    metrics_fkc = compute_metrics(cloud_fkc.x, energy.energy, cloud_fkc.log_weights)
    print_metrics("Diffusion SMC (FKC)", metrics_fkc)

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    print("\n── Generating plots ──")

    if energy.dim == 2:
        xlim = (-8, 8) if cfg.energy.type == "double_well" else (-10, 10)
        ylim = xlim

        # Side-by-side comparison
        fig = compare_methods(
            {
                f"Diffusion (β_M={beta_train})": x_diffusion_betaM.cpu(),
                f"Classical SMC": cloud_classical.x.cpu(),
                f"Diff. SMC (no corr.)": cloud_no_corr.x.cpu(),
                f"Diff. SMC (AIS)": cloud_diffusion_smc.x.cpu(),
                f"Diff. SMC (FKC)": cloud_fkc.x.cpu(),
            },
            energy_fn=energy.energy,
            xlim=xlim, ylim=ylim,
            beta=beta_final,
            save_path=str(out_dir / "comparison.png"),
            device=torch.device("cpu"),
        )
        import matplotlib.pyplot as plt; plt.close(fig)

        # SMC diagnostics
        fig = plot_smc_diagnostics(diag_classical, save_path=str(out_dir / "diag_classical_smc.png"))
        plt.close(fig)
        fig = plot_smc_diagnostics(diag_diffusion_smc, save_path=str(out_dir / "diag_diffusion_smc.png"))
        plt.close(fig)
        fig = plot_smc_diagnostics(diag_fkc, save_path=str(out_dir / "diag_fkc_smc.png"))
        plt.close(fig)

    # Energy histograms
    fig = plot_energy_histogram(
        {
            f"Diffusion (β_M={beta_train})": x_diffusion_betaM.cpu(),
            "Classical SMC": cloud_classical.x.cpu(),
            "Diff. SMC (no corr.)": cloud_no_corr.x.cpu(),
            "Diff. SMC (AIS)": cloud_diffusion_smc.x.cpu(),
            "Diff. SMC (FKC)": cloud_fkc.x.cpu(),
        },
        energy_fn=energy.energy,
        beta=beta_final,
        save_path=str(out_dir / "energy_hist.png"),
    )
    import matplotlib.pyplot as plt; plt.close(fig)

    # ── 10. Save summary ──────────────────────────────────────────────────────
    summary = {
        "config": {
            "energy": cfg.energy.type,
            "dim": energy.dim,
            "beta_train": beta_train,
            "beta_final": beta_final,
            "n_particles": N,
            "seed": cfg.seed,
        },
        "metrics": {
            "direct_diffusion": metrics_diffusion,
            "classical_smc": metrics_classical,
            "diffusion_smc_no_correction": metrics_no_corr,
            "diffusion_smc_ais": metrics_diffusion_smc,
            "diffusion_smc_fkc": metrics_fkc,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── Results saved to {out_dir} ──")
    print("\nKey comparison (best energy found):")
    print(f"  Direct diffusion (β_M):          {metrics_diffusion['best_energy']:.4f}")
    print(f"  Classical annealed SMC:           {metrics_classical['best_energy']:.4f}")
    print(f"  Diffusion SMC (no correction):    {metrics_no_corr['best_energy']:.4f}")
    print(f"  Diffusion SMC (AIS):              {metrics_diffusion_smc['best_energy']:.4f}")
    print(f"  Diffusion SMC (FKC):          {metrics_fkc['best_energy']:.4f}")

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Toy 2D amortised annealing experiment")
    p.add_argument("--energy", default="double_well",
                   choices=["double_well", "gmm", "rastrigin", "ackley"],
                   help="Energy function type")
    p.add_argument("--beta-train", type=float, default=1.0,
                   help="Inverse temperature used for training the score model")
    p.add_argument("--beta-final", type=float, default=10.0,
                   help="Target inverse temperature")
    p.add_argument("--n-particles", type=int, default=512,
                   help="Number of SMC particles")
    p.add_argument("--n-steps", type=int, default=15000,
                   help="Score model training steps")
    p.add_argument("--n-beta-steps", type=int, default=30,
                   help="Number of SMC annealing steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--output-dir", default="results/toy_2d")
    p.add_argument("--no-train", action="store_true",
                   help="Skip training if checkpoint exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = ExperimentConfig(
        name=f"{args.energy}_beta{args.beta_train:.0f}to{args.beta_final:.0f}",
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    cfg.energy.type = args.energy
    cfg.energy.dim = 2
    cfg.energy.beta_train = args.beta_train
    cfg.training.n_steps = args.n_steps
    cfg.training.seed = cfg.seed
    cfg.smc.n_particles = args.n_particles
    cfg.smc.beta_final = args.beta_final
    cfg.smc.n_beta_steps = args.n_beta_steps
    cfg.diffusion_smc.n_beta_steps = args.n_beta_steps

    run(cfg, args)
