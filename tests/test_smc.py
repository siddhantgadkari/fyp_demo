"""Integration tests for the SMC sampler."""
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from amortised_annealing.distributions import DoubleWell
from amortised_annealing.schedules import make_ladder
from amortised_annealing.smc import ParticleCloud, SMCSampler
from amortised_annealing.baselines import ClassicalAnnealedSMC
from amortised_annealing.fkc.weight_updates import annealing_weight_update


def make_simple_smc(energy_fn, ula_steps=3, step_size=1e-3):
    from amortised_annealing.baselines.langevin import ULA

    ula = ULA(energy_fn, step_size=step_size)

    def mutation_kernel(x, beta):
        return ula.run(x, beta, ula_steps)

    def weight_update(x, beta_prev, beta_curr):
        return annealing_weight_update(x, beta_prev, beta_curr, energy_fn)

    return SMCSampler(
        mutation_kernel=mutation_kernel,
        weight_update=weight_update,
        energy_fn=energy_fn,
        ess_threshold=0.5,
    )


def test_smc_runs_and_returns_cloud():
    energy = DoubleWell(dim=2)
    N = 64
    x_init = torch.randn(N, 2)
    log_w_init = torch.zeros(N)
    cloud = ParticleCloud(x=x_init, log_weights=log_w_init)

    ladder = make_ladder(0.5, 2.0, n_steps=5)
    sampler = make_simple_smc(energy.energy)
    final_cloud, diag = sampler.run(cloud, ladder, show_progress=False)

    assert final_cloud.x.shape == (N, 2)
    assert final_cloud.log_weights.shape == (N,)
    assert torch.isfinite(final_cloud.log_weights).all()


def test_smc_diagnostics_length():
    energy = DoubleWell(dim=2)
    N = 32
    n_steps = 4
    cloud = ParticleCloud(
        x=torch.randn(N, 2),
        log_weights=torch.zeros(N),
    )
    ladder = make_ladder(1.0, 3.0, n_steps=n_steps)
    sampler = make_simple_smc(energy.energy)
    _, diag = sampler.run(cloud, ladder, show_progress=False)

    # Diagnostics should have one record per ladder entry
    assert len(diag.betas) == n_steps + 1
    assert len(diag.ess_ratios) == n_steps + 1


def test_classical_annealed_smc_runs():
    energy = DoubleWell(dim=2)
    N = 64
    x_init = torch.randn(N, 2)
    ladder = make_ladder(1.0, 5.0, n_steps=5)

    smc = ClassicalAnnealedSMC(
        energy.energy,
        langevin_step_size=5e-3,
        n_langevin_steps=3,
    )
    cloud, diag = smc.run(x_init, ladder, show_progress=False)
    assert cloud.x.shape == (N, 2)
    assert torch.isfinite(cloud.log_weights).all()


def test_log_weights_finite_throughout():
    energy = DoubleWell(dim=2)
    N = 32
    cloud = ParticleCloud(
        x=torch.randn(N, 2),
        log_weights=torch.zeros(N),
    )
    ladder = make_ladder(1.0, 4.0, n_steps=6)
    sampler = make_simple_smc(energy.energy)
    final_cloud, diag = sampler.run(cloud, ladder, show_progress=False)

    for lw_mean in diag.log_weight_means:
        assert torch.isfinite(torch.tensor(lw_mean))


def test_smc_energy_decreases_with_beta():
    """Higher beta should generally give lower mean energy."""
    energy = DoubleWell(dim=2, a=1.0, b=4.0)
    N = 128
    cloud = ParticleCloud(
        x=torch.randn(N, 2),
        log_weights=torch.zeros(N),
    )
    ladder = make_ladder(1.0, 8.0, n_steps=8)
    sampler = make_simple_smc(energy.energy, ula_steps=5, step_size=1e-2)
    _, diag = sampler.run(cloud, ladder, show_progress=False)

    # Mean energy at high beta should be lower than at low beta
    first_mean = diag.mean_energies[1]
    last_mean = diag.mean_energies[-1]
    assert last_mean < first_mean, (
        f"Expected mean energy to decrease: first={first_mean:.3f}, last={last_mean:.3f}"
    )
