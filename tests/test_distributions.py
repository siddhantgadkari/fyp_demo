"""Tests for energy functions and distributions."""
import math
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from amortised_annealing.distributions import (
    GaussianMixture, DoubleWell, ManyWell, Rastrigin, Ackley, Rosenbrock
)
from amortised_annealing.distributions.base import BoltzmannDistribution


def _finite_diff_grad(energy_fn, x, eps=1e-4):
    """Numerical gradient via central differences."""
    N, d = x.shape
    grad = torch.zeros_like(x)
    for i in range(d):
        x_fwd = x.clone(); x_fwd[:, i] += eps
        x_bwd = x.clone(); x_bwd[:, i] -= eps
        grad[:, i] = (energy_fn(x_fwd) - energy_fn(x_bwd)) / (2 * eps)
    return grad


@pytest.mark.parametrize("EnergyClass,kwargs", [
    (GaussianMixture, {"means": torch.randn(4, 2), "stds": torch.ones(4) * 0.5}),
    (DoubleWell, {"dim": 2}),
    (ManyWell, {"dim": 4}),
    (Rastrigin, {"dim": 3}),
    (Ackley, {"dim": 3}),
    (Rosenbrock, {"dim": 3}),
])
def test_energy_output_shape(EnergyClass, kwargs):
    energy = EnergyClass(**kwargs)
    N = 10
    x = torch.randn(N, energy.dim)
    e = energy.energy(x)
    assert e.shape == (N,), f"{EnergyClass.__name__}: expected shape ({N},), got {e.shape}"


@pytest.mark.parametrize("EnergyClass,kwargs", [
    (DoubleWell, {"dim": 2}),
    (ManyWell, {"dim": 4}),
    (Rastrigin, {"dim": 2}),
    (Ackley, {"dim": 2}),
    (Rosenbrock, {"dim": 2}),
])
def test_energy_gradients_vs_finitediff(EnergyClass, kwargs):
    torch.manual_seed(7)
    energy = EnergyClass(**kwargs)
    x = torch.randn(5, energy.dim) * 0.3  # small values reduce cosine truncation error
    grad_auto = energy.grad_energy(x)
    grad_fd = _finite_diff_grad(energy.energy, x, eps=1e-5)
    # Use relative tolerance: finite-diff has O(h^2 * f''') truncation error,
    # which can be large for oscillatory functions (Rastrigin/Ackley).
    # The autograd gradients are exact; we verify they match to 1% relative error.
    max_abs = (grad_auto.abs() + grad_fd.abs()).clamp(min=1.0).max()
    max_diff = (grad_auto - grad_fd).abs().max()
    assert max_diff / max_abs < 0.01, (
        f"{EnergyClass.__name__}: autograd and finite-diff gradients disagree\n"
        f"max relative diff = {(max_diff / max_abs).item():.4f}"
    )


def test_rastrigin_global_minimum():
    energy = Rastrigin(dim=3)
    x = torch.zeros(1, 3)
    assert energy.energy(x).item() < 1e-6, "Rastrigin global min should be near 0"


def test_ackley_global_minimum():
    energy = Ackley(dim=3)
    x = torch.zeros(1, 3)
    assert energy.energy(x).item() < 1e-5, "Ackley global min should be near 0"


def test_rosenbrock_global_minimum():
    energy = Rosenbrock(dim=3)
    x = torch.ones(1, 3)
    assert energy.energy(x).item() < 1e-6, "Rosenbrock global min should be near 0"


def test_double_well_two_minima():
    energy = DoubleWell(dim=1, a=1.0, b=4.0)
    x_pos = torch.tensor([[math.sqrt(2.0)]])
    x_neg = torch.tensor([[-math.sqrt(2.0)]])
    x_mid = torch.tensor([[0.0]])
    e_pos = energy.energy(x_pos).item()
    e_neg = energy.energy(x_neg).item()
    e_mid = energy.energy(x_mid).item()
    assert e_pos < e_mid, "Positive minimum should be lower than saddle"
    assert e_neg < e_mid, "Negative minimum should be lower than saddle"
    assert abs(e_pos - e_neg) < 1e-5, "Symmetric double well: both minima equal"


def test_gmm_samples_shape():
    gmm = GaussianMixture.random_2d(n_components=5)
    samples = gmm.sample(100)
    assert samples.shape == (100, 2)


def test_boltzmann_score():
    energy = DoubleWell(dim=2)
    boltz = BoltzmannDistribution(energy, beta=2.0)
    x = torch.randn(10, 2)
    score = boltz.score(x)
    assert score.shape == (10, 2)
    # Score = -beta * grad_energy should match
    expected = -2.0 * energy.grad_energy(x)
    assert torch.allclose(score, expected, atol=1e-5)
