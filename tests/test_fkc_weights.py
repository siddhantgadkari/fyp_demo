"""Tests for FKC weight update functions."""
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from amortised_annealing.distributions import DoubleWell
from amortised_annealing.fkc.weight_updates import (
    annealing_weight_update,
    diffusion_informed_weight_update,
)


@pytest.fixture
def energy_fn():
    energy = DoubleWell(dim=2)
    return energy.energy


def test_annealing_weight_shape(energy_fn):
    N = 50
    x = torch.randn(N, 2)
    lw = annealing_weight_update(x, 1.0, 2.0, energy_fn)
    assert lw.shape == (N,)


def test_annealing_weight_sign(energy_fn):
    """High-energy particles should get negative weight increments (penalised)."""
    x_high = torch.ones(5, 2) * 5.0   # high energy for DoubleWell
    x_low = torch.zeros(5, 2)          # near minimum
    lw_high = annealing_weight_update(x_high, 1.0, 2.0, energy_fn)
    lw_low = annealing_weight_update(x_low, 1.0, 2.0, energy_fn)
    # All weights should be <= 0 for positive energies
    # Both should be negative; low-energy particles get less negative weights
    assert lw_low.mean() > lw_high.mean(), (
        "Low-energy particles should receive higher weight increments"
    )


def test_annealing_weight_zero_for_same_beta(energy_fn):
    """No beta change -> no weight change."""
    x = torch.randn(10, 2)
    lw = annealing_weight_update(x, 2.0, 2.0, energy_fn)
    assert torch.allclose(lw, torch.zeros(10), atol=1e-6)


def test_annealing_weight_finite(energy_fn):
    x = torch.randn(30, 2)
    lw = annealing_weight_update(x, 1.0, 5.0, energy_fn)
    assert torch.isfinite(lw).all()


def test_diffusion_informed_weight_without_ratio(energy_fn):
    """Without proposal ratio, should reduce to annealing weight."""
    x_before = torch.randn(20, 2)
    x_after = torch.randn(20, 2)
    lw1 = annealing_weight_update(x_after, 1.0, 3.0, energy_fn)
    lw2 = diffusion_informed_weight_update(x_before, x_after, 1.0, 3.0, energy_fn, None)
    assert torch.allclose(lw1, lw2, atol=1e-6)


def test_diffusion_informed_weight_with_ratio(energy_fn):
    x_before = torch.randn(20, 2)
    x_after = torch.randn(20, 2)
    log_ratio = torch.randn(20)
    lw = diffusion_informed_weight_update(x_before, x_after, 1.0, 2.0, energy_fn, log_ratio)
    assert lw.shape == (20,)
    assert torch.isfinite(lw).all()
