"""Tests for diffusion schedules and beta ladders."""
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from amortised_annealing.schedules import VPSchedule, VESchedule, linear_ladder, geometric_ladder


def test_vp_alpha_boundary():
    sched = VPSchedule()
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    assert sched.alpha(t0).item() == pytest.approx(1.0, abs=1e-5), "alpha(0) should be 1"
    assert sched.alpha(t1).item() < 0.01, "alpha(1) should be near 0 for VP"


def test_vp_sigma_boundary():
    sched = VPSchedule()
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    assert sched.sigma(t0).item() == pytest.approx(0.0, abs=1e-4), "sigma(0) should be near 0"
    assert sched.sigma(t1).item() > 0.99, "sigma(1) should be near 1 for VP"


def test_vp_variance_preserving():
    """alpha^2 + sigma^2 should approximately equal 1 (VP property)."""
    sched = VPSchedule()
    t = torch.linspace(0.01, 0.99, 50)
    alpha2_plus_sigma2 = sched.alpha(t)**2 + sched.sigma(t)**2
    assert torch.allclose(alpha2_plus_sigma2, torch.ones(50), atol=1e-4)


def test_ve_alpha_is_one():
    sched = VESchedule()
    t = torch.linspace(0, 1, 10)
    assert torch.allclose(sched.alpha(t), torch.ones(10), atol=1e-6)


def test_ve_sigma_monotone():
    sched = VESchedule(sigma_min=0.01, sigma_max=50.0)
    t = torch.linspace(0, 1, 20)
    sigmas = sched.sigma(t)
    diffs = sigmas[1:] - sigmas[:-1]
    assert (diffs > 0).all(), "VE sigma should be strictly increasing"


def test_marginal_sample_shape():
    sched = VPSchedule()
    x0 = torch.randn(8, 4)
    t = torch.rand(8)
    x_t, eps = sched.marginal_sample(x0, t)
    assert x_t.shape == (8, 4)
    assert eps.shape == (8, 4)


def test_conditional_score_shape():
    sched = VPSchedule()
    x0 = torch.randn(8, 4)
    t = torch.rand(8)
    x_t, _ = sched.marginal_sample(x0, t)
    score = sched.conditional_score(x_t, x0, t)
    assert score.shape == (8, 4)


def test_conditional_score_formula():
    """Conditional score should be -(x_t - alpha_t * x0) / sigma_t^2."""
    sched = VPSchedule()
    x0 = torch.randn(5, 3)
    t = torch.tensor([0.5] * 5)
    alpha_t = sched.alpha(t).view(-1, 1)
    sigma_t = sched.sigma(t).view(-1, 1)
    eps = torch.randn(5, 3)
    x_t = alpha_t * x0 + sigma_t * eps
    score = sched.conditional_score(x_t, x0, t)
    expected = -eps / sigma_t  # equivalent form
    assert torch.allclose(score, expected, atol=1e-4)


def test_eps_to_score():
    sched = VPSchedule()
    eps = torch.randn(5, 3)
    t = torch.tensor([0.3] * 5)
    score = sched.eps_to_score(eps, t)
    sigma_t = sched.sigma(t).view(-1, 1)
    assert torch.allclose(score, -eps / sigma_t, atol=1e-6)


def test_linear_ladder_monotone():
    ladder = linear_ladder(1.0, 10.0, 20)
    assert (ladder[1:] > ladder[:-1]).all()
    assert ladder[0].item() == pytest.approx(1.0)
    assert ladder[-1].item() == pytest.approx(10.0)


def test_geometric_ladder_monotone():
    ladder = geometric_ladder(1.0, 100.0, 30)
    assert (ladder[1:] > ladder[:-1]).all()
    assert ladder[0].item() == pytest.approx(1.0, rel=1e-4)
    assert ladder[-1].item() == pytest.approx(100.0, rel=1e-4)


def test_geometric_ladder_ratios_constant():
    """Consecutive ratios should be approximately constant (geometric progression)."""
    ladder = geometric_ladder(1.0, 100.0, 10)
    ratios = ladder[1:] / ladder[:-1]
    assert ratios.std().item() < 1e-5, "Geometric ladder should have constant ratio"
