"""Tests for SMC resampling and ESS."""
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from amortised_annealing.smc import (
    multinomial_resample, systematic_resample, stratified_resample,
    ess, log_ess_ratio,
)


def test_systematic_resample_shape():
    N = 100
    log_w = torch.randn(N)
    idx = systematic_resample(log_w)
    assert idx.shape == (N,)
    assert (idx >= 0).all()
    assert (idx < N).all()


def test_multinomial_resample_shape():
    N = 100
    log_w = torch.randn(N)
    idx = multinomial_resample(log_w)
    assert idx.shape == (N,)


def test_stratified_resample_shape():
    N = 100
    log_w = torch.randn(N)
    idx = stratified_resample(log_w)
    assert idx.shape == (N,)


def test_resample_uniform_weights():
    """With uniform weights, all particles should have roughly equal selection probability."""
    N = 1000
    log_w = torch.zeros(N)
    idx = systematic_resample(log_w)
    # Each index should appear at most 2 times with uniform weights (systematic)
    counts = torch.bincount(idx, minlength=N)
    assert counts.max().item() <= 2


def test_resample_degenerate_weights():
    """With all weight on particle 0, resampling should select only particle 0."""
    N = 50
    log_w = torch.full((N,), -1e10)
    log_w[0] = 0.0
    idx = systematic_resample(log_w)
    assert (idx == 0).all(), "All indices should be 0 with degenerate weights"


def test_ess_uniform():
    """Uniform weights -> ESS = N."""
    N = 100
    log_w = torch.zeros(N)
    assert ess(log_w) == pytest.approx(N, rel=1e-3)


def test_ess_degenerate():
    """All weight on one particle -> ESS = 1."""
    N = 100
    log_w = torch.full((N,), -1e10)
    log_w[0] = 0.0
    assert ess(log_w) == pytest.approx(1.0, rel=1e-2)


def test_ess_ratio_bounds():
    N = 50
    log_w = torch.randn(N)
    ratio = ess(log_w) / N
    assert 0.0 < ratio <= 1.0


def test_log_ess_ratio_uniform():
    import math
    N = 100
    log_w = torch.zeros(N)
    assert log_ess_ratio(log_w) == pytest.approx(0.0, abs=1e-4)


def test_resampling_preserves_particle_count():
    """After resampling, number of particles should be unchanged."""
    N = 200
    d = 3
    x = torch.randn(N, d)
    log_w = torch.randn(N)
    idx = systematic_resample(log_w)
    x_resampled = x[idx]
    assert x_resampled.shape == (N, d)
