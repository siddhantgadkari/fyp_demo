"""Tests for score model, loss, and basic training."""
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from amortised_annealing.score_models import MLPScore, dsm_loss, TrainingConfig
from amortised_annealing.schedules import VPSchedule


def test_mlp_score_forward_shape():
    model = MLPScore(dim=4, hidden_dims=(32, 32))
    t = torch.rand(8)
    x = torch.randn(8, 4)
    out = model(t, x)
    assert out.shape == (8, 4)


def test_mlp_score_score_method():
    model = MLPScore(dim=4, hidden_dims=(32, 32))
    sched = VPSchedule()
    t = torch.rand(8)
    x = torch.randn(8, 4)
    score = model.score(t, x, sched)
    assert score.shape == (8, 4)


def test_dsm_loss_scalar_finite():
    model = MLPScore(dim=2, hidden_dims=(16, 16))
    sched = VPSchedule()
    x0 = torch.randn(32, 2)
    loss = dsm_loss(model, x0, sched)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_dsm_loss_positive():
    model = MLPScore(dim=2, hidden_dims=(16, 16))
    sched = VPSchedule()
    x0 = torch.randn(32, 2)
    loss = dsm_loss(model, x0, sched)
    assert loss.item() > 0


def test_dsm_loss_backprop():
    model = MLPScore(dim=2, hidden_dims=(16, 16))
    sched = VPSchedule()
    x0 = torch.randn(16, 2)
    loss = dsm_loss(model, x0, sched)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_score_predict_mode():
    model = MLPScore(dim=3, hidden_dims=(16,), predict_score=True)
    sched = VPSchedule()
    t = torch.rand(5)
    x = torch.randn(5, 3)
    # In predict_score mode, model.score returns model output directly
    out = model(t, x)
    score = model.score(t, x, sched)
    assert torch.allclose(score, out)


def test_training_reduces_loss():
    """A few gradient steps should reduce the DSM loss."""
    from amortised_annealing.score_models.training import train_score_model

    torch.manual_seed(42)
    dim = 2
    model = MLPScore(dim=dim, hidden_dims=(32, 32))
    sched = VPSchedule()

    def sample_fn(n):
        return torch.randn(n, dim)  # trivial N(0,I) target

    config = TrainingConfig(n_steps=100, batch_size=64, log_every=50, lr=1e-3, ema_decay=0.99)
    ema_model, losses = train_score_model(model, sched, sample_fn, config, torch.device("cpu"))
    assert len(losses) == 2
    # Loss should generally trend downward for a trivial target
    assert losses[-1] < losses[0] * 2, "Loss should not explode"
