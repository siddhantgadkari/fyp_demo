"""Thin wrappers around schedule methods for clarity at call sites."""
from __future__ import annotations

import torch


def forward_marginal(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t ~ q(x_t | x_0) = N(alpha_t * x0, sigma_t^2 * I).

    Args:
        x0:       [N, d] clean samples
        t:        [N] or scalar time in [0, 1]
        schedule: DiffusionSchedule

    Returns:
        x_t: [N, d]   noised sample
        eps: [N, d]   noise realisation used (for loss computation)
    """
    return schedule.marginal_sample(x0, t)


def conditional_score(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule,
) -> torch.Tensor:
    """Analytic score of q(x_t | x_0): -(x_t - alpha_t * x0) / sigma_t^2."""
    return schedule.conditional_score(x_t, x0, t)
