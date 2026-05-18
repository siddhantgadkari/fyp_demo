from __future__ import annotations

import torch


def dsm_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    schedule,
    t_eps: float = 1e-4,
) -> torch.Tensor:
    """Denoising Score Matching (DSM) loss.

    For an eps-prediction model:
        L = E_{t, x_0, eps}[ || model(t, x_t) - eps ||^2 ]

    For a score-prediction model, the equivalent is:
        L = E_{t, x_0, eps}[ sigma(t)^2 * || model(t, x_t) - cond_score ||^2 ]

    We use the eps-prediction form (simpler, numerically stable).

    Args:
        model:    MLPScore with predict_score=False (eps-prediction).
        x0:       [N, d] clean samples from the training distribution.
        schedule: DiffusionSchedule providing alpha/sigma/marginal_sample.
        t_eps:    Minimum time value to avoid t=0 singularity.

    Returns:
        Scalar loss.
    """
    N = x0.shape[0]
    t = torch.rand(N, device=x0.device, dtype=x0.dtype) * (1.0 - t_eps) + t_eps
    x_t, eps = schedule.marginal_sample(x0, t)

    eps_pred = model(t, x_t)
    return ((eps_pred - eps) ** 2).sum(-1).mean()


