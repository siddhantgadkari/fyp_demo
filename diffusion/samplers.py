from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm

from .reverse_sde import ReverseSDE


@torch.no_grad()
def euler_maruyama_sample(
    reverse_sde: ReverseSDE,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    t_start: float = 1.0,
    t_end: float = 1e-3,
    temperature_scale: float = 1.0,
    x_init: Optional[torch.Tensor] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Generate samples via Euler-Maruyama integration of the reverse SDE.

    Starts from x_T ~ N(0, I) (or x_init if provided) and integrates
    the reverse SDE from t_start down to t_end.

    Args:
        reverse_sde:       ReverseSDE wrapping model + schedule.
        n_samples:         Number of particles to generate.
        n_steps:           Number of discretisation steps.
        device:            Torch device.
        t_start:           Starting time (≈1 for VP, pure noise).
        t_end:             Ending time (>0 to avoid score singularity at t=0).
        temperature_scale: Score multiplier (use 1.0 for standard sampling).
        x_init:            Optional [n_samples, d] starting particles.
        show_progress:     Show tqdm bar.

    Returns:
        x: [n_samples, d] approximate samples from pi_beta_M.
    """
    dim = reverse_sde.model.dim

    if x_init is not None:
        x = x_init.clone().to(device)
    else:
        # Start from approximate marginal at t=T (pure noise for VP)
        x = torch.randn(n_samples, dim, device=device)

    times = torch.linspace(t_start, t_end, n_steps + 1, device=device)
    dt = (t_start - t_end) / n_steps

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Reverse SDE", dynamic_ncols=True)

    for i in iterator:
        t = times[i].expand(n_samples)
        x = reverse_sde.step(x, t, dt, temperature_scale=temperature_scale)

    return x


@torch.no_grad()
def DDIMSample(
    reverse_sde: ReverseSDE,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    t_start: float = 1.0,
    t_end: float = 1e-3,
    temperature_scale: float = 1.0,
    x_init: Optional[torch.Tensor] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Deterministic DDIM sampling (probability-flow ODE).

    Same interface as euler_maruyama_sample but uses deterministic steps.
    Suitable only for VP schedules.
    """
    dim = reverse_sde.model.dim

    if x_init is not None:
        x = x_init.clone().to(device)
    else:
        x = torch.randn(n_samples, dim, device=device)

    times = torch.linspace(t_start, t_end, n_steps + 1, device=device)

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="DDIM", dynamic_ncols=True)

    for i in iterator:
        t_cur = times[i].expand(n_samples)
        t_next = times[i + 1].expand(n_samples)
        x = reverse_sde.ddim_step(x, t_cur, t_next, temperature_scale=temperature_scale)

    return x
