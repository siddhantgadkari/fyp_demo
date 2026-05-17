"""Diffusion-informed annealed SMC proposal.

This module builds the mutation kernel and weight update that together implement
the "single trained model + inference-time annealing" algorithm.

The proposal:
  - Use the reverse diffusion dynamics of the model trained at beta_M as
    a mutation kernel: run a few Euler-Maruyama reverse steps.
  - Optionally scale the score by (beta_curr / beta_M) as a heuristic to
    bias the proposal toward the colder target.  This is NOT exact — it is
    a proposal heuristic whose error is corrected by the SMC weights.

The weight update:
  - Use the AIS-style incremental weight log(pi_{beta_k}/pi_{beta_{k-1}}) at
    the mutated particles.  This is exact for identity kernels and remains a
    valid IS weight for any ergodic mutation kernel.

For continuous-time FKC weights (from the FKC paper), we would need to
simulate the log-weight SDE alongside the particle SDE.  That is implemented
as a future extension hook in `fkc_continuous_weight_update`.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from ..diffusion.reverse_sde import ReverseSDE
from .weight_updates import annealing_weight_update


class AnnealedDiffusionProposal:
    """Mutation kernel that uses diffusion reverse dynamics for each SMC step.

    At each beta transition, we run `n_diffusion_steps` of Euler-Maruyama
    reverse SDE.  The score can optionally be scaled by (beta / beta_train)
    as a heuristic push toward the colder target — explicitly labelled as
    a proposal heuristic to make clear it does not produce exact samples.

    The combined (mutation_kernel, weight_update) pair is ready for use with
    SMCSampler.
    """

    def __init__(
        self,
        reverse_sde: ReverseSDE,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        beta_train: float,
        n_diffusion_steps: int = 10,
        t_start: float = 0.3,
        t_end: float = 1e-3,
        use_score_scaling: bool = True,
        langevin_steps: int = 0,
        langevin_step_size: float = 1e-3,
    ):
        """
        Args:
            reverse_sde:        ReverseSDE (model + schedule).
            energy_fn:          x -> E(x) for weight updates and optional Langevin.
            beta_train:         beta_M at which the score model was trained.
            n_diffusion_steps:  Euler-Maruyama steps per SMC mutation.
            t_start:            Start time for partial reverse diffusion.
                                Use < 1 to inject partial noise (noise-reinject trick).
            t_end:              End time for reverse diffusion.
            use_score_scaling:  If True, multiply score by (beta_curr / beta_train).
                                Mark as heuristic; corrected by SMC weights.
            langevin_steps:     Additional ULA steps after diffusion mutation.
            langevin_step_size: Step size for ULA.
        """
        self.reverse_sde = reverse_sde
        self.energy_fn = energy_fn
        self.beta_train = beta_train
        self.n_diffusion_steps = n_diffusion_steps
        self.t_start = t_start
        self.t_end = t_end
        self.use_score_scaling = use_score_scaling
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size

    def mutation_kernel(self, x: torch.Tensor, beta_curr: float) -> torch.Tensor:
        """Mutate particles using diffusion reverse dynamics.

        Applies noise injection + reverse diffusion + optional Langevin polish.
        """
        scale = (beta_curr / self.beta_train) if self.use_score_scaling else 1.0
        device = x.device
        schedule = self.reverse_sde.schedule
        N = x.shape[0]

        # Noise-reinject trick: add forward noise at t_start, then denoise
        t_inj = torch.full((N,), self.t_start, device=device)
        x_noisy, _ = schedule.marginal_sample(x, t_inj)

        times = torch.linspace(self.t_start, self.t_end, self.n_diffusion_steps + 1, device=device)
        dt = (self.t_start - self.t_end) / self.n_diffusion_steps

        with torch.no_grad():
            x_cur = x_noisy
            for i in range(self.n_diffusion_steps):
                t = times[i].expand(N)
                x_cur = self.reverse_sde.step(x_cur, t, dt, temperature_scale=scale)

        # Optional ULA polish targeting pi_{beta_curr}
        if self.langevin_steps > 0:
            x_cur = self._ula_steps(x_cur, beta_curr)

        return x_cur

    def weight_update(
        self, x: torch.Tensor, beta_prev: float, beta_curr: float
    ) -> torch.Tensor:
        """AIS-style incremental log-weight: -(beta_curr - beta_prev) * E(x)."""
        return annealing_weight_update(x, beta_prev, beta_curr, self.energy_fn)

    @torch.no_grad()
    def _ula_steps(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        h = self.langevin_step_size
        for _ in range(self.langevin_steps):
            grad_e = self._grad_energy(x)
            x = x - beta * h * grad_e + (2 * h) ** 0.5 * torch.randn_like(x)
        return x

    def _grad_energy(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            e = self.energy_fn(x)
            return torch.autograd.grad(e.sum(), x)[0].detach()
