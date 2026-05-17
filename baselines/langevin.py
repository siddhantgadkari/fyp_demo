"""Unadjusted Langevin Algorithm (ULA) and Metropolis-Adjusted (MALA)."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def _grad_energy(
    x: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        e = energy_fn(x)
        return torch.autograd.grad(e.sum(), x)[0].detach()


class ULA:
    """Unadjusted Langevin Algorithm targeting pi_beta ∝ exp(-beta * E(x)).

    Update rule:
        x_{k+1} = x_k - beta * h * grad E(x_k) + sqrt(2h) * eps

    No Metropolis correction — biased but parallelisable and cheap.
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 1e-3,
    ):
        self.energy_fn = energy_fn
        self.step_size = step_size

    @torch.no_grad()
    def step(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        h = self.step_size
        grad_e = _grad_energy(x, self.energy_fn)
        noise = torch.randn_like(x)
        return x - beta * h * grad_e + (2 * h) ** 0.5 * noise

    @torch.no_grad()
    def run(
        self,
        x: torch.Tensor,
        beta: float,
        n_steps: int,
    ) -> torch.Tensor:
        for _ in range(n_steps):
            x = self.step(x, beta)
        return x


class MALA:
    """Metropolis-Adjusted Langevin Algorithm.

    Proposes ULA step, then applies Metropolis-Hastings acceptance step.
    Unbiased targeting of pi_beta at the cost of a rejection step.
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 1e-3,
    ):
        self.energy_fn = energy_fn
        self.step_size = step_size

    def step(self, x: torch.Tensor, beta: float) -> Tuple[torch.Tensor, float]:
        """One MALA step.

        Returns:
            x_new: [N, d]
            acceptance_rate: float
        """
        h = self.step_size

        with torch.no_grad():
            grad_curr = _grad_energy(x, self.energy_fn)
            mean_fwd = x - beta * h * grad_curr
            x_prop = mean_fwd + (2 * h) ** 0.5 * torch.randn_like(x)

            # Log proposal densities
            grad_prop = _grad_energy(x_prop, self.energy_fn)
            mean_bwd = x_prop - beta * h * grad_prop

            log_q_fwd = -((x_prop - mean_fwd) ** 2).sum(-1) / (4 * h)
            log_q_bwd = -((x - mean_bwd) ** 2).sum(-1) / (4 * h)

            e_curr = self.energy_fn(x)
            e_prop = self.energy_fn(x_prop)
            log_accept = -beta * (e_prop - e_curr) + log_q_bwd - log_q_fwd

            accept = torch.log(torch.rand_like(log_accept)) < log_accept
            x_new = torch.where(accept.unsqueeze(-1), x_prop, x)
            return x_new, accept.float().mean().item()

    def run(
        self,
        x: torch.Tensor,
        beta: float,
        n_steps: int,
    ) -> Tuple[torch.Tensor, float]:
        total_accept = 0.0
        for _ in range(n_steps):
            x, acc = self.step(x, beta)
            total_accept += acc
        return x, total_accept / n_steps
