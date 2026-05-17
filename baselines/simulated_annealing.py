"""Simulated annealing baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch
from tqdm import tqdm


@dataclass
class SAResult:
    x_best: torch.Tensor
    e_best: float
    energy_history: List[float]
    temperature_history: List[float]


class SimulatedAnnealing:
    """Simulated annealing with Gaussian random-walk proposals.

    Runs one chain per particle, independently.
    Beta schedule: linear or geometric from beta_start to beta_end.
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        proposal_std: float = 0.1,
    ):
        self.energy_fn = energy_fn
        self.proposal_std = proposal_std

    def run(
        self,
        x_init: torch.Tensor,
        beta_start: float,
        beta_end: float,
        n_steps: int,
        ladder: str = "geometric",
        show_progress: bool = True,
    ) -> SAResult:
        """Run simulated annealing.

        Args:
            x_init:      [N, d] initial particles
            beta_start:  Starting inverse temperature (low = hot)
            beta_end:    Final inverse temperature (high = cold)
            n_steps:     Total number of MCMC steps
            ladder:      'linear' or 'geometric' temperature schedule
            show_progress: tqdm bar

        Returns:
            SAResult with best particles and diagnostics
        """
        x = x_init.clone()
        device = x.device
        N, d = x.shape

        if ladder == "geometric":
            betas = torch.exp(torch.linspace(
                torch.log(torch.tensor(beta_start)),
                torch.log(torch.tensor(beta_end)),
                n_steps,
                device=device,
            ))
        else:
            betas = torch.linspace(beta_start, beta_end, n_steps, device=device)

        with torch.no_grad():
            e_curr = self.energy_fn(x)

        x_best = x.clone()
        e_best = e_curr.clone()

        energy_history: List[float] = []
        temp_history: List[float] = []

        iterator = range(n_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulated annealing", dynamic_ncols=True)

        for step in iterator:
            beta = betas[step].item()
            proposal = x + self.proposal_std * torch.randn_like(x)

            with torch.no_grad():
                e_prop = self.energy_fn(proposal)

            log_accept = -beta * (e_prop - e_curr)
            accept = torch.log(torch.rand_like(log_accept)) < log_accept

            x = torch.where(accept.unsqueeze(-1), proposal, x)
            e_curr = torch.where(accept, e_prop, e_curr)

            improved = e_curr < e_best
            x_best = torch.where(improved.unsqueeze(-1), x, x_best)
            e_best = torch.where(improved, e_curr, e_best)

            if step % max(1, n_steps // 200) == 0:
                energy_history.append(e_curr.min().item())
                temp_history.append(beta)

        return SAResult(
            x_best=x_best,
            e_best=e_best.min().item(),
            energy_history=energy_history,
            temperature_history=temp_history,
        )
