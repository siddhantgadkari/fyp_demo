from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from .ess import ess
from .particles import ParticleCloud
from .resampling import resample


@dataclass
class SMCDiagnostics:
    """Per-step diagnostics collected during an SMC run."""

    betas: List[float] = field(default_factory=list)
    ess_values: List[float] = field(default_factory=list)
    ess_ratios: List[float] = field(default_factory=list)
    log_weight_means: List[float] = field(default_factory=list)
    log_weight_stds: List[float] = field(default_factory=list)
    best_energies: List[float] = field(default_factory=list)
    mean_energies: List[float] = field(default_factory=list)
    n_resamples: int = 0

    def record(
        self,
        beta: float,
        cloud: ParticleCloud,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        N = cloud.n_particles
        self.betas.append(beta)
        e = cloud.ess()
        self.ess_values.append(e)
        self.ess_ratios.append(e / N)
        lw = cloud.log_weights
        self.log_weight_means.append(lw.mean().item())
        self.log_weight_stds.append(lw.std().item())
        with torch.no_grad():
            energies = energy_fn(cloud.x)
        self.best_energies.append(energies.min().item())
        self.mean_energies.append(energies.mean().item())


MutationKernel = Callable[[torch.Tensor, float], torch.Tensor]
WeightUpdate = Callable[[torch.Tensor, float, float], torch.Tensor]


class SMCSampler:
    """Generic Sequential Monte Carlo sampler.

    The SMC loop:
      for k = 1..K:
        1. Mutate particles: x <- mutation_kernel(x, beta_k)
        2. Update log-weights: lw += weight_update(x, beta_{k-1}, beta_k)
        3. Compute ESS; resample if ESS < threshold * N

    The mutation_kernel and weight_update are injected, making this sampler
    agnostic to whether we use Langevin, diffusion proposals, etc.
    """

    def __init__(
        self,
        mutation_kernel: MutationKernel,
        weight_update: WeightUpdate,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        ess_threshold: float = 0.5,
        resampling_method: str = "systematic",
    ):
        """
        Args:
            mutation_kernel:  (x, beta) -> x_mutated
            weight_update:    (x, beta_prev, beta_curr) -> incremental log-weights [N]
            energy_fn:        x -> E(x) [N], used for diagnostics
            ess_threshold:    Resample when ESS/N < this value
            resampling_method: 'systematic' | 'stratified' | 'multinomial'
        """
        self.mutation_kernel = mutation_kernel
        self.weight_update = weight_update
        self.energy_fn = energy_fn
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method

    def run(
        self,
        initial_cloud: ParticleCloud,
        beta_ladder: torch.Tensor,
        show_progress: bool = True,
    ) -> tuple[ParticleCloud, SMCDiagnostics]:
        """Run the SMC loop over the full beta ladder.

        Args:
            initial_cloud:  Particles approximating pi_{beta_ladder[0]}.
            beta_ladder:    [K+1] tensor; beta_ladder[0] is the initial temperature.
            show_progress:  Print progress.

        Returns:
            (final_cloud, diagnostics)
        """
        from tqdm import tqdm

        cloud = initial_cloud.clone()
        diagnostics = SMCDiagnostics()
        N = cloud.n_particles
        ladder = beta_ladder.tolist()

        diagnostics.record(ladder[0], cloud, self.energy_fn)

        iterator = range(1, len(ladder))
        if show_progress:
            iterator = tqdm(iterator, desc="SMC", dynamic_ncols=True)

        for k in iterator:
            beta_prev = ladder[k - 1]
            beta_curr = ladder[k]

            # --- Mutation ---
            cloud.x = self.mutation_kernel(cloud.x, beta_curr)

            # --- Weight update ---
            delta_lw = self.weight_update(cloud.x, beta_prev, beta_curr)
            delta_lw = torch.nan_to_num(delta_lw, nan=float("-inf"))
            cloud.log_weights = cloud.log_weights + delta_lw

            # Clamp to avoid extreme weights destabilising the run
            finite_mask = torch.isfinite(cloud.log_weights)
            if finite_mask.any():
                cloud.log_weights = cloud.log_weights - cloud.log_weights[finite_mask].max()
            cloud.log_weights = torch.nan_to_num(cloud.log_weights, nan=float("-inf"))

            # --- Resample if needed ---
            current_ess_ratio = ess(cloud.log_weights) / N
            if current_ess_ratio < self.ess_threshold:
                indices = resample(cloud.log_weights, method=self.resampling_method)
                cloud.x = cloud.x[indices]
                cloud.log_weights = torch.zeros(N, device=cloud.log_weights.device)
                diagnostics.n_resamples += 1

            diagnostics.record(beta_curr, cloud, self.energy_fn)

            if show_progress:
                iterator.set_postfix(
                    beta=f"{beta_curr:.2f}",
                    ess=f"{current_ess_ratio:.2f}",
                    E_min=f"{diagnostics.best_energies[-1]:.3f}",
                )

        return cloud, diagnostics
