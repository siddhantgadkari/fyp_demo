"""FKC weighted reverse SDE"""
from __future__ import annotations
from typing import Callable, Optional

import torch

from amortised_annealing.diffusion.reverse_sde import ReverseSDE
from ..smc.particles import ParticleCloud
from ..smc.smc_sampler import SMCDiagnostics

def ess(logw):
    w = torch.exp(logw - logw.max())
    return w.sum() ** 2 / (w**2).sum()

class FKCAnnealedSampler:
    """FKC / particle-filter sampler: one weighted reverse SDE with resampling inside diffusion time."""

    def __init__(
        self,
        reverse_sde: ReverseSDE,
        beta0: float,
        beta1: float,
        n_diffusion_steps: int = 500,
        ess_threshold: float = 0.5,
        resampling_method: str = "systematic",
        include_vp_divergence: bool = True,
        energy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.reverse_sde = reverse_sde
        self.beta0 = beta0
        self.beta1 = beta1
        self.n_diffusion_steps = n_diffusion_steps
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method
        self.include_vp_divergence = include_vp_divergence
        self.energy_fn = energy_fn

    def run(
        self,
        N: int,
        d: int,
        device: torch.device,
        show_progress: bool = True,
    ) -> tuple[ParticleCloud, SMCDiagnostics]:
        from tqdm import tqdm
        
        
        x = torch.randn(N, d, device=device)
        logw = torch.zeros(N, device=device)

        times = torch.linspace(1, 0, self.n_diffusion_steps + 1, device=device)
        dt = abs(times.diff()[0])
        gamma = self.beta1 / self.beta0

        diagnostics = SMCDiagnostics()
        _dummy_energy = self.energy_fn if self.energy_fn is not None else lambda z: torch.zeros(z.shape[0], device=z.device)

        time_steps = times[:-1].tolist()
        iterator = range(len(time_steps))
        if show_progress:
            iterator = tqdm(iterator, desc="FKC", dynamic_ncols=True)
        with torch.no_grad():
            for i in iterator:
                t_val = time_steps[i]
                t = torch.full((N,), t_val, device=device)

                score = self.reverse_sde.model.score(t, x, self.reverse_sde.schedule)
                score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e3, 1e3) # Handle NaNs/Infs in score
                drift = self.reverse_sde.schedule.reverse_drift(t, x, gamma * score)
                diffusion = self.reverse_sde.schedule.reverse_diffusion(t, x)
                beta_t = diffusion.square().view(N)

                score_norm_sq = score.square().sum(dim=1)
                div_term = (gamma - 1) * (-d * 0.5) * beta_t
                correction = (div_term if self.include_vp_divergence else 0) + 0.5 * beta_t * gamma * (gamma - 1) * score_norm_sq

                logw += correction * dt
                x = x + drift * dt + diffusion * (dt**0.5) * torch.randn_like(x)

                current_ess = ess(logw).item()
                ess_ratio = current_ess / N

                if ess_ratio < self.ess_threshold:
                    p = torch.softmax(logw, dim=0)
                    idx = torch.multinomial(p, num_samples=N, replacement=True)
                    x = x[idx]
                    logw = torch.zeros(N, device=device)
                    diagnostics.n_resamples += 1

                cloud = ParticleCloud(x, log_weights=logw)
                diagnostics.record(t_val, cloud, _dummy_energy)

                if show_progress:
                    postfix = dict(t=f"{t_val:.3f}", ess=f"{ess_ratio:.2f}")
                    if self.energy_fn is not None:
                        postfix["E_min"] = f"{diagnostics.best_energies[-1]:.3f}"
                    iterator.set_postfix(postfix)  # type: ignore[union-attr]

        return ParticleCloud(x, log_weights=logw), diagnostics

