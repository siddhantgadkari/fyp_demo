"""FKC path-aware diffusion SMC proposal.

This module implements the Feynman-Kac corrected mutation kernel. Unlike the
AIS-style weight update in `annealed_correctors.py` (which uses the
terminal particle position), the FKC correction accumulates log-weights along
the entire reverse diffusion trajectory:

    d log w_t = g_β(t, X_t) dt

where (score-norm approximation, divergence term omitted):

    g_β(t, x) = (β_rel - 1) * 0.5 * β(t) * β_rel * ||s_θ(t, x)||²

- β_rel = β_curr / β_train  (annealing ratio, same scalar used for score scaling)
- β(t)  = VP instantaneous noise level from the diffusion schedule
- s_θ   = unscaled score from the trained model


The weight is stored in `self._last_fkc_logw` during `mutation_kernel` and
read back in `weight_update`. This is safe because SMCSampler always calls
mutation then weight_update sequentially for the same batch.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from ..diffusion.reverse_sde import ReverseSDE
from .weight_updates import annealing_weight_update


class FKCAnnealedDiffusionProposal:
    """Mutation kernel with FKC path-space log-weight accumulation.

    Inlines the Euler-Maruyama loop (rather than delegating to ReverseSDE.step)
    so the unscaled score is available at every substep for FKC accumulation.
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
        self.reverse_sde = reverse_sde
        self.energy_fn = energy_fn
        self.beta_train = beta_train
        self.n_diffusion_steps = n_diffusion_steps
        self.t_start = t_start
        self.t_end = t_end
        self.use_score_scaling = use_score_scaling
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size
        self._last_fkc_logw: Optional[torch.Tensor] = None

    def mutation_kernel(self, x: torch.Tensor, beta_curr: float) -> torch.Tensor:
        """Reverse SDE with simultaneous FKC log-weight accumulation.

        Runs the noise-reinject trick then n_diffusion_steps of Euler-Maruyama,
        accumulating the FKC path correction at every step.
        """
        beta_rel = beta_curr / self.beta_train
        scale = beta_rel if self.use_score_scaling else 1.0
        device = x.device
        schedule = self.reverse_sde.schedule
        model = self.reverse_sde.model
        N = x.shape[0]

        # Noise-reinject trick: add forward noise then denoise (same as AnnealedDiffusionProposal)
        t_inj = torch.full((N,), self.t_start, device=device)
        x_noisy, _ = schedule.marginal_sample(x, t_inj)

        times = torch.linspace(self.t_start, self.t_end, self.n_diffusion_steps + 1, device=device)
        dt = (self.t_start - self.t_end) / self.n_diffusion_steps

        fkc_logw = torch.zeros(N, device=device)
        # print("using updated kernel")
        with torch.no_grad():
            x_cur = x_noisy
            for i in range(self.n_diffusion_steps):
                t = times[i].expand(N)

                # Unscaled score — used for both FKC accumulation and (scaled) drift
                score = model.score(t, x_cur, schedule)
                raw_score = model.score(t, x_cur, schedule)

                # print("raw finite:", torch.isfinite(raw_score).all().item())
                # print("raw nan:", torch.isnan(raw_score).any().item())
                # print("raw inf:", torch.isinf(raw_score).any().item())
                # print("raw min/max:", raw_score.nan_to_num().min().item(), raw_score.nan_to_num().max().item())
                score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
                score = score.clamp(-1e3, 1e3)
                # print(f"step {i+1}/{self.n_diffusion_steps}, score: mean {score.mean().item():.3f}, std {score.std().item():.3f}")

                # FKC increment: g_β(t, x) * dt  (score-norm term only)
                # β(t) is the VP instantaneous noise level = diffusion coefficient²
                beta_t = schedule.beta(t)                      # [N]
                # print(f"step {i+1}/{self.n_diffusion_steps}, beta_t: mean {beta_t.mean().item():.3f}, std {beta_t.std().item():.3f}")
                norm_sq = (score ** 2).sum(dim=-1)             # [N]
                # print(f"step {i+1}/{self.n_diffusion_steps}, score norm²: mean {norm_sq.mean().item():.3f}, std {norm_sq.std().item():.3f}")
                fkc_logw += (beta_rel - 1.0) * 0.5 * beta_t * beta_rel * norm_sq * dt
                # print(f"step {i+1}/{self.n_diffusion_steps}, FKC logw increment: mean {fkc_logw.mean().item():.3f}, std {fkc_logw.std().item():.3f}")

                # Euler-Maruyama step with (optionally) scaled score
                drift = schedule.reverse_drift(t, x_cur, score * scale)
                diffusion = schedule.reverse_diffusion(t, x_cur)
                eps = torch.randn_like(x_cur)
                x_cur = x_cur + drift * dt + diffusion * (dt ** 0.5) * eps
                x_cur = torch.nan_to_num(x_cur, nan=0.0)

        if self.langevin_steps > 0:
            x_cur = self._ula_steps(x_cur, beta_curr)

        self._last_fkc_logw = fkc_logw
        # print("FKC log-weight stats: mean {:.3f}, std {:.3f}".format(fkc_logw.mean().item(), fkc_logw.std().item()))
        return x_cur

    def weight_update(
        self, x: torch.Tensor, beta_prev: float, beta_curr: float
    ) -> torch.Tensor:
        """FKC path-space IS weight only.

        The accumulated ∫g_β dt is the complete likelihood ratio
        d(annealed path measure)/d(proposal path measure) by Girsanov's theorem
        (Prop D.1 of FKC paper).
        """
        if self._last_fkc_logw is None:
            print("Warning: FKC log-weight is None, returning zeros. This should only happen if weight_update is called before mutation_kernel.")
            return torch.zeros(x.shape[0], device=x.device)
        return self._last_fkc_logw

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
