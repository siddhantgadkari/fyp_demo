from __future__ import annotations

import torch


class ReverseSDE:
    """Reverse-time SDE for a trained score network.

    For VP schedule the reverse SDE (Ito, running t: 1 -> 0) is:
        dx = [0.5*beta(t)*x + beta(t)*s_theta(t,x)] dt + sqrt(beta(t)) dW

    where dt > 0 and we advance from t_curr to t_curr - dt.

    For VE schedule the reverse SDE is:
        dx = -d(sigma^2)/dt * s_theta(t,x) dt + sqrt(d(sigma^2)/dt) dW

    The schedule exposes `reverse_drift` and `reverse_diffusion` so this class
    is schedule-agnostic.

    The score network is called as: model.score(t, x, schedule)
    which handles the eps-to-score conversion internally.
    """

    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        t_cur: torch.Tensor,
        dt: float,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        """One Euler-Maruyama step: x_{t-dt} = x_t + drift*dt + diffusion*sqrt(dt)*eps.

        Args:
            x:                 [N, d] current particles
            t_cur:             [N] or scalar current time
            dt:                Step size (positive; we subtract internally)
            temperature_scale: Multiply the score by this factor.
                               Set to beta_target/beta_train for heuristic annealing.
                               This is a proposal heuristic, NOT an exact correction.

        Returns:
            x_next: [N, d]
        """
        if t_cur.dim() == 0:
            t_cur = t_cur.expand(x.shape[0])

        score = self.model.score(t_cur, x, self.schedule) * temperature_scale
        # Guard against score blow-up from poorly-trained or out-of-distribution inputs
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score = score.clamp(-1e3, 1e3)
        drift = self.schedule.reverse_drift(t_cur, x, score)
        diffusion = self.schedule.reverse_diffusion(t_cur, x)
        eps = torch.randn_like(x)
        x_next = x + drift * dt + diffusion * (dt**0.5) * eps
        return torch.nan_to_num(x_next, nan=0.0)

    @torch.no_grad()
    def ddim_step(
        self,
        x: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        """Deterministic DDIM step (probability-flow ODE) for VP schedules.

        x_{t_next} = alpha_{t_next}/alpha_{t_cur} * x_t
                     + (sigma_{t_next} - alpha_{t_next}/alpha_{t_cur} * sigma_{t_cur}) * eps_pred

        where eps_pred = model(t_cur, x).
        """
        if t_cur.dim() == 0:
            t_cur = t_cur.expand(x.shape[0])
        if t_next.dim() == 0:
            t_next = t_next.expand(x.shape[0])

        alpha_cur = self.schedule.alpha(t_cur).view(-1, 1)
        sigma_cur = self.schedule.sigma(t_cur).view(-1, 1)
        alpha_next = self.schedule.alpha(t_next).view(-1, 1)
        sigma_next = self.schedule.sigma(t_next).view(-1, 1)

        # predict x0 from current x and eps_pred
        eps_pred = self.model(t_cur, x) * temperature_scale
        x0_pred = (x - sigma_cur * eps_pred) / (alpha_cur + 1e-8)

        return alpha_next * x0_pred + sigma_next * eps_pred
