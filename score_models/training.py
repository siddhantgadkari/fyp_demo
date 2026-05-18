from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from .losses import dsm_loss
from .mlp_score import MLPScore
from ..schedules.base import DiffusionSchedule


@dataclass
class TrainingConfig:
    n_steps: int = 20_000
    batch_size: int = 512
    lr: float = 2e-4
    t_eps: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 500
    seed: int = 0


def _ema_update(ema_model: MLPScore, model: MLPScore, decay: float) -> None:
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train_score_model(
    model: MLPScore,
    schedule: DiffusionSchedule,
    sample_fn: Callable[[int], torch.Tensor],
    config: TrainingConfig,
    device: torch.device,
    callback: Optional[Callable[[int, float], None]] = None,
) -> tuple[MLPScore, List[float]]:
    """Train a score model with denoising score matching.

    Args:
        model:      MLPScore (eps-prediction parameterisation).
        schedule:   DiffusionSchedule (VP or VE).
        sample_fn:  Function (n: int) -> [n, d] samples from the training target.
        config:     TrainingConfig dataclass.
        device:     Torch device.
        callback:   Optional (step, loss) -> None hook for logging/plotting.

    Returns:
        (ema_model, loss_history)
        ema_model is the EMA-smoothed model, which is typically used for inference.
    """
    torch.manual_seed(config.seed)
    model = model.to(device)

    ema_model = MLPScore(
        dim=model.dim,
        hidden_dims=tuple(
            l.out_features
            for l in model.net
            if isinstance(l, torch.nn.Linear)
        )[:-1],
        time_embed_dim=model.time_embed.embed_dim,
        predict_score=model.predict_score,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_steps)

    loss_history: List[float] = []
    running_loss = 0.0

    model.train()
    pbar = tqdm(range(1, config.n_steps + 1), desc="Training score model", dynamic_ncols=True)
    for step in pbar:
        x0 = sample_fn(config.batch_size).to(device)

        loss = dsm_loss(model, x0, schedule, t_eps=config.t_eps)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        _ema_update(ema_model, model, config.ema_decay)

        running_loss += loss.item()
        if step % config.log_every == 0:
            avg = running_loss / config.log_every
            loss_history.append(avg)
            pbar.set_postfix(loss=f"{avg:.4f}")
            running_loss = 0.0
            if callback is not None:
                callback(step, avg)

    ema_model.eval()
    return ema_model, loss_history
