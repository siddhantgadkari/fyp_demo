"""Quantitative metrics for evaluating sample quality."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch


def compute_metrics(
    samples: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    log_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute a standard set of evaluation metrics.

    Args:
        samples:     [N, d] particle positions
        energy_fn:   x -> E(x) [N]
        log_weights: [N] optional importance weights (uniform if None)

    Returns:
        Dictionary of metric names to values.
    """
    with torch.no_grad():
        energies = energy_fn(samples)

    if log_weights is not None:
        lw = log_weights - log_weights.max()
        w = torch.exp(lw)
        w = w / w.sum()
        weighted_mean_e = (w * energies).sum().item()
        ess_val = (1.0 / (w**2).sum()).item()
    else:
        weighted_mean_e = energies.mean().item()
        ess_val = float(samples.shape[0])

    return {
        "best_energy": energies.min().item(),
        "mean_energy": energies.mean().item(),
        "weighted_mean_energy": weighted_mean_e,
        "std_energy": energies.std().item(),
        "q10_energy": torch.quantile(energies, 0.1).item(),
        "q25_energy": torch.quantile(energies, 0.25).item(),
        "q50_energy": torch.quantile(energies, 0.5).item(),
        "ess": ess_val,
        "ess_ratio": ess_val / samples.shape[0],
        "n_particles": samples.shape[0],
    }


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: Optional[List[float]] = None,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel (median heuristic bandwidths).

    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]

    Args:
        x: [N, d] samples from distribution p
        y: [M, d] samples from distribution q
        bandwidths: list of kernel bandwidths (median heuristic if None)

    Returns:
        MMD value (non-negative; 0 iff p=q)
    """
    if bandwidths is None:
        # Median heuristic
        xy = torch.cat([x, y], dim=0)
        dists = torch.cdist(xy, xy)
        bandwidths = [dists.median().item() / 2.0 + 1e-6]

    def rbf(a, b):
        d2 = torch.cdist(a, b) ** 2
        k = sum(torch.exp(-d2 / (2 * bw**2)) for bw in bandwidths)
        return k / len(bandwidths)

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return (kxx + kyy - 2 * kxy).item()


def sinkhorn_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 100,
) -> float:
    """Sinkhorn divergence (regularised OT distance) between two sample clouds.

    Uses the log-domain Sinkhorn algorithm for numerical stability.

    Args:
        x:      [N, d]
        y:      [M, d]
        eps:    Regularisation parameter
        n_iter: Number of Sinkhorn iterations

    Returns:
        Sinkhorn divergence value
    """
    N, M = x.shape[0], y.shape[0]
    device = x.device

    C = torch.cdist(x, y) ** 2  # [N, M]

    log_a = -torch.log(torch.tensor(N, dtype=x.dtype, device=device)) * torch.ones(N, device=device)
    log_b = -torch.log(torch.tensor(M, dtype=x.dtype, device=device)) * torch.ones(M, device=device)

    log_u = torch.zeros(N, device=device)
    log_v = torch.zeros(M, device=device)

    K = -C / eps  # [N, M]

    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(K + log_u.unsqueeze(1), dim=0)

    log_T = K + log_u.unsqueeze(1) + log_v.unsqueeze(0)  # [N, M]
    ot_xy = (torch.exp(log_T) * C).sum().item()

    # Self-distances for Sinkhorn divergence (debiased)
    Kxx = -torch.cdist(x, x) ** 2 / eps
    log_ux = torch.zeros(N, device=device)
    log_vx = torch.zeros(N, device=device)
    for _ in range(n_iter):
        log_ux = log_a - torch.logsumexp(Kxx + log_vx.unsqueeze(0), dim=1)
        log_vx = log_a - torch.logsumexp(Kxx + log_ux.unsqueeze(1), dim=0)
    ot_xx = (torch.exp(Kxx + log_ux.unsqueeze(1) + log_vx.unsqueeze(0)) * torch.cdist(x, x)**2).sum().item()

    Kyy = -torch.cdist(y, y) ** 2 / eps
    log_uy = torch.zeros(M, device=device)
    log_vy = torch.zeros(M, device=device)
    for _ in range(n_iter):
        log_uy = log_b - torch.logsumexp(Kyy + log_vy.unsqueeze(0), dim=1)
        log_vy = log_b - torch.logsumexp(Kyy + log_uy.unsqueeze(1), dim=0)
    ot_yy = (torch.exp(Kyy + log_uy.unsqueeze(1) + log_vy.unsqueeze(0)) * torch.cdist(y, y)**2).sum().item()

    return ot_xy - 0.5 * (ot_xx + ot_yy)


def mode_coverage_gmm(
    samples: torch.Tensor,
    means: torch.Tensor,
    radius: float = 1.0,
) -> Dict[str, float]:
    """Measure mode coverage for a GMM target.

    A mode is considered "covered" if at least one particle falls within
    `radius` of its mean.

    Args:
        samples: [N, d]
        means:   [K, d] component means
        radius:  Coverage ball radius

    Returns:
        Dictionary with 'n_modes_covered', 'fraction_covered', 'min_mode_dist'
    """
    K = means.shape[0]
    dists = torch.cdist(samples, means.to(samples.device))  # [N, K]
    min_dists = dists.min(0).values  # [K]
    covered = (min_dists < radius).sum().item()
    return {
        "n_modes_covered": covered,
        "fraction_covered": covered / K,
        "mean_min_dist_to_mode": min_dists.mean().item(),
        "max_min_dist_to_mode": min_dists.max().item(),
    }
