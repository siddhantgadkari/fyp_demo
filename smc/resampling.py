from __future__ import annotations

import torch


def multinomial_resample(log_weights: torch.Tensor) -> torch.Tensor:
    """Standard multinomial resampling.

    Args:
        log_weights: [N] unnormalised log weights

    Returns:
        indices: [N] integer indices into the particle array
    """
    w = _normalise(log_weights)
    N = w.shape[0]
    return torch.multinomial(w, N, replacement=True)


def systematic_resample(log_weights: torch.Tensor) -> torch.Tensor:
    """Systematic resampling (lower variance than multinomial).

    Uses a single uniform random variable to place N evenly-spaced points
    on the CDF, then inverts.
    """
    w = _normalise(log_weights)
    N = w.shape[0]
    cdf = torch.cumsum(w, dim=0)
    u0 = torch.rand(1, device=w.device, dtype=w.dtype) / N
    u = u0 + torch.arange(N, device=w.device, dtype=w.dtype) / N
    # Searchsorted: find where each u falls in the CDF
    indices = torch.searchsorted(cdf, u).clamp(0, N - 1)
    return indices


def stratified_resample(log_weights: torch.Tensor) -> torch.Tensor:
    """Stratified resampling: independent uniform in each stratum [k/N, (k+1)/N]."""
    w = _normalise(log_weights)
    N = w.shape[0]
    cdf = torch.cumsum(w, dim=0)
    u = (torch.rand(N, device=w.device, dtype=w.dtype) + torch.arange(N, device=w.device, dtype=w.dtype)) / N
    indices = torch.searchsorted(cdf, u).clamp(0, N - 1)
    return indices


def resample(
    log_weights: torch.Tensor,
    method: str = "systematic",
) -> torch.Tensor:
    """Dispatch resampling by name.

    Args:
        log_weights: [N]
        method:      'systematic' | 'stratified' | 'multinomial'

    Returns:
        indices: [N]
    """
    if method == "systematic":
        return systematic_resample(log_weights)
    elif method == "stratified":
        return stratified_resample(log_weights)
    elif method == "multinomial":
        return multinomial_resample(log_weights)
    else:
        raise ValueError(f"Unknown resampling method: {method!r}")


def _normalise(log_weights: torch.Tensor) -> torch.Tensor:
    lw = log_weights - log_weights.max()
    w = torch.exp(lw)
    return w / w.sum()
