from __future__ import annotations

import torch


def ess(log_weights: torch.Tensor) -> float:
    """Effective Sample Size from unnormalised log weights.

    ESS = (sum w_i)^2 / sum(w_i^2)

    Ranges from 1 (degenerate) to N (uniform).
    """
    lw = log_weights - log_weights.max()
    w = torch.exp(lw)
    w = w / w.sum()
    return (1.0 / (w**2).sum()).item()


def log_ess_ratio(log_weights: torch.Tensor) -> float:
    """log(ESS / N): convenient for thresholding."""
    import math
    N = log_weights.shape[0]
    return math.log(ess(log_weights)) - math.log(N)
