from __future__ import annotations

import torch


def linear_ladder(beta_min: float, beta_max: float, n_steps: int) -> torch.Tensor:
    """Linearly spaced inverse-temperature ladder from beta_min to beta_max."""
    return torch.linspace(beta_min, beta_max, n_steps + 1)


def geometric_ladder(beta_min: float, beta_max: float, n_steps: int) -> torch.Tensor:
    """Geometrically spaced inverse-temperature ladder from beta_min to beta_max."""
    log_min = torch.log(torch.tensor(beta_min))
    log_max = torch.log(torch.tensor(beta_max))
    return torch.exp(torch.linspace(log_min, log_max, n_steps + 1))


def make_ladder(
    beta_min: float,
    beta_max: float,
    n_steps: int,
    kind: str = "geometric",
) -> torch.Tensor:
    """Factory for beta ladders.

    Args:
        beta_min: Starting inverse temperature (training temperature).
        beta_max: Target inverse temperature.
        n_steps:  Number of annealing steps (ladder has n_steps+1 entries).
        kind:     'linear' or 'geometric'.

    Returns:
        Tensor of shape [n_steps+1] with beta_ladder[0]=beta_min, [-1]=beta_max.
    """
    if kind == "linear":
        return linear_ladder(beta_min, beta_max, n_steps)
    elif kind == "geometric":
        return geometric_ladder(beta_min, beta_max, n_steps)
    else:
        raise ValueError(f"Unknown ladder kind: {kind!r}. Choose 'linear' or 'geometric'.")
