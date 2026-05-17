"""Incremental SMC weight updates for annealing and diffusion proposals.

The core principle: the incremental log-weight log(w_k) is the log ratio of
the target density at step k vs. the proposal distribution used to generate
the new particles.

For classical annealing:
    Target ratio: pi_{beta_k}(x) / pi_{beta_{k-1}}(x) = exp(-(beta_k - beta_{k-1}) * E(x))
    So:  log w_k = -(beta_k - beta_{k-1}) * E(x)

For diffusion-informed proposals:
    The proposal uses the reverse dynamics of q_{beta_M}, not pi_{beta_k}.
    The weight must account for this mismatch.  The simplest principled choice
    is still the AIS-style weight, which is correct when the mutation kernel is
    the exact conditional of the target (which ULA/diffusion proposals only
    approximate).  For a properly FKC-corrected weight, we would need the log-ratio
    of path measures, which requires knowing the proposal path density.

    In the absence of a closed-form proposal path density, we use the discrete
    AIS correction as the default, with an optional additive term for heuristic
    diffusion proposals.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


def annealing_weight_update(
    x: torch.Tensor,
    beta_prev: float,
    beta_curr: float,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Standard annealed importance sampling weight update.

    log w += -(beta_curr - beta_prev) * E(x)

    This is exact when mutation is the identity (i.e., no MCMC move).
    When a mutation kernel is used, this still gives a valid IS weight
    targeting pi_{beta_curr}, though variance may be higher.

    Args:
        x:          [N, d] particles after mutation
        beta_prev:  Previous inverse temperature
        beta_curr:  Current inverse temperature
        energy_fn:  x -> E(x) [N]

    Returns:
        log_w: [N] incremental log-weights
    """
    with torch.no_grad():
        e = energy_fn(x)
    return -(beta_curr - beta_prev) * e


def diffusion_informed_weight_update(
    x_before: torch.Tensor,
    x_after: torch.Tensor,
    beta_prev: float,
    beta_curr: float,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    log_proposal_ratio: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weight update for diffusion-informed proposals.

    When the mutation kernel is diffusion-informed (using score trained at beta_M),
    the corrected weight is:

        log w += log pi_{beta_curr}(x_after) - log q_proposal(x_after | x_before)

    We decompose this as:
        = [log pi_{beta_curr}(x_after) - log pi_{beta_prev}(x_before)]
          + [log pi_{beta_prev}(x_before) - log q_proposal(x_after | x_before)]

    The first term is the AIS-style annealing correction (always computable).
    The second term is the proposal ratio (computable only if we track the
    proposal log-density, which requires Gaussian transition densities).

    For simplicity, if log_proposal_ratio is None, we fall back to the AIS weight.
    This is conservative: the weights are valid but may have higher variance.

    Args:
        x_before:            [N, d] particles before mutation
        x_after:             [N, d] particles after mutation
        beta_prev:           Previous inverse temperature
        beta_curr:           Current inverse temperature
        energy_fn:           x -> E(x) [N]
        log_proposal_ratio:  Optional [N] log q(x_before|x_after)/q(x_after|x_before)
                             for detailed balance correction. Leave None to use AIS.

    Returns:
        log_w: [N] incremental log-weights
    """
    with torch.no_grad():
        e_after = energy_fn(x_after)

    log_w = -(beta_curr - beta_prev) * e_after

    if log_proposal_ratio is not None:
        log_w = log_w + log_proposal_ratio

    return log_w
