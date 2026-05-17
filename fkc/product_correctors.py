"""Product-of-experts score combination with FKC correction placeholder.

Given two independently trained score models q1_t, q2_t, the product target is:
    p_t(x) ∝ q1_t(x) * q2_t(x)

The naive combined score is:  score_prod = score_1 + score_2

Under the reverse SDE, using score_prod as the drift produces a path measure
that does not exactly target the product distribution — there is a correction
term from the Fokker-Planck equation.  This module:

1.  Combines scores naively (product proposal score).
2.  Provides a placeholder for the FKC inner-product correction term.

The FKC correction for the product target (from Section 3.2 of the FKC paper)
introduces a weight process driven by:
    dlog w = [∇·(g^2 * score_prod) - (nabla log q1_t · nabla log q2_t) * g^2] dt

For the standard isotropic Gaussian noising process:
    g^2(t) = beta(t) (VP) or sigma_dot(t) (VE)

Computing the divergence ∇·score requires a Jacobian trace, which is expensive
for large d.  We provide an exact implementation for small d and a
Hutchinson estimator for larger d.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


class ProductScoreProposal:
    """Proposal using sum of two score fields (product-of-experts heuristic).

    For a model trained on distribution q1 and a second model (or analytic
    score) for q2, this combines them as a score for q1*q2.

    Note:
        Using score_1 + score_2 as a reverse SDE drift produces a proposal
        for p ∝ q1*q2, but requires FKC weight correction because the
        marginal path measure of the combined SDE does not equal p_t at all t.
    """

    def __init__(
        self,
        model1,
        model2_score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        schedule,
    ):
        """
        Args:
            model1:           MLPScore trained on q1.
            model2_score_fn:  (t, x) -> score_2(t, x).  Can be analytic or a second network.
            schedule:         DiffusionSchedule shared by both models.
        """
        self.model1 = model1
        self.model2_score_fn = model2_score_fn
        self.schedule = schedule

    @torch.no_grad()
    def combined_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Score of the product target: s1(t,x) + s2(t,x)."""
        s1 = self.model1.score(t, x, self.schedule)
        s2 = self.model2_score_fn(t, x)
        return s1 + s2

    def fkc_weight_increment(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        dt: float,
        g_sq: torch.Tensor,
        n_hutchinson: int = 10,
    ) -> torch.Tensor:
        """Placeholder FKC log-weight increment for the product target.

        The continuous-time weight satisfies:
            d log w = [ div(g^2 * (s1+s2)) - g^2 * (s1 · s2) ] dt

        For linear drift (VP), div(g^2 * s_i) = g^2 * div(s_i) since g^2
        does not depend on x.

        Args:
            t:             [N] current times
            x:             [N, d] current particles
            dt:            Integration step size
            g_sq:          [N] or scalar g(t)^2 diffusion coefficient squared
            n_hutchinson:  Number of Hutchinson probe vectors for trace estimation

        Returns:
            log_w_increment: [N]
        """
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            s1 = self.model1.score(t, x_req, self.schedule)
            s2 = self.model2_score_fn(t, x_req)

            # Inner product correction: -g^2 * (s1 · s2)
            inner = (s1 * s2).sum(-1)  # [N]

            # Divergence via Hutchinson: E_v[v^T J s v] where J is Jacobian
            div_s1 = _hutchinson_trace(s1, x_req, n_hutchinson)
            div_s2 = _hutchinson_trace(s2, x_req, n_hutchinson)
            div_total = (div_s1 + div_s2).detach()

        if g_sq.dim() == 0:
            g_sq = g_sq.expand(x.shape[0])

        increment = (g_sq * div_total - g_sq * inner.detach()) * dt
        return increment


def _hutchinson_trace(
    output: torch.Tensor,
    input_x: torch.Tensor,
    n_probes: int,
) -> torch.Tensor:
    """Hutchinson trace estimator: Tr(J) ≈ 1/K sum_k v_k^T (J v_k).

    Args:
        output:   [N, d] network output (must have grad_fn w.r.t. input_x)
        input_x:  [N, d] inputs (requires_grad=True)
        n_probes: Number of Rademacher probe vectors

    Returns:
        trace: [N]
    """
    N, d = output.shape
    trace = torch.zeros(N, device=output.device)
    for _ in range(n_probes):
        v = torch.randint(0, 2, (N, d), device=output.device, dtype=output.dtype) * 2 - 1
        Jv = torch.autograd.grad(
            output, input_x, grad_outputs=v, retain_graph=True, create_graph=False
        )[0]
        trace += (Jv * v).sum(-1)
    return trace / n_probes
