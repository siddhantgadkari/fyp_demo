"""Plotting utilities for particles, energy landscapes, and diagnostics."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; caller can switch if needed
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_curve(
    losses: List[float],
    title: str = "Training loss",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses)
    ax.set_xlabel("Log step (x log_every)")
    ax.set_ylabel("DSM loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_energy_contour_2d(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    beta: float = 1.0,
    xlim: Tuple[float, float] = (-10, 10),
    ylim: Tuple[float, float] = (-10, 10),
    n_grid: int = 200,
    ax: Optional[plt.Axes] = None,
    device: torch.device = torch.device("cpu"),
    n_levels: int = 30,
) -> plt.Axes:
    """Plot the Boltzmann density exp(-beta * E(x)) as filled contours."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    xs = torch.linspace(xlim[0], xlim[1], n_grid, device=device)
    ys = torch.linspace(ylim[0], ylim[1], n_grid, device=device)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)

    with torch.no_grad():
        energies = energy_fn(grid).reshape(n_grid, n_grid).cpu().numpy()

    log_density = -beta * energies
    log_density -= log_density.max()
    density = np.exp(log_density)

    ax.contourf(
        XX.cpu().numpy(), YY.cpu().numpy(), density,
        levels=n_levels, cmap="Blues",
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    return ax


def plot_particles_2d(
    x: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: str = "steelblue",
    alpha: float = 0.5,
    s: float = 8.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> plt.Axes:
    """Scatter plot of 2D particles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    x_np = x.detach().cpu().numpy()
    ax.scatter(x_np[:, 0], x_np[:, 1], s=s, alpha=alpha, color=color, label=label)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if label:
        ax.legend(fontsize=8)
    ax.set_aspect("equal")
    return ax


def plot_particles_over_betas(
    snapshots: Dict[float, torch.Tensor],
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    xlim: Tuple[float, float] = (-10, 10),
    ylim: Tuple[float, float] = (-10, 10),
    n_cols: int = 4,
    save_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> plt.Figure:
    """Grid of particle scatter plots at different beta values."""
    betas = sorted(snapshots.keys())
    n_rows = (len(betas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, beta in enumerate(betas):
        ax = axes[i]
        plot_energy_contour_2d(energy_fn, beta=beta, xlim=xlim, ylim=ylim, ax=ax, device=device)
        plot_particles_2d(snapshots[beta], ax=ax, color="darkorange", alpha=0.4, s=5)
        ax.set_title(f"β = {beta:.2f}", fontsize=10)

    for j in range(len(betas), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_smc_diagnostics(
    diagnostics,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Four-panel diagnostic plot: ESS ratio, log-weight stats, best/mean energy."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    betas = diagnostics.betas

    axes[0, 0].plot(betas, diagnostics.ess_ratios, color="steelblue")
    axes[0, 0].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="threshold=0.5")
    axes[0, 0].set_xlabel("β")
    axes[0, 0].set_ylabel("ESS / N")
    axes[0, 0].set_title("Effective Sample Size Ratio")
    axes[0, 0].legend()

    axes[0, 1].plot(betas, diagnostics.log_weight_means, label="mean", color="steelblue")
    axes[0, 1].fill_between(
        betas,
        np.array(diagnostics.log_weight_means) - np.array(diagnostics.log_weight_stds),
        np.array(diagnostics.log_weight_means) + np.array(diagnostics.log_weight_stds),
        alpha=0.2, color="steelblue",
    )
    axes[0, 1].set_xlabel("β")
    axes[0, 1].set_ylabel("log weight")
    axes[0, 1].set_title("Log-weight mean ± std")

    axes[1, 0].plot(betas, diagnostics.best_energies, color="darkorange", label="best E")
    axes[1, 0].set_xlabel("β")
    axes[1, 0].set_ylabel("E(x)")
    axes[1, 0].set_title("Best energy found")
    axes[1, 0].legend()

    axes[1, 1].plot(betas, diagnostics.mean_energies, color="green", label="mean E")
    axes[1, 1].set_xlabel("β")
    axes[1, 1].set_ylabel("E(x)")
    axes[1, 1].set_title("Mean energy of particles")
    axes[1, 1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_energy_histogram(
    samples_dict: Dict[str, torch.Tensor],
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    beta: float,
    save_path: Optional[str] = None,
    bins: int = 50,
) -> plt.Figure:
    """Histogram of energies for multiple methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "darkorange", "green", "red", "purple"]

    for (name, x), color in zip(samples_dict.items(), colors):
        with torch.no_grad():
            e = energy_fn(x).cpu().numpy()
        ax.hist(e, bins=bins, alpha=0.5, label=name, color=color, density=True)

    ax.set_xlabel("Energy E(x)")
    ax.set_ylabel("Density")
    ax.set_title(f"Energy distribution at β = {beta:.2f}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def compare_methods(
    results: Dict[str, torch.Tensor],
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    xlim: Tuple[float, float] = (-10, 10),
    ylim: Tuple[float, float] = (-10, 10),
    beta: float = 1.0,
    save_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> plt.Figure:
    """Side-by-side particle scatter plots for each method against the target density."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ["darkorange", "green", "red", "purple", "brown"]
    for i, (name, x) in enumerate(results.items()):
        ax = axes[i]
        plot_energy_contour_2d(energy_fn, beta=beta, xlim=xlim, ylim=ylim, ax=ax, device=device)
        plot_particles_2d(x, ax=ax, color=colors[i % len(colors)], label=name, alpha=0.5, s=8)
        ax.set_title(name, fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
