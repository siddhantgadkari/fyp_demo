"""Configuration dataclasses for the toy 2D experiment."""
from __future__ import annotations

from dataclasses import dataclass, field

from amortised_annealing.score_models.training import TrainingConfig


@dataclass
class EnergyConfig:
    type: str = "double_well"   # "double_well" | "gmm" | "rastrigin" | "ackley"
    dim: int = 2
    beta_train: float = 1.0     # inverse temperature for training data


@dataclass
class DiffusionConfig:
    schedule: str = "vp"         # "vp" | "ve"
    beta_min: float = 0.1
    beta_max: float = 20.0
    t_eps: float = 1e-4
    n_reverse_steps: int = 500
    t_start: float = 1.0
    t_end: float = 1e-3


@dataclass
class ModelConfig:
    hidden_dims: tuple = (128, 128, 128)
    time_embed_dim: int = 64
    activation: str = "silu"


@dataclass
class SMCConfig:
    n_particles: int = 512
    beta_final: float = 10.0
    n_beta_steps: int = 30
    ladder: str = "geometric"    # "linear" | "geometric"
    ess_threshold: float = 0.5
    resampling: str = "systematic"


@dataclass
class DiffusionSMCConfig(SMCConfig):
    n_diffusion_steps: int = 20   # reverse SDE steps per SMC mutation
    t_start: float = 0.4          # partial noise injection time
    t_end: float = 1e-3
    use_score_scaling: bool = True
    langevin_steps: int = 0
    langevin_step_size: float = 5e-3


@dataclass
class LangevinConfig:
    step_size: float = 5e-3
    n_steps_per_beta: int = 20    # for classical annealed SMC


@dataclass
class ExperimentConfig:
    name: str = "toy_2d_double_well"
    seed: int = 0
    device: str = "auto"          # "auto" | "cpu" | "mps" | "cuda"
    output_dir: str = "results/toy_2d"
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(n_steps=15_000))
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    smc: SMCConfig = field(default_factory=SMCConfig)
    diffusion_smc: DiffusionSMCConfig = field(default_factory=DiffusionSMCConfig)
    langevin: LangevinConfig = field(default_factory=LangevinConfig)
