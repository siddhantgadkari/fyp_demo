# amortised_annealing

A research library for **amortised inference-time annealing** using diffusion models, Sequential Monte Carlo (SMC), and Feynman–Kac Correctors (FKC).

## Core idea

We want to sample from the Boltzmann distribution at a cold inverse temperature β_H:

```
π_{β_H}(x) ∝ exp(-β_H E(x))
```

Training directly at high β is hard. Instead:

1. **Train once** at a moderate temperature β_M where sampling is tractable.
2. **At inference time**, use the learned reverse diffusion dynamics as a proposal mechanism inside an SMC loop.
3. **Correct** the proposal/target mismatch with importance weights (AIS/FKC).

The key claim being investigated: *can a single score model trained at β_M be reused as an amortised proposal for β > β_M, with SMC weights making the inference principled?*

## Library structure

```
amortised_annealing/
├── distributions/          # Energy functions (E: R^d → R)
│   ├── base.py             #   Energy ABC + BoltzmannDistribution
│   ├── gaussian_mixture.py #   GMM (analytic samples at β=1)
│   ├── double_well.py      #   DoubleWell, ManyWell
│   ├── rastrigin.py        #   Rastrigin (multimodal)
│   ├── ackley.py           #   Ackley
│   └── rosenbrock.py       #   Rosenbrock (banana)
│
├── schedules/              # Diffusion noise schedules
│   ├── base.py             #   DiffusionSchedule ABC: alpha(t), sigma(t)
│   ├── vp.py               #   VP (Variance Preserving): standard DDPM-style
│   ├── ve.py               #   VE (Variance Exploding): Song et al.
│   └── beta_ladders.py     #   linear / geometric β ladders
│
├── score_models/           # Score network + training
│   ├── mlp_score.py        #   MLP with sinusoidal time embedding
│   ├── losses.py           #   Denoising Score Matching loss
│   └── training.py         #   Training loop with EMA
│
├── diffusion/              # Forward/reverse diffusion
│   ├── forward.py          #   Marginal sampling, conditional score
│   ├── reverse_sde.py      #   Reverse-time SDE (Euler-Maruyama + DDIM)
│   └── samplers.py         #   Full reverse trajectory samplers
│
├── smc/                    # Sequential Monte Carlo
│   ├── particles.py        #   ParticleCloud dataclass
│   ├── resampling.py       #   Multinomial / systematic / stratified
│   ├── ess.py              #   Effective Sample Size
│   └── smc_sampler.py      #   Generic SMC loop (mutation + weight + resample)
│
├── fkc/                    # Feynman–Kac Correctors
│   ├── weight_updates.py   #   AIS weight + diffusion-informed weight
│   ├── annealed_correctors.py  # AnnealedDiffusionProposal (main algorithm)
│   └── product_correctors.py   # ProductScoreProposal + Hutchinson FKC term
│
├── baselines/              # Comparison methods
│   ├── langevin.py         #   ULA + MALA
│   ├── simulated_annealing.py  # Simulated annealing
│   └── annealed_smc.py     #   Classical annealed SMC with ULA mutation
│
├── diagnostics/            # Evaluation + plotting
│   ├── plotting.py         #   Particle scatter, contour, SMC diagnostics
│   └── metrics.py          #   ESS, energy stats, MMD, Sinkhorn, mode coverage
│
├── experiments/
│   └── toy_2d/
│       ├── config.py       #   Typed experiment config dataclasses
│       └── run.py          #   Main experiment script
│
└── tests/                  # 57 unit tests (all passing)
    ├── test_distributions.py
    ├── test_schedules.py
    ├── test_resampling.py
    ├── test_smc.py
    ├── test_score_model.py
    └── test_fkc_weights.py
```

## Quick start

### Run the toy 2D experiment

```bash
cd /path/to/FYP
uv run python amortised_annealing/experiments/toy_2d/run.py \
    --energy double_well \
    --beta-train 1.0 \
    --beta-final 10.0 \
    --n-particles 512 \
    --n-steps 15000
```

This will:
1. Train a score model on `DoubleWell` at β_M = 1.0
2. Generate β_M samples via reverse diffusion
3. Run four methods to target β_H = 10.0
4. Save comparison plots and metrics to `results/toy_2d/`

Available energy types: `double_well`, `gmm`, `rastrigin`, `ackley`

### Run tests

```bash
cd /path/to/FYP
uv run python -m pytest amortised_annealing/tests/ -v
```

### Use interactively

```python
import torch
from amortised_annealing.distributions import DoubleWell
from amortised_annealing.schedules import VPSchedule, make_ladder
from amortised_annealing.score_models import MLPScore, TrainingConfig, train_score_model
from amortised_annealing.diffusion.reverse_sde import ReverseSDE
from amortised_annealing.diffusion.samplers import euler_maruyama_sample
from amortised_annealing.baselines import ClassicalAnnealedSMC
from amortised_annealing.fkc.annealed_correctors import AnnealedDiffusionProposal
from amortised_annealing.smc import ParticleCloud, SMCSampler

device = torch.device("mps")  # or "cuda" / "cpu"

# 1. Energy and schedule
energy = DoubleWell(dim=2)
schedule = VPSchedule()

# 2. Train score model at beta_M
model = MLPScore(dim=2, hidden_dims=(128, 128, 128))
cfg = TrainingConfig(n_steps=10000, batch_size=512)
sample_fn = lambda n: energy.sample(n, device=device) or torch.randn(n, 2, device=device)
ema_model, losses = train_score_model(model, schedule, sample_fn, cfg, device)

# 3. Generate beta_M samples
reverse_sde = ReverseSDE(ema_model, schedule)
x_init = euler_maruyama_sample(reverse_sde, n_samples=512, n_steps=500, device=device)

# 4. Anneal to beta_H with diffusion-informed SMC
beta_ladder = make_ladder(1.0, 10.0, n_steps=30)
proposal = AnnealedDiffusionProposal(
    reverse_sde, energy.energy, beta_train=1.0,
    n_diffusion_steps=20, use_score_scaling=True
)
cloud = ParticleCloud(x=x_init, log_weights=torch.zeros(512, device=device))
sampler = SMCSampler(proposal.mutation_kernel, proposal.weight_update, energy.energy)
final_cloud, diagnostics = sampler.run(cloud, beta_ladder.to(device))
```

## Mathematical background

### Forward diffusion (VP)

```
x_t = alpha(t) * x_0 + sigma(t) * eps,   eps ~ N(0, I)
alpha(t) = exp(-0.5 * int_0^t beta(s) ds)
sigma(t) = sqrt(1 - alpha(t)^2)
```

### Denoising score matching

```
L(θ) = E_{t, x_0, eps} [ || eps_θ(t, x_t) - eps ||^2 ]
```

The network predicts noise; convert to score via `s(t, x) = -eps(t, x) / sigma(t)`.

### SMC annealing weight update

```
log w_k += -(beta_k - beta_{k-1}) * E(x_after_mutation)
```

This is the standard AIS incremental weight. It is exact for identity mutations and valid (though higher-variance) for ergodic mutation kernels.

### Why score scaling is a heuristic, not a fix

Multiplying the score by `beta_k / beta_M` biases the proposal toward the colder target but does **not** make the proposal distribution exactly equal to `pi_{beta_k}`. The SMC weights must correct this mismatch. Marking this clearly prevents the common error of reporting "cold samples" without acknowledging the correction step.

## Extending to PITA-style progressive training

The library is designed to support a full progressive training loop later:

```python
for m in range(M):
    train_score_model(model_m, beta_m, samples_from_previous_stage)
    cloud = diffusion_informed_smc(model_m, beta_m, beta_{m+1})
    samples_next = cloud.x  # pass to next stage
```

This is not yet implemented but the SMCSampler and AnnealedDiffusionProposal interfaces are designed to support it with minimal changes.

## Device notes

- Default: `float32` everywhere (MPS on Apple Silicon dislikes `float64`).
- Device selection: set `--device auto` to prefer CUDA > MPS > CPU.
- Avoid storing full particle trajectories when on MPS to prevent OOM.

## References

1. Doucet, A. et al. (2024). *Feynman–Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts.*
2. Phillips et al. (2024). *PITA: Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities.*
3. Del Moral, P. (2004). *Feynman–Kac Formulae.*
4. Song, Y. et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.*
