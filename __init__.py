"""
amortised_annealing: A library for amortised annealing using diffusion models + SMC/FKC.

Core idea: train a score model once at a moderate inverse temperature beta_M, then
use it as a proposal mechanism inside an SMC loop to target colder distributions.
"""
