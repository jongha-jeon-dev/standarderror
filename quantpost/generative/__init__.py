"""Generative models for sequences, and the controls that keep them honest.

    from quantpost.generative import diffusion, stylised

    ddpm = diffusion.DDPM.budget(length=64, hidden=256, max_iter=80)
    ddpm.fit(ddpm.windows(returns, stride=8))
    facts = stylised.stylised_facts(ddpm.sample(600), n_boot=200)

Two halves, and the second is the one that does the work:

* `diffusion` — a denoising diffusion probabilistic model small enough to train on
  two CPUs, written to make the mechanism legible rather than to win anything.
* `stylised` — the battery a generated return series is usually judged on, plus the
  two trivial generators that have to appear beside it before a match means
  anything: a shuffle of the training data, and a moving-block bootstrap of it.
"""

from . import diffusion, stylised

__all__ = ["diffusion", "stylised"]
