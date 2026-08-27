"""standarderror — a pipeline for research-grade blog posts on financial ML.

Layout:

* `sources`   public data adapters (FRED, ECB, ECOS, BIS, HMDA, market, local)
* `dynamics`  ODE / PDE / SDE generators with known ground truth
* `models`    echo state networks, NG-RC, and the baselines you must beat
* `xai`       attribution methods and reservoir-specific probes
* `uq`        conformal prediction, and SCMs with a known causal effect
* `viz`       one validated chart style, light and dark
* `render`    Post object -> Hugo page, Medium crosspost, Notion page

The design bias throughout: make the *honest* thing the easy thing. Baselines are
first-class, the split helpers refuse to leak, `Post.audit()` blocks a post with
undocumented figures, and every fetch records provenance.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import dynamics, models, render, sources, uq, viz, xai  # noqa: E402
from .config import SETTINGS  # noqa: E402
from .render import Post, Section  # noqa: E402

__all__ = ["sources", "dynamics", "models", "xai", "uq", "viz", "render",
           "Post", "Section", "SETTINGS", "__version__"]


def environment() -> dict:
    """Version stamp for the reproducibility block of a post."""
    import platform
    import subprocess

    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=SETTINGS.repo_root, capture_output=True, text=True,
            timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "standarderror": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "git_commit": commit,
        "seed": SETTINGS.seed,
    }
