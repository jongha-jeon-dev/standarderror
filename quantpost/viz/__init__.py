"""Charts with one consistent, accessibility-validated style.

    from quantpost.viz import charts, theme
    fig_meta, _ = charts.lines(df, title="...", path="build/img/f1.png")

Note there is no dual-axis helper anywhere in this package, by design.
"""

from . import charts, theme
from .charts import Figure

__all__ = ["charts", "theme", "Figure"]
