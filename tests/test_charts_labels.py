"""Direct labels have to stay readable when several series end together.

Three curves ending at exactly zero printed three labels on top of each other and
the chart still rendered, which is the kind of failure that only a person looking
at the picture catches. These tests catch it instead.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from standarderror.viz import charts  # noqa: E402


class TestLabelOffsets:
    def _axes(self, ylim=(0.0, 1.0)):
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.set_ylim(*ylim)
        fig.canvas.draw()
        return ax

    def test_well_separated_labels_are_not_moved(self):
        """Because the fix must not disturb charts that were already fine."""
        ax = self._axes()
        got = charts._label_offsets(ax, [0.05, 0.35, 0.65, 0.95])
        assert np.allclose(got, 0.0), got
        plt.close("all")

    def test_labels_at_the_same_value_are_separated(self):
        ax = self._axes()
        got = charts._label_offsets(ax, [0.0, 0.0, 0.0, 0.4])
        moved = np.sort(got[:3])
        assert moved[0] == 0.0
        gaps = np.diff(moved)
        assert (gaps >= charts.LABEL_GAP_POINTS - 1e-9).all(), got
        plt.close("all")

    def test_the_lowest_label_stays_on_its_own_value(self):
        """So the chart still has one label the reader can trust exactly."""
        ax = self._axes()
        got = charts._label_offsets(ax, [0.0, 0.0, 0.0])
        assert got.min() == 0.0
        plt.close("all")

    def test_a_stack_that_would_overflow_slides_down(self):
        ax = self._axes()
        got = charts._label_offsets(ax, [1.0, 1.0, 1.0, 1.0])
        height = ax.get_window_extent().height * 72.0 / ax.figure.dpi
        top = height + got.max()
        assert top <= height + 1e-9, (got, height)
        assert (np.diff(np.sort(got)) >= charts.LABEL_GAP_POINTS - 1e-9).all()
        plt.close("all")

    def test_order_is_preserved(self):
        """A label must never be pushed past a series it started below, or the
        chart would read as though two lines had swapped places."""
        ax = self._axes()
        values = np.array([0.0, 0.01, 0.02, 0.9])
        got = charts._label_offsets(ax, values)
        height = ax.get_window_extent().height * 72.0 / ax.figure.dpi
        placed = values * height + got            # ylim is (0, 1) here
        assert np.argsort(placed).tolist() == np.argsort(values).tolist(), placed
        plt.close("all")


class TestTheChartUsesThem:
    def test_a_chart_whose_series_converge_still_labels_all_of_them(self):
        frame = pd.DataFrame(
            {"a": [1.0, 0.5, 0.0], "b": [1.0, 0.4, 0.0], "c": [1.0, 0.3, 0.0]},
            index=[1, 2, 3])
        fig, ax = charts.lines(frame, title="t", alt="a", ylim=(0.0, 1.0))
        texts = [t.get_text().strip() for t in ax.texts]
        assert {"a", "b", "c"} <= set(texts), texts
        offsets = sorted(t.xyann[1] for t in ax.texts
                         if t.get_text().strip() in {"a", "b", "c"})
        assert (np.diff(offsets) >= charts.LABEL_GAP_POINTS - 1e-9).all(), offsets
        plt.close("all")

    def test_a_crowded_chart_colours_every_label(self):
        """Including the one that did not move: its marker is under someone
        else's, so the colour is the only mapping the reader has left."""
        frame = pd.DataFrame({"a": [1.0, 0.0], "b": [1.0, 0.0]}, index=[1, 2])
        fig, ax = charts.lines(frame, title="t", alt="a", ylim=(0.0, 1.0))
        by_label = {t.get_text().strip(): t for t in ax.texts}
        line_colour = {ln.get_label(): ln.get_color() for ln in ax.lines}
        assert any(abs(by_label[lbl].xyann[1]) > 0.5 for lbl in ("a", "b")), \
            "nothing was nudged, so the test proves nothing"
        for lbl in ("a", "b"):
            assert by_label[lbl].get_color() == line_colour[lbl], lbl
        plt.close("all")

    def test_an_uncrowded_chart_keeps_its_labels_recessive(self):
        """The fix must not restyle charts that were already readable."""
        frame = pd.DataFrame({"a": [0.0, 0.9], "b": [1.0, 0.1]}, index=[1, 2])
        fig, ax = charts.lines(frame, title="t", alt="a", ylim=(0.0, 1.0))
        line_colour = {ln.get_label(): ln.get_color() for ln in ax.lines}
        for txt in ax.texts:
            lbl = txt.get_text().strip()
            if lbl in line_colour:
                assert txt.xyann[1] == 0.0, lbl
                assert txt.get_color() != line_colour[lbl], lbl
        plt.close("all")


class TestLabelsSurviveTheNotionTransport:
    """SVG minification strips leading and trailing whitespace from character
    data, and `tools/notion_figures.py` fails the build when a label changes. A
    label must therefore not carry whitespace it does not need."""

    def test_a_direct_label_has_no_padding_whitespace(self):
        frame = pd.DataFrame({"α = 0.1": [1.0, 0.4], "β": [0.2, 0.9]},
                             index=[1, 2])
        fig, ax = charts.lines(frame, title="t", alt="a")
        drawn = [t.get_text() for t in ax.texts]
        assert "α = 0.1" in drawn and "β" in drawn, drawn
        for txt in drawn:
            assert txt == txt.strip(), repr(txt)
        plt.close("all")
