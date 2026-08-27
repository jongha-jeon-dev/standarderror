"""Tests for standarderror.ts.panelpairs, on panels whose answer is known.

The panels here are constructed so the right answer is arithmetic: series built
to be identical must come back at r = 1, series built from independent noise
must come back near zero, and the pair-type masks must partition the upper
triangle exactly. Nothing here depends on real World Bank data being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standarderror.ts import panelpairs as pp


def _long(iso_values: dict, years) -> pd.DataFrame:
    rows = []
    for iso, vals in iso_values.items():
        for y, v in zip(years, vals):
            rows.append({"iso3": iso, "year": int(y), "value": float(v)})
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    rng = np.random.default_rng(5)
    years = np.arange(2000, 2020)
    isos = ["AAA", "BBB", "CCC", "DDD"]
    frames = {}
    for name in ("alpha", "beta", "gamma"):
        frames[name] = _long(
            {iso: np.cumsum(rng.normal(0.3, 1.0, years.size)) for iso in isos},
            years)
    return pp.stack(frames, start=2000, end=2019, min_countries=4)


class TestStack:
    def test_shape_and_labels(self, panel):
        assert len(panel) == 12                 # 3 indicators x 4 countries
        assert panel.n_years == 20
        assert set(panel.indicator) == {"alpha", "beta", "gamma"}
        assert set(panel.country) == {"AAA", "BBB", "CCC", "DDD"}

    def test_drops_countries_missing_from_one_indicator(self):
        years = np.arange(2000, 2010)
        rng = np.random.default_rng(1)
        full = {i: rng.normal(size=years.size) for i in ("AAA", "BBB", "CCC")}
        short = {i: rng.normal(size=years.size) for i in ("AAA", "BBB")}
        frames = {"a": _long(full, years), "b": _long(short, years)}
        p = pp.stack(frames, start=2000, end=2009, min_countries=2)
        assert set(p.country) == {"AAA", "BBB"}
        assert len(p) == 4

    def test_drops_a_country_with_a_hole_in_the_middle(self):
        years = np.arange(2000, 2010)
        rng = np.random.default_rng(2)
        a = _long({i: rng.normal(size=years.size)
                   for i in ("AAA", "BBB", "CCC")}, years)
        a = a[~((a["iso3"] == "CCC") & (a["year"] == 2004))]
        b = _long({i: rng.normal(size=years.size)
                   for i in ("AAA", "BBB", "CCC")}, years)
        p = pp.stack({"a": a, "b": b}, start=2000, end=2009, min_countries=2)
        assert "CCC" not in set(p.country)

    def test_raises_when_too_few_countries_survive(self):
        years = np.arange(2000, 2010)
        rng = np.random.default_rng(3)
        frames = {"a": _long({"AAA": rng.normal(size=10)}, years)}
        with pytest.raises(ValueError, match="complete, non-constant data"):
            pp.stack(frames, start=2000, end=2009, min_countries=5)

    def test_drops_a_country_whose_series_never_moves(self):
        """Singapore is 100% urban every year; r against a flat line is undefined."""
        years = np.arange(2000, 2012)
        rng = np.random.default_rng(31)
        moving = {i: np.cumsum(rng.normal(size=years.size))
                  for i in ("AAA", "BBB", "CCC")}
        flat = dict(moving)
        flat["CCC"] = np.full(years.size, 100.0)
        p = pp.stack({"a": _long(moving, years), "b": _long(flat, years)},
                     start=2000, end=2011, min_countries=2)
        assert "CCC" not in set(p.country)
        assert p.dropped_constant == ("CCC",)
        # and the country goes whole, not just its flat series
        assert (p.country == "CCC").sum() == 0

    def test_keeping_constants_is_possible_but_then_correlation_raises(self):
        years = np.arange(2000, 2012)
        rng = np.random.default_rng(32)
        moving = {i: np.cumsum(rng.normal(size=years.size)) for i in ("AAA", "BBB")}
        flat = dict(moving)
        flat["BBB"] = np.full(years.size, 7.0)
        p = pp.stack({"a": _long(moving, years), "b": _long(flat, years)},
                     start=2000, end=2011, min_countries=2, drop_constant=False)
        assert len(p) == 4
        with pytest.raises(ValueError, match="constant"):
            pp.summarise(p)

    def test_rejects_an_empty_window(self):
        with pytest.raises(ValueError, match="empty window"):
            pp.stack({}, start=2010, end=2000)


class TestCorrelationMatrix:
    def test_identical_rows_give_one(self):
        x = np.array([[1.0, 2, 3, 4], [1.0, 2, 3, 4]])
        r = pp.correlation_matrix(x)
        assert r[0, 1] == pytest.approx(1.0)

    def test_reversed_row_gives_minus_one(self):
        x = np.array([[1.0, 2, 3, 4], [4.0, 3, 2, 1]])
        assert pp.correlation_matrix(x)[0, 1] == pytest.approx(-1.0)

    def test_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(12, 40))
        assert pp.correlation_matrix(x) == pytest.approx(np.corrcoef(x))

    def test_a_constant_row_raises_instead_of_returning_nan(self):
        x = np.array([[1.0, 1, 1, 1], [1.0, 2, 3, 4]])
        with pytest.raises(ValueError, match="constant"):
            pp.correlation_matrix(x)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(10)
        r = pp.correlation_matrix(rng.normal(size=(6, 30)))
        assert np.diag(r) == pytest.approx(np.ones(6))


class TestPairGroups:
    def test_the_three_groups_partition_the_upper_triangle(self, panel):
        g = pp.pair_groups(panel)
        n = len(panel)
        total = n * (n - 1) // 2
        covered = (g["unrelated"].astype(int)
                   + g["same country, different indicator"].astype(int)
                   + g["same indicator, different country"].astype(int))
        # every pair falls in exactly one group: same country AND same
        # indicator is impossible, because that would be a series with itself
        assert covered.sum() == total
        assert set(np.unique(covered)) == {1}

    def test_counts_are_what_combinatorics_says(self, panel):
        g = pp.pair_groups(panel)
        n_ind, n_ctry = 3, 4
        assert g["same country, different indicator"].sum() == \
            n_ctry * n_ind * (n_ind - 1) // 2
        assert g["same indicator, different country"].sum() == \
            n_ind * n_ctry * (n_ctry - 1) // 2
        assert g["unrelated"].sum() == \
            n_ind * (n_ind - 1) // 2 * n_ctry * (n_ctry - 1)


class TestSummarise:
    def test_reports_every_group(self, panel):
        out = pp.summarise(panel)
        assert set(out) >= {"unrelated", "same country, different indicator",
                            "same indicator, different country", "_meta"}
        assert out["_meta"]["n_series"] == 12

    def test_differencing_shortens_the_window_by_one(self, panel):
        assert pp.summarise(panel, difference=True)["_meta"]["n_years"] == \
            panel.n_years - 1

    def test_trending_series_look_related_and_their_differences_do_not(self):
        """The control the post rests on, on data whose truth is known."""
        rng = np.random.default_rng(77)
        years = np.arange(1960, 2024)
        isos = [f"C{i:02d}" for i in range(12)]
        frames = {}
        for name in ("a", "b", "c"):
            frames[name] = _long(
                {iso: np.cumsum(rng.normal(0.4, 1.0, years.size))
                 for iso in isos}, years)
        p = pp.stack(frames, start=1960, end=2023, min_countries=10)

        levels = pp.summarise(p)["unrelated"]
        diffs = pp.summarise(p, difference=True)["unrelated"]
        assert levels["median_abs_r"] > 0.6
        assert diffs["median_abs_r"] < 0.15
        assert diffs["p_over_90"] < 0.01

    def test_independent_white_noise_is_near_zero_in_levels_too(self):
        """Without the unit root there is no artefact — the trend is the cause."""
        rng = np.random.default_rng(88)
        years = np.arange(1960, 2024)
        isos = [f"C{i:02d}" for i in range(12)]
        frames = {n: _long({iso: rng.normal(size=years.size) for iso in isos},
                           years)
                  for n in ("a", "b", "c")}
        p = pp.stack(frames, start=1960, end=2023, min_countries=10)
        assert pp.summarise(p)["unrelated"]["median_abs_r"] < 0.15


class TestExtremes:
    def test_returns_labelled_pairs_in_descending_order(self, panel):
        top = pp.extremes(panel, top=5)
        assert len(top) == 5
        assert [t["abs_r"] for t in top] == sorted(
            [t["abs_r"] for t in top], reverse=True)
        for t in top:
            assert t["a_country"] != t["b_country"]
            assert t["a_indicator"] != t["b_indicator"]

    def test_signed_and_absolute_agree(self, panel):
        for t in pp.extremes(panel, top=8):
            assert abs(t["signed_r"]) == pytest.approx(t["abs_r"])
