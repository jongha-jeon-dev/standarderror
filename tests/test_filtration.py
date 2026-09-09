"""What H0 of a Rips filtration is, and what it therefore inherits.

The load-bearing test is `test_it_is_single_linkage_to_the_last_bit`: if that
identity holds then every known property of single-linkage clustering is a
property of 0-dimensional persistent homology, including the one the rest of
this file is about.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from standarderror.topology import filtration as ft


class TestTheIdentity:
    @pytest.mark.parametrize("n,d,seed", [(60, 5, 0), (40, 2, 1), (25, 30, 2),
                                          (12, 1, 3)])
    def test_it_is_single_linkage_to_the_last_bit(self, n, d, seed):
        """Not "agrees closely". The same floats."""
        X = np.random.default_rng(seed).standard_normal((n, d))
        D = ft.pairwise(X)
        got = ft.rips_h0(D)
        assert np.array_equal(got.deaths, ft.single_linkage_heights(D))

    def test_n_points_give_n_minus_one_finite_bars(self):
        """One component survives forever and its bar is not stored."""
        X = np.random.default_rng(0).standard_normal((17, 3))
        bc = ft.rips_h0(ft.pairwise(X))
        assert len(bc.deaths) == 16
        assert bc.n_points == 17

    def test_every_death_is_an_actual_pairwise_distance(self):
        """Because a component dies exactly when some edge joins it to another,
        so a death time is that edge's length and nothing else."""
        X = np.random.default_rng(4).standard_normal((20, 4))
        D = ft.pairwise(X)
        bc = ft.rips_h0(D)
        available = set(np.round(pdist(X), 12))
        assert all(round(float(x), 12) in available for x in bc.deaths)

    def test_the_deaths_come_back_sorted(self):
        bc = ft.rips_h0(ft.pairwise(
            np.random.default_rng(5).standard_normal((30, 3))))
        assert np.all(np.diff(bc.deaths) >= 0)

    def test_alive_at_counts_components(self):
        X, _ = ft.two_blobs()
        bc = ft.rips_h0(ft.pairwise(X))
        assert bc.alive_at(0.0) == len(X)
        assert bc.alive_at(bc.longest + 1) == 1
        # Between the two longest bars there are exactly two components, which
        # is what the two-cluster reading of a barcode means.
        mid = 0.5 * (bc.deaths[-1] + bc.deaths[-2])
        assert bc.alive_at(mid) == 2

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(ValueError, match="square distance matrix"):
            ft.rips_h0(np.zeros((3, 4)))

    def test_two_identical_points_die_at_zero(self):
        X = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
        bc = ft.rips_h0(ft.pairwise(X))
        assert bc.deaths[0] == 0.0


class TestWhatItInherits:
    """Single linkage's chaining, stated as a property of the barcode."""

    @pytest.fixture(scope="class")
    def sweep(self):
        return {r["bridge"]: r for r in ft.chaining_sweep()}

    def test_two_blobs_alone_claim_two_clusters_loudly(self, sweep):
        r = sweep[0]
        assert r["separation_ratio"] > 6.0
        assert r["cut_sizes"] == [30, 30]

    def test_three_bridge_points_erase_the_claim(self, sweep):
        """Three points out of sixty-three, adding no cluster of their own, and
        the two longest bars become the same length to three decimals."""
        r = sweep[3]
        assert r["separation_ratio"] == pytest.approx(1.0, abs=0.02)
        assert r["longest"] < 2.0 < sweep[0]["longest"]

    def test_twelve_bridge_points_make_the_cut_return_a_singleton(self, sweep):
        """The k = 2 cut stops separating the blobs and starts shaving off one
        point, which is the failure mode dressed as an answer."""
        assert sweep[12]["cut_sizes"] == [71, 1]
        assert sweep[24]["cut_sizes"][1] == 1

    def test_the_bridge_reduces_the_longest_bar_to_within_blob_spacing(self, sweep):
        """At twelve points the longest bar equals the *second* bar of the
        bridge-free cloud: nothing is left but the spacing inside a blob."""
        assert sweep[12]["longest"] == pytest.approx(sweep[0]["second"],
                                                     rel=1e-9)

    def test_the_ratio_is_monotone_in_the_bridge(self, sweep):
        ratios = [sweep[k]["separation_ratio"] for k in (0, 3, 6)]
        assert ratios[0] > 5.0 > ratios[1]


class TestTheStabilityTheorem:
    @pytest.fixture(scope="class")
    def sweep(self):
        X, _ = ft.two_blobs(per=40)
        return ft.stability_sweep(X)

    def test_the_bound_holds_at_every_perturbation(self, sweep):
        for row in sweep:
            assert row["bound_holds"], row

    def test_and_with_room_to_spare(self, sweep):
        """At eps = 0.5 the points move 1.39 and the barcode 0.44."""
        big = [r for r in sweep if r["epsilon"] == 0.5][0]
        assert big["bottleneck"] < 0.5 * big["hausdorff"]

    def test_an_unperturbed_copy_has_bottleneck_zero(self, sweep):
        assert sweep[0]["bottleneck"] == 0.0

    def test_the_bottleneck_saturates_while_the_points_keep_moving(self, sweep):
        """Past eps = 0.5 the barcode cannot move much further -- it is pinned
        against the cloud's own diameter -- so the bound stops being the
        interesting constraint and `TestTheGapRuleAcrossDraws` takes over."""
        by = {r["epsilon"]: r for r in sweep}
        assert by[1.0]["hausdorff"] > 2 * by[0.5]["hausdorff"]
        assert by[1.0]["bottleneck"] < 5 * by[0.5]["bottleneck"]

    def test_hausdorff_wants_a_point_for_point_copy(self):
        with pytest.raises(ValueError, match="point-for-point"):
            ft.hausdorff(np.zeros((4, 2)), np.zeros((5, 2)))


class TestTheBottleneckDistanceItself:
    """Written here, so it needs checking against something that is not it."""

    def _brute(self, d1, d2):
        """Every permutation, for barcodes small enough to enumerate."""
        assert len(d1) == len(d2)
        return min(max(abs(d1[i] - d2[p[i]]) for i in range(len(d1)))
                   for p in itertools.permutations(range(len(d2))))

    @pytest.mark.parametrize("seed", range(6))
    def test_it_matches_a_brute_force_matching(self, seed):
        rng = np.random.default_rng(seed)
        # Deaths well away from zero, so sending a bar to the diagonal is never
        # the cheap option and the brute force above is the right reference.
        d1 = np.sort(rng.uniform(5.0, 6.0, 5))
        d2 = np.sort(rng.uniform(5.0, 6.0, 5))
        got = ft.bottleneck_h0(ft.Barcode(d1), ft.Barcode(d2))
        assert got == pytest.approx(self._brute(d1, d2), rel=1e-9)

    def test_a_barcode_is_zero_from_itself(self):
        d = np.sort(np.random.default_rng(0).uniform(1, 4, 9))
        assert ft.bottleneck_h0(ft.Barcode(d), ft.Barcode(d)) == 0.0

    def test_an_extra_short_bar_costs_half_its_length(self):
        """Because the cheapest thing to do with an unmatched bar is send it to
        the diagonal, at distance death/2."""
        d = np.array([3.0, 3.5, 4.0])
        extra = np.array([0.02, 3.0, 3.5, 4.0])
        assert ft.bottleneck_h0(ft.Barcode(d), ft.Barcode(extra)) == \
            pytest.approx(0.01, rel=1e-9)

    def test_it_is_symmetric(self):
        rng = np.random.default_rng(3)
        a = ft.Barcode(np.sort(rng.uniform(1, 5, 7)))
        b = ft.Barcode(np.sort(rng.uniform(1, 5, 7)))
        assert ft.bottleneck_h0(a, b) == pytest.approx(ft.bottleneck_h0(b, a))


class TestTheDisplayRatherThanTheMethod:
    @pytest.fixture(scope="class")
    def dims(self):
        return {r["d"]: r for r in ft.dimension_sweep()}

    def test_the_barcodes_dynamic_range_collapses_with_dimension(self, dims):
        """7.16 at d = 2 against 0.083 at d = 768: every bar in a 768-dimensional
        barcode is born and dies within a few percent of one radius."""
        assert dims[2]["barcode_spread"] > 6.0
        assert dims[768]["barcode_spread"] < 0.12
        spreads = [dims[d]["barcode_spread"] for d in (2, 8, 64, 768)]
        assert spreads == sorted(spreads, reverse=True)

    def test_the_pairwise_distances_concentrate_too(self, dims):
        assert dims[2]["distance_spread"] > 3.5
        assert dims[768]["distance_spread"] < 0.3

    def test_a_barcode_with_one_bar_has_no_separation_ratio(self):
        assert np.isnan(ft.Barcode(np.array([1.0])).separation_ratio)
        assert ft.Barcode(np.array([1.0])).gap_k() == 1


class TestTheNormalisations:
    def test_centring_removes_the_mean(self):
        X = np.random.default_rng(0).standard_normal((20, 5)) + 7.0
        assert np.abs(ft.normalise(X, "centred").mean(0)).max() < 1e-12

    def test_l2_puts_everything_on_the_sphere(self):
        X = np.random.default_rng(0).standard_normal((20, 5)) + 7.0
        n = np.linalg.norm(ft.normalise(X, "l2"), axis=1)
        assert np.abs(n - 1.0).max() < 1e-12

    def test_whitening_gives_unit_variance_per_coordinate(self):
        X = np.random.default_rng(0).standard_normal((50, 4)) * [1, 10, 100, 1e3]
        assert np.abs(ft.normalise(X, "whitened").std(0) - 1.0).max() < 1e-12

    def test_a_zero_vector_survives_l2(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        got = ft.normalise(X, "l2")
        assert np.all(np.isfinite(got))
        assert got[1] == pytest.approx([0.6, 0.8])

    def test_cosine_is_refused_as_a_normalisation(self):
        with pytest.raises(ValueError, match="cosine is a metric"):
            ft.normalise(np.zeros((3, 2)), "cosine")

    def test_an_unknown_normalisation_is_refused(self):
        with pytest.raises(ValueError, match="unknown normalisation"):
            ft.normalise(np.zeros((3, 2)), "softmax")

    def test_the_narrow_cone_collapses_cosine_similarity(self):
        """Episode 3's design, checked here because episode 1 introduces it: a
        mean offset drives every pairwise cosine similarity close to 1."""
        X, _ = ft.narrow_cone(cone=6.0)
        sims = 1.0 - pdist(X, metric="cosine")
        assert sims.mean() > 0.97
        assert sims.max() - sims.min() < 0.15
        flat, _ = ft.narrow_cone(cone=0.0)
        base = 1.0 - pdist(flat, metric="cosine")
        assert base.max() - base.min() > 1.5


class TestTheGapRuleAcrossDraws:
    """One perturbation is an anecdote. Forty is the claim."""

    @pytest.fixture(scope="class")
    def rows(self):
        X, _ = ft.two_blobs(per=40)
        return {e: ft.gap_rule_spread(X, e) for e in (0.0, 0.2, 1.0, 1.5)}

    def test_an_unperturbed_cloud_always_says_two(self, rows):
        assert rows[0.0]["counts"] == {2: 40}
        assert rows[0.0]["bottleneck_max"] == 0.0

    def test_a_small_perturbation_still_always_says_two(self, rows):
        assert rows[0.2]["counts"] == {2: 40}

    def test_at_epsilon_one_it_disagrees_with_itself_in_half_the_draws(self, rows):
        r = rows[1.0]
        assert r["share_modal"] < 0.6
        assert r["k_min"] == 2 and r["k_max"] >= 5

    def test_the_integer_keeps_moving_after_the_barcode_stops(self, rows):
        """The point of the episode. Between eps 1.0 and 1.5 the largest
        bottleneck distance does not grow at all -- the barcode is pinned
        against the diameter of the cloud -- and the answer runs to 11."""
        assert rows[1.5]["bottleneck_max"] == pytest.approx(
            rows[1.0]["bottleneck_max"], rel=1e-9)
        assert rows[1.5]["k_max"] > rows[1.0]["k_max"]
        assert rows[1.5]["k_max"] >= 10

    def test_a_sweep_row_does_not_depend_on_the_other_rows(self):
        """The first version of `stability_sweep` advanced one generator through
        the loop, so a three-row sweep and a six-row sweep disagreed at the same
        epsilon and the article's snippet contradicted its own figure."""
        X, _ = ft.two_blobs(per=40)
        full = {r["epsilon"]: r for r in ft.stability_sweep(X)}
        part = {r["epsilon"]: r for r in ft.stability_sweep(X, (0.0, 0.5, 1.0))}
        for eps in (0.0, 0.5, 1.0):
            assert full[eps]["hausdorff"] == pytest.approx(part[eps]["hausdorff"])
            assert full[eps]["bottleneck"] == pytest.approx(part[eps]["bottleneck"])
            assert full[eps]["gap_k"] == part[eps]["gap_k"]
