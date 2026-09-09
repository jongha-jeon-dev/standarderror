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
        """Median over 15 draws: 4.87 at d = 2 against 0.078 at d = 768, so
        every bar in a 768-dimensional barcode is born and dies within a few
        percent of one radius."""
        assert dims[2]["barcode_spread"] > 4.0
        assert dims[768]["barcode_spread"] < 0.10
        # And the reason it is a median: at d = 2 a single draw lands anywhere
        # between 3.8 and 6.7, which is why the first version of this function
        # made a snippet disagree with a figure.
        assert dims[2]["barcode_spread_max"] / dims[2]["barcode_spread_min"] > 1.5
        assert dims[768]["barcode_spread_max"] < 0.15
        spreads = [dims[d]["barcode_spread"] for d in (2, 8, 64, 768)]
        assert spreads == sorted(spreads, reverse=True)

    def test_the_pairwise_distances_concentrate_too(self, dims):
        assert dims[2]["distance_spread"] > 3.0
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


class TestWhatTheBarcodeSaysAboutNoise:
    """Episode 2's payoff, and it corrected the episode's own plan."""

    @pytest.fixture(scope="class")
    def noise(self):
        return {d: ft.gap_rule_on_noise(d) for d in (2, 64, 768)}

    def test_the_gap_rule_never_says_one_cluster(self, noise):
        """The structural defect, and the same one the scree-plot episode found
        in the elbow: the rule locates the biggest jump in a list, and a list of
        pure noise has a biggest jump."""
        for d, r in noise.items():
            assert r["said_one"] == 0, (d, r["counts"])

    def test_its_failure_changes_shape_with_dimension(self, noise):
        """At d = 2 it claims two clusters; in high dimensions it claims almost
        as many clusters as there are points, because the largest gap in a
        concentrated barcode is the first one."""
        assert noise[2]["modal_k"] == 2
        assert noise[768]["modal_k"] == noise[768]["n"] - 1

    def test_the_null_tightens_by_two_orders_of_magnitude(self, noise):
        """Which is the fact the rest of the episode turns on."""
        spread2 = noise[2]["ratio_max"] - noise[2]["ratio_min"]
        spread768 = noise[768]["ratio_max"] - noise[768]["ratio_min"]
        assert spread2 / spread768 > 30


class TestTheRatioAsADetector:
    @pytest.fixture(scope="class")
    def rows(self):
        return {d: ft.separation_ratio_auc(d) for d in (2, 64, 768)}

    def test_in_two_dimensions_it_is_worse_than_a_coin(self, rows):
        """Noise has a *higher* median ratio than a cloud whose clusters single
        linkage recovers. So the absolute reading is not weak, it is inverted."""
        r = rows[2]
        assert r["auc"] < 0.5
        assert r["noise_median"] > r["signal_median"]
        assert r["purity"] > 0.95

    def test_in_high_dimensions_it_becomes_usable(self, rows):
        """The opposite of what "the barcode loses its dynamic range" suggests,
        because that sentence is about the absolute reading. Concentration
        tightens the null faster than it shrinks the signal."""
        assert rows[768]["auc"] > 0.75
        assert rows[768]["auc"] > rows[64]["auc"] > rows[2]["auc"]

    def test_the_method_itself_works_at_every_dimension(self, rows):
        """Which is the distinction the episode exists to draw: the clustering
        is fine and the summary is what fails."""
        for d, r in rows.items():
            assert r["purity"] > 0.95, (d, r["purity"])

    def test_no_absolute_threshold_survives_the_dimension_change(self, rows):
        """A cutoff tuned at d = 2 would have to sit near 1.2; at d = 768 every
        signal cloud is below 1.06."""
        assert rows[2]["noise_median"] > 1.15
        assert rows[768]["signal_median"] < 1.06


class TestTheConcentrationAlgebra:
    """The derivation the episode rests on, checked against a measurement --
    which is how the first version of it was caught predicting sqrt(2)."""

    @pytest.fixture(scope="class")
    def rows(self):
        return {d: ft.concentration_check(d) for d in (2, 32, 768)}

    def test_the_mean_grows_like_the_root_of_twice_the_dimension(self, rows):
        for d, r in rows.items():
            assert r["mean"] == pytest.approx(r["mean_predicted"], rel=0.12), d
        assert rows[768]["mean"] == pytest.approx(rows[768]["mean_predicted"],
                                                 rel=0.01)

    def test_the_absolute_spread_tends_to_one_and_not_to_root_two(self, rows):
        assert rows[768]["sd"] == pytest.approx(1.0, rel=0.05)
        assert abs(rows[768]["sd"] - np.sqrt(2.0)) > 0.3

    def test_so_the_relative_spread_falls_like_one_over_the_root(self, rows):
        for d, r in rows.items():
            assert r["relative"] == pytest.approx(r["relative_predicted"],
                                                 rel=0.5), d
        assert rows[768]["relative"] == pytest.approx(
            rows[768]["relative_predicted"], rel=0.05)


class TestWhereTheLargestGapFalls:
    @pytest.fixture(scope="class")
    def rows(self):
        return {d: ft.largest_gap_position(d) for d in (2, 32, 768)}

    def test_at_two_dimensions_it_is_among_the_long_bars(self, rows):
        """Which is why the gap rule reports a small cluster count there."""
        assert rows[2]["median_position"] > 0.9 * rows[2]["gaps"]
        assert rows[2]["at_front"] == 0

    def test_in_high_dimensions_it_becomes_bimodal(self, rows):
        """Not "migrates to the front", which is what I first wrote. Either end
        or nothing in between, which is why the gap rule answers 199 or 2 with
        no drift between them."""
        for d in (32, 768):
            r = rows[d]
            assert r["median_position"] < 10
            assert 0 < r["at_front"] < r["draws"]
            # 0 of 30 at d = 32 and 2 of 30 at d = 768 land anywhere but the
            # two ends, which is what "bimodal" has to mean to be worth saying.
            middle = r["draws"] - r["in_first_ten"] - r["in_last_ten"]
            assert middle <= 0.1 * r["draws"], (d, middle)
        assert rows[2]["in_last_ten"] == rows[2]["draws"]

    def test_the_skew_of_the_deaths_is_what_the_position_follows(self):
        """The mechanism, and it is checkable. Neighbouring order statistics are
        spaced like 1 / (n f(x)), so the largest gap sits in the thinnest tail.
        One thin tail while the deaths are right-skewed; two once concentration
        has symmetrised them, and then the draw decides. Measured skews: 2.18,
        0.66, 0.14, 0.09 at d = 2, 8, 32, 768."""
        rows = {d: ft.largest_gap_position(d) for d in (2, 8, 32, 768)}
        skews = [rows[d]["death_skew"] for d in (2, 8, 32, 768)]
        assert skews == sorted(skews, reverse=True), skews
        assert skews[0] > 1.5 and skews[-1] < 0.2
        # While one tail is thin the gap is always in it; once both are, it is
        # not. That is the whole claim, and it is the same rows either way.
        assert rows[2]["in_first_ten"] == 0
        assert rows[768]["in_first_ten"] > rows[8]["in_first_ten"] > 0

    def test_the_positions_it_reports_are_the_positions_it_summarises(self):
        """The figure draws `positions` and the prose quotes the counts, so a
        disagreement between them would be a caption that contradicts its own
        picture -- which has happened in this series before."""
        r = ft.largest_gap_position(32)
        pos = r["positions"]
        assert len(pos) == r["draws"]
        assert sum(p == 0 for p in pos) == r["at_front"]
        assert sum(p < 10 for p in pos) == r["in_first_ten"]
