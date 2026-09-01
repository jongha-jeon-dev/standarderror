"""Tests for the spectral module behind episode four.

The claims under test are the ones the episode makes: a tied eigenspace has no
distinguished axes, a subspace angle does not depend on the basis chosen for it,
and eigenvector movement is governed by the gap rather than by the share of
variance. Each is checked against a construction whose spectrum is known in
closed form, so a failure is a failure of the code and not of the example.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.linalg import spectral as sp


class TestConstructions:
    def test_equicorrelation_has_the_closed_form_spectrum(self):
        p, rho = 6, 0.35
        s = sp.spectrum(sp.equicorrelation(p, rho))
        assert s.values[0] == pytest.approx(1 + (p - 1) * rho)
        assert s.values[1:] == pytest.approx(np.full(p - 1, 1 - rho))

    def test_the_repeated_eigenvalue_is_an_exact_tie(self):
        s = sp.spectrum(sp.equicorrelation(5, 0.4))
        assert s.ties == [(1, 2), (2, 3), (3, 4)], (
            "four equal eigenvalues are three adjacent ties")
        assert max(s.gaps[1:]) < sp.TIE_TOLERANCE

    def test_block_pairs_puts_the_gap_where_it_was_asked_for(self):
        s = sp.spectrum(sp.block_pairs([0.80, 0.78, 0.30]))
        assert s.values == pytest.approx([1.80, 1.78, 1.30, 0.70, 0.22, 0.20])
        # The point of the construction: the gap is a difference of parameters.
        assert s.gaps[0] == pytest.approx(0.02)

    @pytest.mark.parametrize("p, rho", [(3, -0.6), (4, 1.0), (1, 0.5)])
    def test_an_impossible_equicorrelation_is_refused(self, p, rho):
        with pytest.raises(ValueError):
            sp.equicorrelation(p, rho)

    def test_block_pairs_refuses_a_correlation_outside_the_open_interval(self):
        with pytest.raises(ValueError, match="within-pair correlation"):
            sp.block_pairs([0.5, 1.0])


class TestATiedEigenspaceHasNoAxes:
    @pytest.mark.parametrize("degrees", [1.0, 17.0, 45.0, 90.0, 123.4])
    def test_rotating_tied_axes_reconstructs_the_same_matrix(self, degrees):
        E = sp.equicorrelation(5, 0.4)
        s = sp.spectrum(E)
        V = sp.rotate_within(s.vectors, 1, 2, degrees)
        assert V @ np.diag(s.values) @ V.T == pytest.approx(E, abs=1e-12)

    def test_rotating_untied_axes_does_not(self):
        """The contrast that gives the previous test its meaning: if any rotation
        were harmless the demonstration would be about rotation, not about ties."""
        C = sp.block_pairs([0.80, 0.60, 0.30])
        s = sp.spectrum(C)
        V = sp.rotate_within(s.vectors, 0, 1, 30.0)
        rebuilt = V @ np.diag(s.values) @ V.T
        assert np.abs(rebuilt - C).max() > 0.01

    def test_the_rotated_basis_is_still_orthonormal(self):
        s = sp.spectrum(sp.equicorrelation(5, 0.4))
        V = sp.rotate_within(s.vectors, 1, 3, 37.0)
        assert V.T @ V == pytest.approx(np.eye(5), abs=1e-12)


class TestSubspaceAngle:
    def test_it_does_not_depend_on_the_basis_chosen_for_the_subspace(self):
        """The invariance the individual eigenvectors lack, and the reason a
        near-tied plane can be reported when neither axis in it can be."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 2))
        B = rng.standard_normal((6, 2))
        base = sp.principal_angles(A, B)
        for _ in range(20):
            t = rng.uniform(0, 360)
            c, s_ = np.cos(np.radians(t)), np.sin(np.radians(t))
            spun = A @ np.array([[c, -s_], [s_, c]])
            assert sp.principal_angles(spun, B) == pytest.approx(base, abs=1e-6)

    def test_a_subspace_against_itself_is_zero(self):
        # Tolerance in micro-degrees, not nano-degrees, and the reason is this
        # series' own subject: arccos is ill-conditioned at 1. A singular value
        # of 1 - 1e-16 comes back as an angle near sqrt(2e-16) radians, so the
        # error in the *angle* is the square root of the error in the cosine.
        rng = np.random.default_rng(1)
        A = rng.standard_normal((5, 2))
        assert sp.principal_angles(A, A) == pytest.approx([0.0, 0.0], abs=1e-5)

    def test_orthogonal_subspaces_are_ninety_degrees(self):
        A = np.eye(4)[:, :2]
        B = np.eye(4)[:, 2:]
        assert sp.principal_angles(A, B) == pytest.approx([90.0, 90.0], abs=1e-9)


class TestSignConvention:
    def test_align_signs_removes_a_flip(self):
        rng = np.random.default_rng(2)
        V = np.linalg.qr(rng.standard_normal((5, 5)))[0]
        flipped = V * np.array([1, -1, 1, -1, 1])
        assert sp.align_signs(flipped, V) == pytest.approx(V)

    def test_vector_angle_reads_a_flip_as_zero_not_as_disagreement(self):
        v = np.array([1.0, 2.0, 3.0])
        assert sp.vector_angle(v, -v) == pytest.approx(0.0, abs=1e-9)
        assert sp.vector_angle(v, v) == pytest.approx(0.0, abs=1e-9)


class TestDavisKahan:
    def test_a_vanishing_gap_buys_no_information(self):
        assert sp.davis_kahan_bound(0.1, 0.0) == 90.0
        assert sp.davis_kahan_bound(1.0, 0.001) == 90.0

    def test_the_bound_tightens_as_the_gap_opens(self):
        wide = sp.davis_kahan_bound(0.05, 1.0)
        narrow = sp.davis_kahan_bound(0.05, 0.2)
        assert wide < narrow < 90.0

    def test_it_is_never_violated_on_a_well_separated_component(self):
        """200 fresh samples, and not one eigenvector moves further than the
        theorem allows. A bound that is never attained here is still a bound:
        the episode says it is loose, which is a different claim from wrong."""
        rng = np.random.default_rng(3)
        C = sp.block_pairs([0.80, 0.50, 0.20])
        truth = sp.spectrum(C)
        k, gap = 2, truth.neighbour_gap(2)
        violations = 0
        for _ in range(200):
            R = sp.sample_correlation(300, C, rng=rng)
            est = sp.spectrum(R)
            angle = sp.vector_angle(est.vectors[:, k], truth.vectors[:, k])
            if angle > sp.davis_kahan_bound(np.linalg.norm(R - C, 2), gap) + 1e-9:
                violations += 1
        assert violations == 0


class TestTheGapGovernsTheAxes:
    def test_a_smaller_gap_moves_the_axis_further(self):
        rng = np.random.default_rng(5)
        wide = sp.bootstrap_components(sp.block_pairs([0.80, 0.30, 0.10]),
                                       n=300, reps=60, rng=rng)
        narrow = sp.bootstrap_components(sp.block_pairs([0.80, 0.78, 0.10]),
                                         n=300, reps=60, rng=rng)
        # A 25-fold narrower gap moves the axis about 4 times further. The
        # assertion is 3x rather than the measured 4x so that it tests the
        # direction of the effect without pinning a simulated constant.
        assert narrow["median_angle"][0] > 3 * wide["median_angle"][0]
        assert narrow["swap_rate"][0] > wide["swap_rate"][0]

    def test_the_share_of_variance_does_not_order_the_stability(self):
        """The claim of the episode, as an assertion rather than a figure: the
        component carrying the most variance is the one that moves most."""
        rng = np.random.default_rng(6)
        res = sp.bootstrap_components(sp.block_pairs([0.80, 0.78, 0.30]),
                                      n=400, reps=120, rng=rng)
        rows = sp.stability_table(res)
        biggest = max(rows, key=lambda r: r["variance_share"])
        steadiest = min(rows, key=lambda r: r["median_angle"])
        assert biggest["component"] == 1
        assert steadiest["component"] != 1
        assert biggest["median_angle"] > 3 * steadiest["median_angle"]

    def test_the_plane_holds_still_while_its_axes_do_not(self):
        rng = np.random.default_rng(7)
        res = sp.bootstrap_components(sp.block_pairs([0.80, 0.78, 0.30]),
                                      n=400, reps=120, rng=rng)
        assert res["median_angle"][0] > 3 * res["median_plane_angle"][0]

    def test_the_table_pairs_the_gap_and_the_swap_with_the_same_neighbour(self):
        """A regression: the swap rate is indexed by adjacent *pair* and the gap
        by component, and reporting `swap_rate[k]` beside `neighbour_gap(k)` put
        a component's tightest gap next to a different pair's swap rate — which
        read as a component squeezed hard that never swaps."""
        rng = np.random.default_rng(8)
        res = sp.bootstrap_components(sp.block_pairs([0.80, 0.78, 0.30]),
                                      n=300, reps=40, rng=rng)
        rows = {r["component"]: r for r in sp.stability_table(res)}
        assert rows[1]["neighbour"] == 2 and rows[2]["neighbour"] == 1
        assert rows[1]["swap_rate"] == rows[2]["swap_rate"], (
            "one pair, one swap rate, seen from either side")
        assert rows[3]["neighbour"] == 2, "0.48 above beats 0.60 below"


class TestRowBootstrap:
    def test_it_understates_how_far_the_axis_actually_is(self):
        """The episode's third finding, as an assertion. The bootstrap centres on
        the sample's own axis, which under a near-tie is one arbitrary direction
        in the shared plane, so it measures scatter around a point that is itself
        wrong -- and reports a number several times too small."""
        C = sp.block_pairs([0.80, 0.78, 0.30])
        truth = sp.spectrum(C)
        reported, actual = [], []
        for seed in range(8):
            rng = np.random.default_rng(700 + seed)
            X = rng.standard_normal((400, 6)) @ np.linalg.cholesky(C).T
            reported.append(sp.row_bootstrap(X, reps=80, rng=rng)["median_angle"][0])
            actual.append(sp.vector_angle(sp.spectrum(np.corrcoef(X.T)).vectors[:, 0],
                                          truth.vectors[:, 0]))
        assert np.median(actual) > 2 * np.median(reported)

    def test_it_reports_the_sample_size_and_replicate_count(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((400, 6))
        rows = sp.row_bootstrap(X, reps=30, rng=rng)
        assert rows["n"] == 400 and rows["reps"] == 30

    def test_a_well_separated_axis_barely_moves_under_resampling(self):
        rng = np.random.default_rng(10)
        C = sp.block_pairs([0.80, 0.40, 0.10])
        X = rng.standard_normal((600, 6)) @ np.linalg.cholesky(C).T
        res = sp.row_bootstrap(X, reps=100, rng=rng)
        assert res["median_angle"][0] < 12.0
        assert res["swap_rate"][0] == 0.0


class TestSpectrumObject:
    def test_gaps_are_positive_and_one_shorter_than_the_spectrum(self):
        s = sp.spectrum(sp.block_pairs([0.7, 0.4, 0.1]))
        assert len(s.gaps) == len(s.values) - 1
        assert (s.gaps >= 0).all()

    def test_variance_share_sums_to_one(self):
        s = sp.spectrum(sp.block_pairs([0.7, 0.4, 0.1]))
        assert s.variance_share.sum() == pytest.approx(1.0)

    def test_a_non_square_input_is_refused(self):
        with pytest.raises(ValueError, match="square"):
            sp.spectrum(np.ones((3, 4)))
