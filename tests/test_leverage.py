"""Tests for the leverage module behind episode six."""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.linalg import leverage as lv


class TestTheIdentity:
    @pytest.mark.parametrize("n, p", [(50, 3), (200, 7), (1000, 4)])
    def test_the_leverages_sum_to_p_whatever_the_data(self, n, p):
        rng = np.random.default_rng(n)
        for _ in range(5):
            X = rng.standard_normal((n, p))
            X[:, 0] = 1.0
            assert lv.hat_diagonal(X).sum() == pytest.approx(p)

    def test_it_survives_an_extreme_row(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(200), rng.standard_normal(200)])
        X[0, 1] = 1e6
        assert lv.hat_diagonal(X).sum() == pytest.approx(2.0)
        assert lv.hat_diagonal(X)[0] > 0.99, "and one row can take almost all of it"

    def test_the_report_gives_p_over_n_as_the_mean(self):
        rng = np.random.default_rng(1)
        rep = lv.leverage_report(lv.rare_dummy_design(1000, 5, rng=rng))
        assert rep.mean == pytest.approx(4 / 1000)
        assert rep.h.mean() == pytest.approx(rep.mean)

    def test_hat_diagonal_matches_the_explicit_projection(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 4))
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        assert lv.hat_diagonal(X) == pytest.approx(np.diag(H))


class TestTheRareDummy:
    """Episode two set this aside as 'not a conditioning problem'. It is this one."""

    @pytest.mark.parametrize("k", [2, 3, 5, 10, 50, 200])
    def test_leverage_of_a_category_with_k_members_is_one_over_k(self, k):
        """`1/k` is a floor, not the value. The intercept and the dummy together
        contribute exactly `1/k` to one of those rows; the *other* `p - 2`
        columns add about `(p - 2)/n` on top. At k=2 that correction is a fifth
        of a percent and at k=200 it is a sixth of the total, which is why the
        floor is the part worth remembering."""
        rng = np.random.default_rng(10 + k)
        X = lv.rare_dummy_design(2000, k, rng=rng)
        n, p = X.shape
        rows = np.flatnonzero(X[:, -1] == 1.0)
        h = lv.hat_diagonal(X)[rows].mean()
        assert h >= 1.0 / k
        assert h == pytest.approx(1.0 / k + (p - 2) / n, rel=0.02)

    def test_a_category_of_one_has_leverage_exactly_one(self):
        rng = np.random.default_rng(3)
        X = lv.rare_dummy_design(1000, 1, rng=rng)
        rep = lv.leverage_report(X)
        assert rep.h.max() == pytest.approx(1.0, abs=1e-12)
        assert len(rep.saturated) == 1
        assert rep.max_ratio == pytest.approx(1000 / 4)

    def test_and_therefore_a_residual_of_exactly_zero(self):
        rng = np.random.default_rng(4)
        X = lv.rare_dummy_design(1000, 1, rng=rng)
        y = X @ np.array([1.0, 2.0, -1.0, 3.0]) + rng.normal(0, 1.0, 1000)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        i = int(lv.leverage_report(X).saturated[0])
        assert abs(y[i] - X[i] @ beta) < 1e-9

    def test_the_influence_measures_report_nothing_there(self):
        rng = np.random.default_rng(5)
        X = lv.rare_dummy_design(1000, 1, rng=rng)
        y = X @ np.array([1.0, 2.0, -1.0, 3.0]) + rng.normal(0, 1.0, 1000)
        i = int(lv.leverage_report(X).saturated[0])
        assert np.isnan(lv.cook_distance(X, y)[i]), (
            "undefined, not zero -- and zero is what an implementation that "
            "divides two underflows will print")
        assert np.isnan(lv.dfbeta(X, y)[i]).all()

    def test_deleting_it_removes_the_coefficient_rather_than_moving_it(self):
        rng = np.random.default_rng(6)
        X = lv.rare_dummy_design(1000, 1, rng=rng)
        y = X @ np.array([1.0, 2.0, -1.0, 3.0]) + rng.normal(0, 1.0, 1000)
        d = lv.deletion_refit(X, y, int(lv.leverage_report(X).saturated[0]))
        assert d["lost_rank"] and d["rank_after"] == d["rank_before"] - 1

    def test_the_sweep_traces_one_over_k(self):
        rng = np.random.default_rng(7)
        for row in lv.leverage_sweep((2, 4, 8, 16), n=2000, rng=rng):
            assert row["leverage"] == pytest.approx(row["one_over_k"], rel=0.05)


class TestLeverageIsNotInfluence:
    """The distinction the episode is built on, and the one practitioners
    collapse: leverage says a row *could* matter, not that it does."""

    def _design(self, seed=103, n=500, sigma=1.6):
        rng = np.random.default_rng(seed)
        x = rng.lognormal(0.0, sigma, n)
        X = np.column_stack([np.ones(n), x, rng.standard_normal(n)])
        beta = np.array([1.0, 0.5, -0.3])
        return X, X @ beta + rng.normal(0, 1.0, n), beta, rng

    def test_a_high_leverage_row_can_move_nothing(self):
        X, y, _, _ = self._design()
        h = lv.hat_diagonal(X)
        i = int(np.argmax(h))
        assert h[i] > 20 * (X.shape[1] / len(y)), "far out in x"
        assert abs(lv.dfbeta(X, y)[i, 1]) < 1.0, "and barely moves the slope"

    def test_the_same_error_costs_far_more_at_high_leverage(self):
        X, y, beta, _ = self._design()
        h = lv.hat_diagonal(X)
        hi = int(np.argmax(h))
        mid = int(np.argsort(h)[len(h) // 2])
        moved = []
        for i in (hi, mid):
            yy = y.copy()
            yy[i] = X[i] @ beta + 3.0
            moved.append(abs(lv.dfbeta(X, yy)[i, 1]))
        assert moved[0] > 100 * moved[1]

    def test_both_together_move_a_coefficient_by_several_standard_errors(self):
        """The syllabus promise, with no outlier placed by hand: a lognormal
        column and ordinary noise are enough."""
        X, y, _, _ = self._design(seed=201, sigma=2.4)
        h = lv.hat_diagonal(X)
        i = int(np.argmax(h))
        assert h[i] > 0.9
        assert abs(lv.dfbeta(X, y)[i, 1]) > 3.0


class TestClosedFormAgreesWithRefitting:
    def test_dfbeta_matches_an_actual_deletion(self):
        rng = np.random.default_rng(8)
        X = np.column_stack([np.ones(300), rng.standard_normal((300, 3))])
        y = rng.standard_normal(300)
        db = lv.dfbeta(X, y)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X))
                     * float(((y - X @ beta) ** 2).sum()) / (300 - 4))
        for row in (0, 17, 199):
            d = lv.deletion_refit(X, y, row)
            assert -d["change"] / se == pytest.approx(db[row], rel=1e-6, abs=1e-9)
