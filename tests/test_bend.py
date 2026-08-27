"""Tests for quantpost.ts.bend.

Two kinds of test carry the weight here.

The first kind checks the fast scan against something that is not itself: the
same quantities computed the slow way through `ols_hac`, and — for one
configuration — against statsmodels' own HAC covariance. The fast path uses a
Frisch-Waugh identity for both the coefficient and its long-run variance, and an
identity that is *nearly* right would be invisible in every result the module
produces. These tests are the only thing standing between that and a wrong post.

The second kind measures what the methods actually achieve rather than what
would be convenient. The model race between a bend and a step has poor power at
realistic signal sizes; the date interval for a bend is years wide. Both facts
are asserted at the values measured, so that an improvement breaks the test and
has to be looked at, and so that nothing downstream quietly assumes better.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantpost.ts import bend as bd
from quantpost.ts import detect as dt


def ar1(n, rho, rng, sigma=1.0):
    x = np.zeros(n)
    e = rng.normal(scale=sigma, size=n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def noisy_bend(n, tau, size, rng, sigma=0.2, rho=0.0):
    y = bd.break_column(n, tau, "bend") * size
    return y + (ar1(n, rho, rng, sigma) if rho else rng.normal(scale=sigma, size=n))


# ---------------------------------------------------------------------------
# the two columns
# ---------------------------------------------------------------------------

class TestBreakColumn:
    def test_step_is_zero_then_one(self):
        c = bd.break_column(10, 4, "step")
        assert c.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    def test_bend_is_zero_then_rising(self):
        c = bd.break_column(11, 5, "bend")
        assert np.allclose(c[:6], 0.0)
        d = np.diff(c[5:])
        assert np.allclose(d, d[0]) and d[0] > 0

    def test_bend_is_continuous_at_the_break(self):
        """The defining difference from a step: no jump."""
        c = bd.break_column(40, 20, "bend")
        assert c[19] == 0.0 and c[20] == 0.0
        assert abs(c[21] - c[20]) < 1.0

    def test_bend_reaches_one_over_the_sample_scale(self):
        c = bd.break_column(101, 1, "bend")
        assert c[-1] == pytest.approx(99 / 100)

    def test_bend_and_trend_are_on_the_same_scale(self):
        """Otherwise their coefficients differ by orders of magnitude."""
        X = bd.bend_design(200, tau=100, kind="bend", trend=True)
        assert X[:, 1].max() == pytest.approx(1.0)
        assert 0.4 < X[:, -1].max() < 0.6

    @pytest.mark.parametrize("tau", [0, 40])
    def test_break_outside_the_sample_is_refused(self, tau):
        with pytest.raises(ValueError, match="one side empty"):
            bd.break_column(40, tau, "step")

    def test_a_bend_with_one_post_observation_is_refused(self):
        with pytest.raises(ValueError, match="at least two"):
            bd.break_column(40, 39, "bend")

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="kind must be"):
            bd.break_column(40, 20, "kink")


class TestDesign:
    def test_break_column_is_last(self):
        X = bd.bend_design(60, tau=30, kind="step", trend=2, seasonal=12)
        assert np.allclose(X[:, -1], bd.break_column(60, 30, "step"))

    def test_no_tau_gives_the_base_design(self):
        a = bd.bend_design(60, trend=2, seasonal=12)
        b = dt.design_matrix(60, trend=2, seasonal=12)
        assert np.allclose(a, b)

    def test_slope_change_per_year_has_the_units_it_claims(self):
        """A series whose growth halves should report half the growth as change.

        Built by hand: 0.02 per month before the kink, 0.01 after. The annual
        growth change is therefore -0.12 log points, whatever the sample length.
        """
        n, tau = 241, 120
        t = np.arange(n, dtype=float)
        y = 0.02 * t - 0.01 * np.maximum(t - tau, 0.0)
        f = bd.fit_at(y, tau, kind="bend", trend=True, lags=0)
        assert bd.slope_change_per_year(f["coef"], n) == pytest.approx(-0.12, abs=1e-9)


# ---------------------------------------------------------------------------
# the fast scan, against answers that are not itself
# ---------------------------------------------------------------------------

class TestScanAgainstAnswersThatAreNotItself:
    CONFIGS = [
        ("step", 0, 1, 0), ("step", 0, 1, 5), ("step", 12, 2, 12),
        ("step", 4, 0, 3), ("bend", 0, 1, 0), ("bend", 0, 1, 5),
        ("bend", 12, 2, 12), ("bend", 4, 0, 3), ("bend", 12, 3, 24),
    ]

    @pytest.mark.parametrize("kind,seasonal,degree,lags", CONFIGS)
    def test_fast_scan_equals_the_slow_one(self, kind, seasonal, degree, lags):
        rng = np.random.default_rng(11)
        y = np.cumsum(rng.normal(size=200)) * 0.3 + np.arange(200) * 0.01
        kw = dict(kind=kind, trend=degree, seasonal=seasonal, lags=lags,
                  stride=7)
        fast, slow = bd.scan(y, **kw), bd.slow_scan(y, **kw)
        assert fast.dates.tolist() == slow.dates.tolist()
        for name in ("coef", "t_ols", "t_hac", "ssr"):
            a, b = getattr(fast, name), getattr(slow, name)
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12), name
        assert fast.ssr_null == pytest.approx(slow.ssr_null)

    def test_hac_t_matches_statsmodels_on_the_full_design(self):
        """The Frisch-Waugh long-run variance, checked outside this codebase."""
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(3)
        n, tau, lags = 180, 90, 12
        y = noisy_bend(n, tau, 3.0, rng, sigma=0.3, rho=0.7)
        X = bd.bend_design(n, tau=tau, kind="bend", trend=2, seasonal=12)
        ref = sm.OLS(y, X).fit(cov_type="HAC",
                               cov_kwds={"maxlags": lags, "use_correction": True})
        got = bd.scan(y, kind="bend", trend=2, seasonal=12, lags=lags).at(tau)
        assert got["coef"] == pytest.approx(ref.params[-1], rel=1e-10)
        assert got["t_hac"] == pytest.approx(ref.tvalues[-1], rel=1e-8)

    def test_ols_t_matches_the_textbook_formula(self):
        rng = np.random.default_rng(5)
        n, tau = 150, 70
        y = noisy_bend(n, tau, 2.0, rng)
        got = bd.scan(y, kind="bend", trend=True, lags=0).at(tau)
        ref = bd.fit_at(y, tau, kind="bend", trend=True, lags=0)
        assert got["t_ols"] == pytest.approx(ref["t_ols"], rel=1e-10)

    def test_ssr_never_exceeds_the_null(self):
        """Adding a column cannot make the fit worse."""
        rng = np.random.default_rng(7)
        y = np.cumsum(rng.normal(size=160))
        for kind in ("step", "bend"):
            s = bd.scan(y, kind=kind, trend=2, seasonal=12, lags=6)
            assert s.ssr.max() <= s.ssr_null + 1e-9

    def test_a_noiseless_bend_is_recovered_exactly(self):
        n, tau = 200, 88
        y = bd.break_column(n, tau, "bend") * 2.5
        s = bd.scan(y, kind="bend", trend=True, lags=0)
        assert s.best == tau
        assert s.min_ssr < 1e-18
        assert s.at(tau)["coef"] == pytest.approx(2.5)

    def test_a_noiseless_step_is_recovered_exactly(self):
        n, tau = 200, 88
        y = bd.break_column(n, tau, "step") * 2.5
        s = bd.scan(y, kind="step", trend=True, lags=0)
        assert s.best == tau and s.min_ssr < 1e-18

    def test_an_infinite_t_is_reported_rather_than_a_zero_division(self):
        n, tau = 120, 60
        y = bd.break_column(n, tau, "bend") * 2.0
        with np.errstate(all="raise"):
            s = bd.scan(y, kind="bend", trend=True, lags=0)
        assert np.isinf(s.at(tau)["t_hac"])

    def test_stride_thins_the_grid_without_moving_it(self):
        rng = np.random.default_rng(9)
        y = noisy_bend(200, 90, 2.0, rng)
        full = bd.scan(y, kind="bend", trend=True, lags=0)
        thin = bd.scan(y, kind="bend", trend=True, lags=0, stride=5)
        assert set(thin.dates.tolist()) <= set(full.dates.tolist())
        for tau in thin.dates:
            assert thin.at(int(tau))["ssr"] == pytest.approx(full.at(int(tau))["ssr"])

    def test_bend_grid_stops_two_from_the_end(self):
        s = bd.scan(np.random.default_rng(1).normal(size=100), kind="bend",
                    trim=0.0, trend=True, lags=0)
        assert s.dates.max() <= 98

    def test_at_refuses_a_date_off_the_grid(self):
        s = bd.scan(np.random.default_rng(1).normal(size=100), kind="step",
                    trend=True, lags=0, stride=4)
        off = next(t for t in range(s.dates.min(), s.dates.max())
                   if t not in s.dates.tolist())
        with pytest.raises(KeyError):
            s.at(off)


# ---------------------------------------------------------------------------
# the bias that does not go away
# ---------------------------------------------------------------------------

class TestStepFittedToABend:
    def test_the_bend_is_recovered_and_the_step_is_not(self):
        r = bd.noise_free_step_date(300, 150, trend=True)
        assert r["bend_error"] == 0
        assert abs(r["step_error"]) > 24

    @pytest.mark.parametrize("n", [100, 200, 400, 800])
    def test_the_misplacement_is_a_fixed_share_of_the_sample(self, n):
        """Not sampling error: no amount of data shrinks it.

        With a linear trend the step lands about 17% of the sample late,
        whatever the sample is, which is the cleanest available statement that
        this is bias rather than noise.
        """
        tau = int(0.6 * n)
        r = bd.noise_free_step_date(n, tau, trend=True)
        assert r["step_error"] / n == pytest.approx(0.174, abs=0.01)

    def test_the_misplacement_does_not_depend_on_how_sharp_the_bend_is(self):
        a = bd.noise_free_step_date(300, 150, size=0.01, trend=True)
        b = bd.noise_free_step_date(300, 150, size=100.0, trend=True)
        assert a["step_date"] == b["step_date"]

    def test_under_a_quadratic_trend_even_the_direction_moves(self):
        """A quadratic absorbs part of the bend, so the sign is not stable.

        This is worse for a practitioner than a predictable bias: there is no
        rule of thumb to apply, because the direction depends on where in the
        sample the bend sits.
        """
        errs = [bd.noise_free_step_date(312, tau, trend=2, seasonal=12)["step_error"]
                for tau in (108, 150, 186, 222)]
        assert max(errs) > 0 > min(errs)
        assert min(abs(e) for e in errs) > 12

    def test_with_real_persistence_the_step_date_scatters_much_wider(self):
        rng = np.random.default_rng(21)
        resid = ar1(312, 0.85, rng, 0.15)
        r = bd.step_on_bend(resid, n=312, tau=150, size=4.0, block=36, reps=60,
                            trend=2, seasonal=12, lags=24,
                            rng=np.random.default_rng(22))
        step_iqr = r["step_iqr"][1] - r["step_iqr"][0]
        bend_iqr = r["bend_iqr"][1] - r["bend_iqr"][0]
        # Measured at about 3x on this synthetic AR(0.85) noise and about 6x on
        # the real residuals; the threshold is set below both rather than at
        # whichever one flatters the claim.
        assert step_iqr > 2.5 * bend_iqr
        assert abs(r["bend_bias"]) < 12


# ---------------------------------------------------------------------------
# choosing between the shapes
# ---------------------------------------------------------------------------

class TestModelRace:
    def test_a_loud_bend_is_called_a_bend(self):
        rng = np.random.default_rng(31)
        y = noisy_bend(300, 140, 6.0, rng, sigma=0.15)
        r = bd.model_race(y, trend=True, lags=6)
        assert r["winner"] == "bend"
        assert r["bend_date"] == pytest.approx(140, abs=12)

    def test_a_loud_step_is_called_a_step(self):
        rng = np.random.default_rng(32)
        y = bd.break_column(300, 140, "step") * 3.0 + rng.normal(scale=0.2, size=300)
        r = bd.model_race(y, trend=True, lags=6)
        assert r["winner"] == "step"
        assert r["step_date"] == pytest.approx(140, abs=3)

    def test_the_null_split_is_not_fifty_fifty(self):
        """A bend is a smoother function of the date, so it wins less often.

        Reported because "the bend fits better" is unreadable without it: the
        baseline is not a coin flip and assuming it is would overstate every
        bend result.
        """
        rng = np.random.default_rng(33)
        resid = ar1(312, 0.8, rng, 0.15)
        r = bd.null_race(resid, n=312, block=36, reps=80, trend=2, seasonal=12,
                         lags=24, stride=3, rng=np.random.default_rng(34))
        assert 0.05 < r["bend_win_share"] < 0.40
        assert r["bend_gain_q95"] > 0 and r["step_gain_q95"] > 0

    def test_the_race_is_valid_but_underpowered(self):
        """Both halves measured on the same footing, and both reported.

        The null margin sets the bar; the alternative says how often a real bend
        of a realistic size clears it. It is well under half, which means a
        negative race result is not evidence of a step.
        """
        rng = np.random.default_rng(35)
        resid = ar1(312, 0.85, rng, 0.15)
        kw = dict(trend=2, seasonal=12, lags=24)
        null = bd.null_race(resid, n=312, block=36, reps=60, stride=3,
                            rng=np.random.default_rng(36), **kw)
        alt = bd.step_on_bend(resid, n=312, tau=150, size=4.0, block=36,
                              reps=60, rng=np.random.default_rng(37), **kw)
        power = float((alt["margin"] > null["bend_gain_q95"]).mean())
        assert 0.10 < power < 0.70


# ---------------------------------------------------------------------------
# the date, which is the worst estimated thing in the regression
# ---------------------------------------------------------------------------

class TestDate:
    def test_a_bend_date_is_far_less_identified_than_a_step_date(self):
        """Same data length, same noise, comparable signal strength."""
        rng = np.random.default_rng(41)
        n, tau = 240, 120
        noise = ar1(n, 0.7, rng, 0.15)
        bend_y = bd.break_column(n, tau, "bend") * 3.0 + noise
        step_y = bd.break_column(n, tau, "step") * 0.6 + noise
        kw = dict(trend=True, seasonal=12, lags=12, block=24, reps=120)
        b = bd.date_bootstrap(bend_y, kind="bend", rng=np.random.default_rng(42), **kw)
        s = bd.date_bootstrap(step_y, kind="step", rng=np.random.default_rng(43), **kw)
        assert (b["hi"] - b["lo"]) > 2 * (s["hi"] - s["lo"])

    def test_the_interval_contains_its_own_point_estimate(self):
        rng = np.random.default_rng(44)
        y = noisy_bend(240, 120, 3.0, rng, sigma=0.2, rho=0.6)
        ci = bd.date_bootstrap(y, kind="bend", block=24, reps=150, trend=True,
                               lags=12, rng=np.random.default_rng(45))
        assert ci["lo"] <= ci["tau_hat"] <= ci["hi"]

    def test_the_interval_covers_the_truth_at_about_the_rate_it_claims(self):
        """The control. A date interval is not guaranteed to be honest.

        Nominal 90%; anything from 80 to 99 is accepted because the check itself
        is a simulation with a few dozen replications, and a point estimate of
        coverage from 60 draws has a standard error near 4 points.
        """
        rng = np.random.default_rng(46)
        resid = ar1(312, 0.85, rng, 0.15)
        cov = bd.date_coverage(resid, n=312, tau=150, size=4.0, block=36,
                               kind="bend", reps=40, inner=100, level=0.90,
                               trend=2, seasonal=12, lags=24,
                               rng=np.random.default_rng(47))
        assert 0.80 <= cov["covered"] <= 0.99
        assert cov["median_width"] > 12


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

class TestCalibratedSup:
    def test_searching_dates_costs_more_than_a_fixed_date(self):
        rng = np.random.default_rng(51)
        resid = ar1(312, 0.85, rng, 0.15)
        cs = bd.calibrated_sup(resid, n=312, block=36, kind="bend", reps=80,
                               trend=2, seasonal=12, lags=24, stride=4,
                               rng=np.random.default_rng(52))
        assert cs["sup"] > cs["fixed"] > 1.96

    def test_the_conventional_bar_is_nowhere_near_correctly_sized(self):
        rng = np.random.default_rng(53)
        resid = ar1(312, 0.85, rng, 0.15)
        cs = bd.calibrated_sup(resid, n=312, block=36, kind="bend", reps=80,
                               trend=2, seasonal=12, lags=24, stride=4,
                               rng=np.random.default_rng(54))
        assert cs["size_of_1p96_sup"] > 0.30

    def test_a_bend_search_costs_less_than_a_step_search(self):
        """Adjacent bend columns are more alike, so the maximum is less extreme."""
        rng = np.random.default_rng(55)
        resid = ar1(312, 0.85, rng, 0.15)
        kw = dict(n=312, block=36, reps=80, trend=2, seasonal=12, lags=24,
                  stride=4)
        b = bd.calibrated_sup(resid, kind="bend",
                              rng=np.random.default_rng(56), **kw)
        s = bd.calibrated_sup(resid, kind="step",
                              rng=np.random.default_rng(56), **kw)
        assert b["sup"] < s["sup"]


# ---------------------------------------------------------------------------
# power for an announced bend
# ---------------------------------------------------------------------------

class TestAnnouncedBendPower:
    def test_slope_scaling_round_trips(self):
        """The claim goes in per-period, comes back per-period, at any length."""
        for n in (50, 200, 1000):
            y = bd.break_column(n, n // 2, "bend") * bd.size_from_slope(0.01, n)
            f = bd.fit_at(y, n // 2, kind="bend", trend=True, lags=0)
            assert bd.slope_change_per_year(f["coef"], n, 1) == pytest.approx(0.01)

    def test_the_same_claim_is_the_same_power_whatever_the_scaling(self):
        """A coefficient is length-dependent; a slope change is not.

        This is the test that would have caught carrying a fitted coefficient
        into a power curve over growing samples, which silently shrinks the
        claim as the horizon lengthens.
        """
        rng = np.random.default_rng(61)
        r = rng.normal(scale=0.04, size=40)
        a = bd.bend_power(r, n_pre=33, n_post=15, slope_change=0.01, block=8,
                          reps=200, trend=1, lags=4,
                          rng=np.random.default_rng(1))
        b = bd.bend_power(r, n_pre=33, n_post=15,
                          slope_change=0.01, block=8, reps=200, trend=1, lags=4,
                          rng=np.random.default_rng(1))
        assert a["power_hac"] == b["power_hac"]
        assert bd.size_from_slope(0.01, 48) != bd.size_from_slope(0.01, 63)

    def test_calibrated_bar_beats_1p96_and_says_by_how_much(self):
        rng = np.random.default_rng(62)
        r = ar1(33, 0.5, rng, 0.04)
        cb = bd.calibrated_fixed(r, n_pre=33, n_post=15, block=8, reps=600,
                                 rng=np.random.default_rng(63), trend=1, lags=4)
        assert cb["critical"] > 1.96
        assert cb["size_of_1p96"] > 0.05

    def test_size_is_right_at_the_calibrated_bar_by_construction(self):
        """Power at a zero slope change must come back at the level.

        Not a tautology worth skipping: the bar is calibrated by one function
        and the power measured by another, and a mismatch in how either builds
        the design would show up here and nowhere else.
        """
        rng = np.random.default_rng(64)
        r = rng.normal(scale=0.04, size=33)
        kw = dict(trend=1, lags=4)
        cb = bd.calibrated_fixed(r, n_pre=33, n_post=15, block=8, reps=1500,
                                 rng=np.random.default_rng(65), **kw)
        p = bd.bend_power(r, n_pre=33, n_post=15, slope_change=0.0, block=8,
                          reps=1500, critical=cb["critical"],
                          rng=np.random.default_rng(66), **kw)
        assert 0.02 <= p["power_hac"] <= 0.09

    def test_power_rises_with_the_claim_and_with_the_horizon(self):
        rng = np.random.default_rng(67)
        r = rng.normal(scale=0.04, size=33)
        kw = dict(trend=1, lags=4, block=8, reps=300, n_pre=33)
        by_size = [bd.bend_power(r, n_post=12, slope_change=s,
                                 rng=np.random.default_rng(7), **kw)["power_hac"]
                   for s in (0.002, 0.01, 0.03)]
        by_horizon = [bd.bend_power(r, n_post=h, slope_change=0.01,
                                    rng=np.random.default_rng(7), **kw)["power_hac"]
                      for h in (5, 12, 25)]
        assert by_size == sorted(by_size)
        assert by_horizon == sorted(by_horizon)

    def test_a_bend_the_data_already_shows_comes_back_at_full_power(self):
        """The known-answer check. A detected bend must simulate as detectable.

        Built from a series with a real kink: the fitted alternative's residual
        is the noise the power calculation is entitled to assume, and against it
        the kink that produced a large t-statistic has to be near-certain to
        detect. Running this against the *no-break* residual instead returns
        about 74%, because that residual still contains the kink — which is the
        error this test exists to catch.
        """
        rng = np.random.default_rng(68)
        n, tau = 33, 17
        t = np.arange(n, dtype=float)
        y = (0.062 * t - 0.043 * np.maximum(t - tau, 0.0)
             + rng.normal(scale=0.04, size=n))
        f = bd.fit_at(y, tau, kind="bend", trend=1, lags=4)
        assert abs(f["t_hac"]) > 6
        kw = dict(trend=1, lags=4)
        cb = bd.calibrated_fixed(f["resid"], n_pre=tau, n_post=n - tau, block=8,
                                 reps=1200, rng=np.random.default_rng(69), **kw)
        p = bd.bend_power(f["resid"], n_pre=tau, n_post=n - tau,
                          slope_change=-0.043, block=8, reps=400,
                          critical=cb["critical"],
                          rng=np.random.default_rng(70), **kw)
        assert p["power_hac"] > 0.95

    def test_minimum_detectable_bend_hits_its_target(self):
        rng = np.random.default_rng(71)
        r = rng.normal(scale=0.04, size=33)
        kw = dict(trend=1, lags=4)
        cb = bd.calibrated_fixed(r, n_pre=33, n_post=15, block=8, reps=800,
                                 rng=np.random.default_rng(72), **kw)
        m = bd.minimum_detectable_bend(r, n_pre=33, n_post=15, block=8,
                                       reps=400, hi=0.06,
                                       critical=cb["critical"], **kw)
        p = bd.bend_power(r, n_pre=33, n_post=15, slope_change=m["mde"],
                          block=8, reps=800, critical=cb["critical"],
                          rng=np.random.default_rng(73), **kw)
        assert 0.70 <= p["power_hac"] <= 0.90

    def test_minimum_detectable_bend_reports_when_nothing_is_detectable(self):
        rng = np.random.default_rng(74)
        r = rng.normal(scale=2.0, size=33)
        m = bd.minimum_detectable_bend(r, n_pre=33, n_post=4, block=8, reps=120,
                                       hi=1e-4, trend=1, lags=4)
        assert m["mde"] == float("inf") and "note" in m

    def test_periods_to_detect_agrees_with_its_own_curve(self):
        rng = np.random.default_rng(75)
        r = rng.normal(scale=0.04, size=33)
        res = bd.periods_to_detect(r, n_pre=33, slope_change=0.02, block=8,
                                   candidates=(5, 10, 20), reps=300,
                                   trend=1, lags=4)
        assert set(res["curve"]) == {5, 10, 20}
        if res["first_cleared"] is not None:
            assert res["curve"][res["first_cleared"]] >= res["target"]
            assert all(res["curve"][k] < res["target"]
                       for k in res["curve"] if k < res["first_cleared"])
