"""Machine Learning, Taught Through What Breaks: the measurement layer.

Theorems and identities are pinned tightly; properties of one dataset or
one simulation are pinned as inequalities with room for seed noise.
"""
import math

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from standarderror.ml import evaluation as ev  # noqa: E402
from standarderror.ml import leakage as lk  # noqa: E402
from standarderror.ml import linear as ln  # noqa: E402
from standarderror.ml import reading as rd  # noqa: E402
from standarderror.ml import trees as tr  # noqa: E402


class TestATestScoreHasAnErrorBar:

    def test_the_pool_correction_is_the_hypergeometric_factor(self):
        b, p = ev.binomial_se(0.98, 360), ev.pool_se(0.98, 360, 1797)
        assert (p / b) ** 2 == pytest.approx((1797 - 360) / 1796)

    def test_a_fixed_correctness_vector_gives_the_pool_spread(self):
        c = np.r_[np.ones(1761, bool), np.zeros(36, bool)]
        got = ev.subset_sd(c, 360, draws=4000)
        assert got == pytest.approx(ev.pool_se(c.mean(), 360, len(c)),
                                    rel=0.05)

    def test_pairing_counts_only_disagreements(self):
        a = np.array([1, 1, 1, 0, 0, 1, 1, 1], bool)
        b = np.array([1, 1, 0, 1, 0, 1, 1, 1], bool)
        r = ev.paired(a, b)
        assert (r["only_a"], r["only_b"]) == (1, 1)
        assert r["difference"] == 0 and r["p_value"] == 1.0

    def test_rows_needed_scales_with_disagreement_over_gap_squared(self):
        assert ev.rows_needed(0.01, 0.04) == pytest.approx(
            4 * ev.rows_needed(0.02, 0.04), rel=0.01)
        assert ev.rows_needed(0.01, 0.04) == pytest.approx(
            2 * ev.rows_needed(0.01, 0.02), rel=0.01)

    def test_on_digits_the_top_two_are_not_separable(self):
        X, y = ev.digits()
        cvc = ev.cv_correct(X, y)
        r = ev.paired(cvc["3-NN"], cvc["RBF SVM"])
        assert r["p_value"] > 0.05
        assert r["paired_se"] < r["unpaired_se"]
        assert ev.rows_needed(abs(r["difference"]),
                              r["discordance"]) > len(y)
        assert ev.paired(cvc["RBF SVM"], cvc["logistic"])["p_value"] < 0.001


class TestWhatCrossValidationEstimates:

    @pytest.fixture(scope="class")
    def lin(self):
        return ev.linear_cv_study(reps=600)

    def test_cv_is_uncorrelated_with_the_fitted_models_error(self, lin):
        assert abs(np.corrcoef(lin["cv"], lin["err_xy"])[0, 1]) < 0.1

    def test_and_far_noisier_than_what_it_should_track(self, lin):
        assert lin["cv"].std() > 2 * lin["err_xy"].std()

    def test_a_held_out_set_does_track_it(self, lin):
        assert np.corrcoef(lin["holdout"], lin["err_xy"])[0, 1] > 0.3

    def test_naive_cv_intervals_undercover(self, lin):
        assert ev.coverage(lin["cv"], lin["cv_se"], lin["err_xy"]) < 0.86

    def test_the_conditional_error_is_exact(self):
        """1 + |b - beta|^2 against a Monte Carlo of fresh rows."""
        rng = np.random.default_rng(3)
        beta, b = np.full(5, 0.5), np.full(5, 0.5) + rng.normal(0, 0.3, 5)
        X = rng.standard_normal((200_000, 5))
        y = X @ beta + rng.standard_normal(200_000)
        mc = float(np.mean((y - X @ b) ** 2))
        assert mc == pytest.approx(1 + np.sum((b - beta) ** 2), rel=0.01)


class TestNotEveryLeakLeaks:

    def test_selection_from_noise_leaks_a_lot(self):
        r = lk.noise_selection(p=2000, reps=12)
        assert r["leaky"].mean() - r["honest"].mean() > 0.2
        assert abs(r["honest"].mean() - 0.5) < 0.07

    def test_with_nothing_to_choose_there_is_nothing_to_leak(self):
        r = lk.noise_selection(p=20, k=20, reps=5)
        assert np.array_equal(r["leaky"], r["honest"])

    @pytest.fixture(scope="class")
    def table(self):
        return {r["leak"]: r for r in lk.leak_table(reps=4)}

    def test_label_free_steps_leak_nothing_measurable(self, table):
        assert abs(table["standardise on all rows"]["gap"]) < 0.004
        assert abs(table["impute means on all rows"]["gap"]) < 0.004

    def test_strong_features_make_selection_harmless(self, table):
        assert abs(table["select 5 of 1,030 features on all rows"]["gap"]
                   ) < 0.004

    def test_target_encoding_and_duplicates_leak(self, table):
        assert table["target-encode a noise ID on all rows"]["gap"] > 0.01
        dl = table["duplicate rows across folds (logistic)"]["gap"]
        dk = table["duplicate rows across folds (1-NN)"]["gap"]
        assert dk > 3 * dl > 0

    def test_the_encoder_sees_only_what_it_was_fitted_on(self):
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        y = np.array([1, 0, 1, 1])
        enc = lk._TargetEncoder(0).fit(X[:2], y[:2])
        got = enc.transform(X)[:, 0]
        assert list(got) == [0.5, 0.5, 0.5, 0.5]
        assert math.isclose(enc.prior_, 0.5)


class TestLearningCurves:

    def test_an_exact_power_law_is_recovered(self):
        from standarderror.ml import curves as cu
        sizes = [50, 100, 200, 400, 800]
        curve = {"sizes": sizes,
                 "error": {"m": {n: 2.0 * n ** -0.4 for n in sizes}}}
        f = cu.fit(curve, "m", until=800, floor=False)
        assert f["b"] == pytest.approx(0.4, abs=1e-4)
        assert all(s == pytest.approx(-0.4) for s in cu.log_slope(curve, "m"))

    def test_a_floor_bends_the_log_slope_towards_zero(self):
        from standarderror.ml import curves as cu
        sizes = [50, 100, 200, 400, 800, 1600]
        curve = {"sizes": sizes,
                 "error": {"m": {n: 2.0 * n ** -0.5 + 0.1 for n in sizes}}}
        sl = cu.log_slope(curve, "m")
        assert all(a < b < 0 for a, b in zip(sl, sl[1:]))
        f = cu.fit(curve, "m", until=1600, floor=True)
        assert f["c"] == pytest.approx(0.1, abs=1e-3)
        two = cu.fit(curve, "m", until=400, floor=False)
        assert two["predict"](1600) < curve["error"]["m"][1600]

    def test_the_task_has_the_bayes_error_the_episode_quotes(self):
        from standarderror.ml import curves as cu
        assert cu.GaussianTask().bayes == pytest.approx(0.268, abs=0.002)

    def test_on_digits_the_pilot_winner_is_the_full_size_loser(self):
        from standarderror.ml import curves as cu
        d = cu.digits_curve(sizes=(50, 1200), reps=10)
        best = cu.best_at(d)
        assert best[50] == "logistic"
        assert max(d["error"], key=lambda k: d["error"][k][1200]) == "logistic"


class TestImbalance:

    def test_smote_rows_lie_between_minority_neighbours(self):
        from standarderror.ml import imbalance as im
        rng = np.random.default_rng(0)
        X = np.r_[rng.standard_normal((200, 2)), [[10, 10], [11, 10],
                                                  [10, 11], [11, 11],
                                                  [10.5, 10.5], [10.2, 10.8]]]
        y = np.r_[np.zeros(200, int), np.ones(6, int)]
        Xs, ys = im.smote(X, y, rng, k=3)
        new = Xs[len(X):]
        assert ys.mean() == 0.5
        assert new.min() >= 10 - 1e-9 and new.max() <= 11 + 1e-9

    def test_the_prior_correction_inverts_a_known_shift(self):
        from standarderror.ml import imbalance as im
        p = np.array([0.01, 0.02, 0.2, 0.6])
        odds = p / (1 - p) * (0.5 / 0.5) / (0.02 / 0.98)
        balanced = odds / (1 + odds)
        back = im.prior_correct(balanced, 0.5, 0.02)
        assert np.allclose(back, p)

    def test_reweighting_logistic_regression_is_mostly_a_threshold(self):
        from standarderror.ml import imbalance as im
        r = im.compare(model="logistic", reps=4, test=50_000)
        s, m = r["summary"], r["matched"]["class weights"]
        assert abs(s["class weights"]["auc"] - s["plain"]["auc"]) < 0.005
        assert s["class weights"]["mean_p"] > 5 * r["prevalence"]
        assert m["plain_precision"] >= m["method_precision"] - 0.005
        assert abs(r["corrected"]["mean_p"] - r["prevalence"]) < 0.005


class TestLinearModelsPushed:

    def test_a_separable_split_has_a_unit_max_margin_direction(self):
        Xa, _, ya, _ = ln.breast_cancer_split(0)
        w, sep = ln.max_margin(Xa, ya)
        assert sep and np.linalg.norm(w) == pytest.approx(1.0)

    def test_a_tighter_tolerance_only_grows_the_coefficients(self):
        rows = ln.tolerance_sweep(tols=(1e-2, 1e-6))
        assert rows[1]["norm"] > 10 * rows[0]["norm"]

    def test_rescaling_a_feature_changes_what_the_lasso_keeps(self):
        X, y, _ = ln.diabetes()
        assert (ln._lasso_cv(X, y)[0] != 0)[0]
        Z = X.copy()
        Z[:, 0] *= 0.01           # age, in much larger units
        assert not (ln._lasso_cv(Z, y)[0] != 0)[0]

    def test_the_one_se_rule_keeps_no_more_than_the_minimum(self):
        X, y, _ = ln.diabetes()
        assert (np.count_nonzero(ln._lasso_cv(X, y, rule="1se")[0])
                <= np.count_nonzero(ln._lasso_cv(X, y)[0]))

    def test_the_bias_variance_terms_add_up_to_the_measured_error(self):
        from sklearn.neighbors import KNeighborsRegressor
        r = ln.bias_variance(lambda: KNeighborsRegressor(4), reps=80)
        assert r["total"] == pytest.approx(r["mse"], abs=0.01)

    def test_breimans_formula_recovers_a_planted_correlation(self):
        s2, rho = 0.2, 0.4
        sweep = [{"trees": 1, "variance": s2},
                 {"trees": 50, "variance": rho * s2 + (1 - rho) * s2 / 50}]
        assert ln.forest_rho(sweep) == pytest.approx(rho)

    def test_min_norm_interpolates_past_the_threshold_and_peaks_at_it(self):
        c = ln.random_features(n=40, ps=[10, 30, 40, 50, 400], reps=8)
        assert ln.peak(c) == 40
        assert c["train"][50] < 1e-8 and c["train"][10] > 0.01
        assert ln.best_under(c)[0] in (10, 30)

    def test_ridge_flattens_the_peak(self):
        kw = dict(n=40, ps=[30, 40, 50], reps=8)
        assert (max(ln.random_features(ridge=1.0, **kw)["test"].values())
                < max(ln.random_features(**kw)["test"].values()) / 5)


class TestTreesAndEnsembles:

    def test_bootstrapped_trees_vary_more_in_structure_than_accuracy(self):
        b = tr.tree_bootstrap(boots=40)
        assert b["accuracy_sd"] < 0.03
        assert b["root_same"] < 0.9 and b["distinct_sets"] > 20

    def test_impurity_importance_favours_many_split_points_under_the_null(self):
        r = tr.importance_null(signal=0.0, reps=4, trees=50)
        assert r["mdi"][-1] > 3 * r["mdi"][0]

    def test_a_row_is_out_of_bag_for_about_one_tree_in_e(self):
        assert tr.oob_coverage(200) == pytest.approx(math.exp(-1), abs=0.01)

    def test_a_forest_never_predicts_above_its_largest_label(self):
        ex = tr.extrapolation(trees=30, bands=((1.5, 2.0),))
        assert ex["rows"][0]["forest_max"] <= ex["train_y_max"] + 1e-9

    def test_staged_boosting_curves_have_one_value_per_checkpoint(self):
        c = tr.boost_curves(rounds=30, reps=1)
        assert len(c["error"]) == len(c["rounds"]) == 4


class TestReadingAModel:

    def test_exact_shapley_matches_the_linear_closed_form(self):
        assert rd.linear_shapley_check() < 1e-9

    def test_exact_shapley_is_efficient_for_a_nonlinear_function(self):
        g = np.random.default_rng(0)
        B, x = g.standard_normal((50, 3)), g.standard_normal(3)
        phi, base, fx = rd.exact_shapley(
            lambda Z: Z[:, 0] * Z[:, 1] + np.sin(Z[:, 2]), x, B)
        assert phi.sum() == pytest.approx(fx - base, abs=1e-12)

    def test_ale_and_pdp_agree_on_an_additive_function_of_independent_inputs(self):
        class Additive:
            def predict(self, Z):
                return 2 * Z[:, 0] + Z[:, 1] ** 2

        X, _ = rd.correlated_pair(noise=None, seed=0)
        grid = np.linspace(0.1, 0.9, 9)
        p = rd.pdp(Additive(), X, 0, grid)
        a = rd.ale(Additive(), X, 0, grid)
        assert np.ptp(p - a) < 1e-9
        assert (p[-1] - p[0]) == pytest.approx(2 * 0.8)

    def test_shuffling_a_correlated_feature_leaves_the_data(self):
        o = rd.off_support()
        assert o["outside"] > 10 * o["expected"]

    def test_planted_markers_sit_on_blank_pixels(self):
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        assert X[:, list(rd.SHORTCUT.values())].max() <= 1
        Z = rd.plant(X, y, 1.0, seed=0)
        for cls, pix in rd.SHORTCUT.items():
            assert (Z[y == cls, pix] == 16).all()


class TestTrainingANetwork:

    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    def test_constant_training_pixels_keep_unit_scale(self):
        from standarderror.ml import nets
        for seed in range(5):
            _, Xb, _, _ = nets.digits(seed)
            assert float(Xb.abs().max()) < 1e3

    def test_fit_takes_exactly_the_steps_asked_for(self):
        from standarderror.ml import nets
        Xa, _, ya, _ = nets.digits(0)
        m = nets.mlp([64, 8, 10])
        calls = []
        m.register_forward_hook(lambda *a: calls.append(1))
        nets.fit(m, Xa, ya, batch=512, steps=7)
        assert len(calls) == 7

    def test_two_example_batch_norm_outputs_are_plus_or_minus_one(self):
        import torch
        bn = torch.nn.BatchNorm1d(16, affine=False).train()
        out = bn(torch.randn(2, 16, generator=torch.Generator().manual_seed(0)))
        # Short of 1 only where the two values nearly coincide (the epsilon).
        assert torch.allclose(out.abs(), torch.ones_like(out), atol=0.05)
        assert torch.allclose(out[0], -out[1])

    def test_a_deep_tanh_network_with_small_gain_loses_its_signal(self):
        from standarderror.ml import nets
        Xa, _, ya, _ = nets.digits(0)
        sizes = [64] + [128] * 30 + [10]
        small, _ = nets._layer_stats(nets.mlp(sizes, act="tanh", gain=0.5),
                                     Xa, ya)
        good, _ = nets._layer_stats(nets.mlp(sizes, act="tanh", gain=1.0),
                                    Xa, ya)
        assert small[-1] < 1e-6 < 0.05 < good[-1]


class TestGeneralisationInNetworks:

    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    def test_mirror_and_half_turn_are_involutions(self):
        from sklearn.datasets import load_digits

        from standarderror.ml import nets
        X = load_digits().data[:20]
        g = np.random.default_rng(0)
        for kind in ("flip", "rot180"):
            twice = nets.augment(nets.augment(X, kind, g), kind, g)
            assert np.array_equal(twice, X)

    def test_without_a_shift_the_importance_weights_are_one(self):
        from standarderror.ml import nets
        r = nets.covariate_shift(shifts=(0.0,), reps=2, test=2000)[0]
        assert r["ess"] == pytest.approx(1.0)
        assert r["acc_linear_iw"] == r["acc_linear"]

    def test_the_network_memorises_random_labels(self):
        from standarderror.ml import nets
        r = nets.random_labels(fractions=(1.0,), epochs=100)[0]
        assert r["train_acc"] == 1.0 and r["test_acc"] < 0.2

    def test_the_committed_grokking_run_reproduces(self):
        from standarderror.ml import nets
        stored = nets.grokking_runs()[0]
        assert stored["weight_decay"] == 1.0 and stored["frac"] == 0.4
        short = nets.modular(steps=200, every=100)
        for a, b in zip(short["history"], stored["history"][:2]):
            assert a["step"] == b["step"]
            assert a["train"] == pytest.approx(b["train"], abs=1e-6)
            assert a["test"] == pytest.approx(b["test"], abs=1e-6)
            assert a["norm"] == pytest.approx(b["norm"], rel=1e-4)

    def test_an_exact_power_law_is_fitted_exactly(self):
        from standarderror.ml import scaling as sc
        N = np.array([1e4, 3e4, 1e5, 3e5, 1e6])
        p = sc.fit(N, 20 * N ** -0.3 + 1.2)
        assert (p["a"], p["b"], p["c"]) == pytest.approx((20, 0.3, 1.2),
                                                         rel=1e-3)
        assert sc.predict(p, 1e7) == pytest.approx(20 * 1e7 ** -0.3 + 1.2,
                                                   rel=1e-4)


class TestAroundTheModel:

    def test_pca_ranks_by_variance_so_the_signal_comes_last(self):
        from standarderror.ml import around
        s = around.pca_simulation(n=600)
        assert s["share"] == sorted(s["share"], reverse=True)
        assert max(s["acc"][:9]) < 0.6 < 0.85 < s["acc"][9]

    def test_kmeans_recovers_separated_blobs_and_cuts_noise_anyway(self):
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        from standarderror.ml import around
        X, y, k = around._shapes("separated", 0)
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, lab) > 0.99
        U, _, _ = around._shapes("uniform", 0)
        assert len(set(KMeans(4, n_init=10, random_state=0).fit_predict(U))) == 4

    def test_distance_contrast_shrinks_with_dimension(self):
        from standarderror.ml import around
        c = around.contrast(dims=(2, 64), n=300, queries=50)
        assert c[0]["ratio"] < 0.1 < 0.6 < c[1]["ratio"]

    def test_platt_scaling_keeps_the_ranking(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        from standarderror.ml import around
        g = np.random.default_rng(0)
        y = g.integers(0, 2, 400)
        s = 1 / (1 + np.exp(-(2 * y - 1 + g.normal(size=400)) * 3))
        z = around._logit(s)[:, None]
        p = LogisticRegression().fit(z, y).predict_proba(z)[:, 1]
        assert roc_auc_score(y, p) == pytest.approx(roc_auc_score(y, s))

    def test_the_one_cell_grid_is_the_default(self):
        from standarderror.ml import around
        assert around._grid(1) == {"C": [1.0], "gamma": [0.1]}
        g = around._grid(16)
        assert len(g["C"]) * len(g["gamma"]) == 16


class TestWhereTheRowsComeFrom:

    def test_lags_line_up_with_their_targets(self):
        from standarderror.ml import rows
        y = np.arange(10.0)
        X, t = rows._lags(y, 2)
        assert X[0].tolist() == [1.0, 0.0] and t[0] == 2.0
        assert len(X) == len(t) == 8

    def test_a_short_window_correlates_independent_slow_processes(self):
        from standarderror.ml import rows
        w = rows.window_correlation(phis=(0.0, 0.99), reps=40)
        assert w[0.0] < 0.06 < 0.1 < w[0.99]

    def test_subjects_share_their_signature(self):
        from standarderror.ml import rows
        X, y, s = rows._subjects(5, 40, np.random.default_rng(0))
        assert X.shape == (200, 10) and len(set(s)) == 5
        within = np.mean([X[s == k].std(0).mean() for k in range(5)])
        assert within < X.std(0).mean()

    def test_missingness_masks_hit_their_rate_and_their_target(self):
        from standarderror.ml import rows
        X, y = rows._breast_cancer()
        g = np.random.default_rng(0)
        for kind in ("random", "large values", "by diagnosis"):
            m = rows._missing_mask(kind, X, y, g)
            assert abs(m.mean() - 0.3) < 0.02
        m = rows._missing_mask("by diagnosis", X, y, g)
        assert m[y == 1].mean() > 5 * m[y == 0].mean()

    def test_the_old_policy_accepts_half(self):
        from standarderror.ml import rows
        X, y, a = rows._applicants(4000, np.random.default_rng(0),
                                   curved=False, officer=True)
        assert a.mean() == pytest.approx(0.5, abs=0.001)
        assert y[a].mean() < y[~a].mean()

    def test_mean_encoding_reads_a_rows_own_label(self):
        from standarderror.ml import rows
        c = np.array([0, 0, 1, 1, 2])
        y = np.array([1, 0, 1, 1, 0])
        a, b = rows._mean_encode(c, y, np.array([0, 3]))
        assert a.tolist() == [0.5, 0.5, 1.0, 1.0, 0.0]
        assert b.tolist() == [0.5, 0.6]


class TestCombiningAndComparing:

    def test_the_ambiguity_identity_is_exact(self):
        from standarderror.ml import combine
        a = combine.ambiguity()
        assert a["identity_gap"] < 1e-9
        assert a["ensemble"] == pytest.approx(a["mean_member"] - a["ambiguity"])

    def test_naive_bayes_copies_scale_the_log_odds(self):
        from sklearn.naive_bayes import GaussianNB

        from standarderror.ml import combine
        X, y, _ = combine._bc()
        j1 = GaussianNB().fit(X, y).predict_joint_log_proba(X)
        j3 = GaussianNB().fit(np.tile(X, 3), y).predict_joint_log_proba(
            np.tile(X, 3))
        prior = np.log(np.bincount(y) / len(y))
        lo1 = (j1[:, 1] - prior[1]) - (j1[:, 0] - prior[0])
        lo3 = (j3[:, 1] - prior[1]) - (j3[:, 0] - prior[0])
        assert np.allclose(lo3, 3 * lo1, rtol=1e-6, atol=1e-6)

    def test_the_rotated_task_is_rotation_invariant(self):
        from standarderror.ml import combine
        X, y = combine._ttask(20000, np.random.default_rng(0))
        r0, r1 = np.linalg.norm(X[y == 0], axis=1), np.linalg.norm(X[y == 1], axis=1)
        assert r1.mean() > 1.4 * r0.mean()
        assert abs(np.corrcoef(X.T)[0, 1]) < 0.03

    def test_a_leaderboard_rewards_kept_random_guesses(self):
        from standarderror.ml import combine
        r = combine.leaderboard_attack(queries=(1, 300), n=500, reps=10)
        assert r[1]["public"] > 0.58 and abs(r[1]["private"] - 0.5) < 0.03


class TestPredictionToDecision:

    def test_best_cost_matches_a_brute_force_search(self):
        from standarderror.ml import decide
        g = np.random.default_rng(0)
        y = g.integers(0, 2, 300)
        p = np.clip(0.5 * y + 0.5 * g.random(300), 0, 1)
        brute = min(decide._cost(p, y, t, 5) for t in np.r_[p, 2.0])
        assert decide._best_cost(p, y, 5) == pytest.approx(brute)
        t = decide._best_threshold(p, y, 5)
        assert decide._cost(p, y, t, 5) == pytest.approx(brute)

    def test_intervening_downstream_changes_nothing(self):
        from standarderror.ml import decide
        _, y0, _ = decide._scm(5000, np.random.default_rng(3))
        _, y1, _ = decide._scm(5000, np.random.default_rng(3), do="x3", shift=2.0)
        _, y2, _ = decide._scm(5000, np.random.default_rng(3), do="x2", shift=2.0)
        assert (y0 == y1).all() and y2.mean() > y0.mean()

    def test_the_offer_backfires_for_a_minority(self):
        from standarderror.ml import decide
        _, p0, p1 = decide._customers(50_000, np.random.default_rng(0))
        harmed = np.mean(p1 > p0)
        assert 0.05 < harmed < 0.08 and np.mean(p0 - p1) > 0.1

    def test_greedy_counting_locks_in(self):
        from standarderror.ml import decide
        r = decide.feedback_loop(days=60, runs=40)
        assert r["greedy"]["locked"] == 1.0
        assert r["per visit"]["within_10"] > 0.8


class TestGradingALanguageModel:

    def test_paired_se_uses_only_disagreements(self):
        from standarderror.ml import llmeval as le
        g = np.random.default_rng(0)
        a = g.random(400) < 0.7
        b = a.copy()
        b[:10] = ~b[:10]
        r = le.paired_difference(a, b)
        assert r["only_a"] + r["only_b"] == 10
        assert r["se_paired"] < r["se_unpaired"] / 3

    def test_cluster_se_counts_questions(self):
        from standarderror.ml import llmeval as le
        g = np.random.default_rng(1)
        q = g.random(400) < 0.5
        mat = np.repeat(q[:, None], 5, 1)          # five identical answers
        r = le.cluster_study(mat, questions=50, draws=800)
        assert r["icc"] == pytest.approx(1.0)
        assert r["empirical"] > 2 * r["naive"]
        assert r["robust"] == pytest.approx(r["empirical"], rel=0.15)

    def test_ppi_removes_a_known_bias(self):
        from standarderror.ml import llmeval as le
        g = np.random.default_rng(2)
        y = g.random(2000) < 0.5
        judge = y | (g.random(2000) < 0.3)          # passes 30% of wrong ones
        r = le.ppi_study(judge, y, labeled=80, draws=600)
        assert r["judge"]["bias"] > 0.1
        assert abs(r["ppi"]["bias"]) < 0.01 and r["ppi"]["coverage"] > 0.9
        assert r["tuned"]["width"] <= r["labels"]["width"] + 0.01

    def test_distractors_are_wrong_and_of_their_kind(self):
        from standarderror.ml import llmeval as le
        rng = np.random.default_rng(3)
        for ans in ("1712", "-36", "8"):
            for kind in le.DISTRACTORS:
                d = le.distractors(ans, kind, rng)
                assert len(d) == 3 and ans not in d
                if kind in ("random", "near"):
                    assert all(len(x) == len(ans) for x in d)

    def test_contexts_of_each_kind(self):
        from standarderror.ml import llmeval as le
        for kind in le.FORMATS:
            c = le.contexts(kind, "mul", 3, seed=0)
            assert len(c) == 3
            lines = [x for x in c[0].split("\n") if x]
            assert len(lines) == {"empty": 0, "two": 2}.get(kind, 12)
            if kind == "same":
                assert all("*" in x and "=" in x for x in lines)

    def test_logp_prefers_the_right_sum(self):
        from standarderror.ml import llmeval as le
        from standarderror.schoolmath import model as sm
        b = sm.load()
        p = le.items("test", "add", 5)
        lp_right = le.answer_logp(b, [x.prompt for x in p],
                                  [x.answer for x in p], [""] * 5)
        lp_wrong = le.answer_logp(b, [x.prompt for x in p],
                                  [x.answer + "1" for x in p], [""] * 5)
        assert (lp_right > lp_wrong).all() and (lp_right <= 0).all()


class TestWithoutLabels:

    def test_noise_points_count_as_singletons(self):
        from standarderror.ml import unsup
        y = np.repeat([0, 1], 50)
        assert unsup.noise_ari(y, y) == pytest.approx(1.0)
        lab = y.copy()
        lab[50:] = -1                               # a whole cluster as noise
        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score(y, lab) == pytest.approx(1.0)
        assert unsup.noise_ari(y, lab) < 0.55

    def test_bic_keeps_one_gaussian_and_icl_one_square(self):
        from standarderror.ml import unsup
        rng = np.random.default_rng(0)
        b, _ = unsup._picks(unsup._bic_icl(rng.normal(size=(500, 2)),
                                           range(1, 5)))
        assert b == 1
        rows = unsup._bic_icl(rng.random((2000, 2)), range(1, 8))
        b, i = unsup._picks(rows)
        assert b > 2 and i == 1

    def test_knee_lies_on_the_k_distance_curve(self):
        from sklearn.datasets import make_moons

        from standarderror.ml import unsup
        X, _ = make_moons(300, noise=0.05, random_state=0)
        e = unsup.knee_eps(X)
        assert 0 < e < 0.5

    def test_stability_and_gap_on_three_blobs(self):
        from standarderror.ml import unsup
        rng = np.random.default_rng(1)
        X = np.r_[rng.normal(size=(150, 2)), rng.normal(size=(150, 2)) + [8, 0],
                  rng.normal(size=(150, 2)) + [4, 7]]
        assert unsup.stability(X, 3, pairs=5) > 0.95
        g = unsup.gap(X, ks=range(1, 6), refs=4)
        assert g["argmax"] == 3

    def test_contamination_is_a_quantile(self):
        from standarderror.ml import unsup
        r = unsup.anomaly_contamination(rates=(0.02,), settings=(0.05,),
                                        n=1000)
        assert r[0]["0.05"]["flagged"] == pytest.approx(0.05, abs=1e-3)


class TestAfterShipping:
    def test_drift_world_moves_inputs_or_relation(self):
        from standarderror.ml import deploy
        rng = np.random.default_rng(0)
        X, _ = deploy.drift_world(20_000, rng, shift=1.0)
        assert X[:, 0].mean() == pytest.approx(1.0, abs=0.03)
        X, y = deploy.drift_world(20_000, rng, flip=1.0)
        assert abs(X[:, 0].mean()) < 0.03
        assert np.corrcoef(X[:, 0], y)[0, 1] < -0.2      # the sign reversed

    def test_domain_weights_are_flat_without_shift(self):
        from standarderror.ml import deploy
        rng = np.random.default_rng(1)
        w = deploy._domain_weights(rng.normal(size=(2000, 3)),
                                   rng.normal(size=(2000, 3)))
        assert w.mean() == pytest.approx(1.0, abs=0.05) and w.std() < 0.2

    def test_bonferroni_alarms_less_than_uncorrected(self):
        from standarderror.ml import deploy
        r = deploy.drift_false_alarms(features=(20,), batch=200, batches=30)[0]
        assert r["bonferroni"] <= r["uncorrected"]
        assert r["uncorrected"] > 0.3

    def test_calibrated_groups_with_unequal_error_rates(self):
        from standarderror.ml import deploy
        r = deploy.fairness_study(n=8000)
        m = r["models"]["with group"] if "models" in r else r["with group"]
        assert abs(m["A"]["calibration_gap"]) < 0.03
        assert abs(m["B"]["calibration_gap"]) < 0.03
        assert m["B"]["fpr"] > m["A"]["fpr"] and m["A"]["fnr"] > m["B"]["fnr"]


class TestCourseMeasured:
    def test_design_problems_do_not_shrink(self):
        from standarderror.ml import course
        a = course._selection_bias(400, reps=3)
        b = course._selection_bias(3200, reps=3)
        assert a < -0.3 and b < -0.3 and abs(a - b) < 0.05

    def test_noise_problems_shrink(self):
        from standarderror.ml import course
        assert course._encoding_leak(200, reps=3) > 2 * course._encoding_leak(
            3200, reps=3) > 0

    def test_census_counts_every_episode(self):
        from standarderror.ml import course
        c = course.assert_census()
        assert c["episodes"] >= 69 and c["asserts"] > 10 * c["episodes"]
        assert set(range(1, 70)) <= set(c["per_episode"])

    def test_claims_are_twelve_and_seeded(self):
        from standarderror.ml import course
        assert len(course.CLAIMS) == 12
        assert len({c[0] for c in course.CLAIMS}) == 12
