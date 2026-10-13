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
