"""The embedding table's geometry, and what a gradient step does in it.

The geometry claims are pinned tightly because they follow from dimension
rather than from training. The behavioural ones are pinned as inequalities,
and deliberately in both directions: the episode's point is that the gradient
fails at one job and succeeds at another, so a test suite that only pinned
the failure would let the interesting half rot.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from standarderror.calculus import embeddings as em  # noqa: E402
from standarderror.llm import tiny  # noqa: E402


@pytest.fixture(scope="module")
def tbl():
    return em.table()


@pytest.fixture(scope="module")
def sub():
    x, y = next(iter(tiny.batches(count=1, size=1, seed=0)))
    return em.substitutions(x[0, :tiny.BLOCK - 1], int(y[0, tiny.BLOCK - 1]))


class TestTheTableIsNearlyOrthogonal:

    def test_pairwise_distance_matches_the_orthogonal_prediction(self, tbl):
        """sqrt(2) times the norm, to better than 1%. This is the episode's
        foundation and it is arithmetic, not a fit."""
        assert tbl["median_pairwise"] == pytest.approx(
            tbl["orthogonal_prediction"], rel=0.01)

    def test_the_cosines_are_centred_on_zero(self, tbl):
        assert abs(tbl["median_cosine"]) < 0.02
        assert tbl["abs_cosine_p90"] < 0.3

    def test_there_is_no_meaningful_nearest_neighbour(self, tbl):
        """The closest other token is nearly as far as the typical one, so
        'nearby in embedding space' is not a distinguished relation."""
        assert tbl["nearest_over_pairwise"] > 0.7

    def test_the_rows_sit_near_a_sphere(self, tbl):
        assert tbl["norm_spread"] < 0.15

    def test_the_prediction_is_what_high_dimensions_do(self, tbl):
        """A control: random vectors of the same norm and dimension reproduce
        the same relationship, so nothing here is about this model."""
        rng = np.random.default_rng(0)
        r = rng.normal(size=(tbl["rows"], tbl["dim"]))
        r *= tbl["median_norm"] / np.linalg.norm(r, axis=1, keepdims=True)
        d = np.linalg.norm(r[:, None] - r[None], axis=-1)
        np.fill_diagonal(d, np.nan)
        assert np.nanmedian(d) == pytest.approx(
            tbl["median_norm"] * math.sqrt(2.0), rel=0.02)


class TestSubstitutionsAreEnumerated:

    def test_the_current_token_is_in_the_table(self, sub):
        assert sub["loss"][sub["current"]] == sub["current_loss"]

    def test_there_is_room_to_improve(self, sub):
        assert sub["best_loss"] < sub["current_loss"]

    def test_the_table_has_one_row_per_token(self, sub):
        assert sub["loss"].shape == (sub["vocab"],)


class TestADescentStepGoesNowhere:

    def test_the_nearest_row_is_still_where_you_started(self, sub):
        """For a step of the size anyone would actually take."""
        land = em.step_landing(sub, limit=1.5, step=0.05)
        assert land["flip_at"] is None

    def test_the_walk_needed_is_absurd(self, sub):
        land = em.step_landing(sub, limit=60.0)
        assert land["flip_at"] is not None
        assert land["flip_at"] > 1.5
        assert land["distance_walked"] > land["embedding_norm"]

    def test_and_it_does_not_land_on_the_best_token(self, sub):
        land = em.step_landing(sub, limit=60.0)
        assert land["landed"] != sub["best"]
        assert land["landed_rank"] > 0


class TestTheRankingIsTheOtherQuestion:

    @pytest.fixture(scope="class")
    def rank(self, sub):
        return em.ranking(sub)

    def test_the_shortlist_recovers_nearly_all_of_the_gain(self, sub, rank):
        """The metric that matches how the score is actually used. It can be
        excellent while the rank of the single best token is poor, and the
        episode turns on exactly that."""
        assert rank["recovered5"] > 0.9

    def test_rank_and_recovery_can_disagree(self):
        """Not a hypothetical: across six contexts one has the best token at
        rank 55 and still recovers most of the improvement."""
        s = em.survey(contexts=6)
        assert s["rank_max"] > 20
        assert s["recovered_above_95"] >= 4

    def test_the_ordering_is_only_weakly_right(self, rank):
        assert -0.6 < rank["spearman"] < 0.8

    def test_the_score_is_a_directional_derivative(self, sub, rank):
        """`<g, E_j - E_cur>`, so the current token scores exactly zero -- a
        cheap check that the anchor has not drifted."""
        assert abs(rank["score"][sub["current"]]) < 1e-6
