"""What autodiff returns where the derivative does not exist.

The catalogue is measured rather than remembered, so these tests pin the
*pattern* -- some operations return zero, some split the tie, one returns nan
-- rather than individual floats. If a torch release changes a kernel's choice,
the test that fails should be the one describing the shape of the
inconsistency, not a hard-coded number.
"""

import pytest

pytest.importorskip("torch")

from standarderror.calculus import kinks as kk  # noqa: E402


@pytest.fixture(scope="module")
def catalogue():
    return {(k.name, k.point): k for k in kk.catalogue()}


class TestTheChoicesAtAKink:
    def test_the_zero_choosers(self, catalogue):
        for key in (("relu(x)", 0.0), ("abs(x)", 0.0),
                    ("clamp(x, 0, 1)", 0.0), ("clamp(x, 0, 1)", 1.0),
                    ("hardtanh(x)", 1.0), ("norm(x)", 0.0)):
            assert catalogue[key].returned == 0.0, catalogue[key]

    def test_the_tie_splitters(self, catalogue):
        """`maximum`, `minimum` and `max` hand half the gradient to each side,
        which is a different convention from `relu` in the same library."""
        for key in (("maximum(x, 0)", 0.0), ("minimum(x, 0)", 0.0),
                    ("max(stack(x, 0))", 0.0)):
            assert catalogue[key].returned == 0.5, catalogue[key]

    def test_two_spellings_of_abs_disagree(self, catalogue):
        """`sqrt(x*x)` is `abs(x)`. One returns 0 and the other nan."""
        assert catalogue[("abs(x)", 0.0)].returned == 0.0
        got = catalogue[("sqrt(x * x)", 0.0)].returned
        assert got != got, got                        # nan

    def test_every_choice_but_one_is_at_least_a_subgradient(self, catalogue):
        bad = [k for k in catalogue.values() if not k.admissible]
        assert [k.name for k in bad] == ["sqrt(x * x)"], bad

    def test_the_catalogue_is_measured_not_stored(self, catalogue):
        """Three distinct answers appear, which is the finding; if a release
        made them consistent this would fail and the episode would need
        rewriting rather than quietly going stale."""
        answers = {k.returned for k in catalogue.values()
                   if k.returned == k.returned}
        assert answers == {0.0, 0.5}, answers


class TestAFunctionOfTheExpression:
    @pytest.fixture(scope="class")
    def forms(self):
        return {r["form"]: r for r in kk.identity_three_ways(0.0)}

    def test_all_three_are_the_identity(self, forms):
        for point in (-1.0, 0.0, 1.0):
            for r in kk.identity_three_ways(point):
                assert r["value"] == pytest.approx(point), r

    def test_and_they_disagree_about_the_derivative_at_zero(self, forms):
        assert forms["x"]["derivative"] == 1.0
        assert forms["relu(x) - relu(-x)"]["derivative"] == 0.0
        assert forms["relu(x) + min(x, 0)"]["derivative"] == 0.5

    def test_two_of_the_three_are_not_even_subgradients(self, forms):
        """The subdifferential of the identity is the single point {1}."""
        assert sum(not r["is_subgradient"] for r in forms.values()) == 2

    def test_away_from_the_kink_they_agree(self):
        for point in (-1.5, -0.25, 0.25, 1.5):
            got = {r["derivative"] for r in kk.identity_three_ways(point)}
            assert got == {1.0}, (point, got)


class TestTheMaskedRow:
    def test_a_fully_masked_softmax_is_nan_in_the_forward_pass(self):
        """The usual mental model puts NaNs in the gradient. This one is in the
        forward pass: exp(-inf) is 0 everywhere and 0/0 is nan."""
        row = kk.fully_masked_softmax()
        assert row["forward_all_nan"] is True


class TestWhatTheModelActuallyDoes:
    @pytest.fixture(scope="module")
    def survey(self):
        return kk.attention_survey(count=8)

    def test_no_probability_is_exactly_zero_or_one(self, survey):
        """Away from position 0, in four million of them. Which is why the
        kink catalogue above does not describe this model."""
        assert survey["exact_zero"] == 0
        assert survey["exact_one"] == 0
        assert survey["probabilities"] > 1_000_000

    def test_but_every_first_row_is_exactly_one(self, survey):
        """A softmax over a single element is the constant 1, so the gradient
        there is identically zero. Structural: the causal mask puts it there,
        not the training."""
        assert survey["first_rows_onehot"] == survey["first_rows"]
        assert survey["first_rows"] > 0

    def test_the_heads_are_confident_without_being_saturated(self, survey):
        """Which is episode 2's subject: the gradient scale is p(1-p), so 13%
        of rows are already passing under a tenth of it while remaining
        perfectly smooth."""
        assert 0.4 < survey["median_max_p"] < 0.7
        assert survey["share_above"][0.9] > 0.05
        assert survey["share_above"][0.9999] == 0.0
        assert survey["largest_max_p"] < 1.0
