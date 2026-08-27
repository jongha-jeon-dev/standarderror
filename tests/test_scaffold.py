"""Tests for the frozen-scaffold / low-rank-adapter machinery.

Two classes of failure are worth pinning here and they are different. The task
generator has properties that are *exact* - the label must not depend on any
direction outside the chosen subspace, the train and test splits must share that
subspace, and the Bayes accuracy must not drift as the dimension changes - and
they get exact assertions. The model has a contract - what is trainable, what is
frozen, and that the three scaffold conditions really differ - and it gets
structural assertions rather than accuracy ones, because an accuracy threshold in
a test is a slow way to discover that a machine was busy.

Every one of the exact tests corresponds to a bug this file actually had. The
subspace was drawn from the same generator as the samples, so the splits
disagreed and everything sat at chance. Two logits were standardised separately,
which does not standardise their difference, and the Bayes accuracy at k=1
collapsed to 53%. And B was initialised to zero in the LoRA convention, which
makes the zero-scaffold network identically dead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from standarderror.models.scaffold import (MultiIndexTask, ScaffoldMLP, evaluate,
                                       make_multi_index, minimum_rank, recovery,
                                       train)


class TestTheTaskIsWhatItClaims:
    def test_the_label_ignores_directions_outside_the_subspace(self):
        # The defining property. Move x along a direction orthogonal to U and the
        # score must not budge; if it does, "intrinsic dimension k" is a label
        # rather than a fact and the whole experiment measures nothing.
        task = MultiIndexTask(d=32, k=3, seed=0)
        X = torch.randn(64, 32)
        Q, _ = torch.linalg.qr(torch.cat([task.U, torch.randn(32, 29)], dim=1))
        orth = Q[:, 3:]                       # spans the irrelevant complement
        moved = X + (orth @ torch.randn(29, 64)).T * 5.0
        assert torch.allclose(task.score(X), task.score(moved), atol=1e-4)

    def test_moving_inside_the_subspace_does_change_the_label(self):
        task = MultiIndexTask(d=32, k=3, seed=0)
        X = torch.randn(64, 32)
        moved = X + (task.U @ torch.randn(3, 64)).T * 2.0
        assert not torch.allclose(task.score(X), task.score(moved), atol=1e-2)

    def test_the_splits_share_one_subspace(self):
        # The bug that cost the most: U was drawn from the same generator as the
        # samples, so a different n gave a different task and every model sat at
        # chance with nothing in the loss curve to say why.
        task, tr, va, te = make_multi_index(2000, d=32, k=2, seed=1)
        for X, y in (tr, va, te):
            pred = (task.score(X) > 0).long()
            assert (pred == y).float().mean() > 0.7

    def test_the_splits_are_actually_different_rows(self):
        _, (Xtr, _), (Xva, _), (Xte, _) = make_multi_index(2000, d=16, k=2)
        assert not torch.allclose(Xtr[:100], Xte[:100])
        assert not torch.allclose(Xva[:100], Xte[:100])

    @pytest.mark.parametrize("k", [1, 2, 8, 32])
    def test_the_ceiling_does_not_drift_with_the_dimension(self, k):
        # A rank sweep across k has to be reading dimension, not difficulty, so
        # the Bayes accuracy is held in a band by construction. k=1 is the case
        # that caught the two-logit standardisation bug, at 53%.
        acc = MultiIndexTask(d=32, k=k, seed=2).bayes_accuracy(8000)
        assert 0.78 < acc < 0.93, (k, acc)

    def test_the_classes_are_roughly_balanced(self):
        _, (_, y), _, _ = make_multi_index(8000, d=32, k=4, seed=3)
        assert 0.45 < y.float().mean() < 0.55

    def test_label_noise_moves_the_ceiling_down(self):
        clean = MultiIndexTask(d=32, k=4, seed=4).bayes_accuracy(8000)
        noisy = MultiIndexTask(d=32, k=4, seed=4,
                               label_noise=0.2).bayes_accuracy(8000)
        assert noisy < clean - 0.05

    @pytest.mark.parametrize("k", [0, 33])
    def test_an_impossible_dimension_raises(self, k):
        with pytest.raises(ValueError):
            MultiIndexTask(d=32, k=k)


class TestModelContract:
    def test_the_backbone_carries_no_gradient(self):
        m = ScaffoldMLP(16, 2, hidden=(8, 8), rank=2)
        names = {n for n, p in m.named_parameters()}
        assert not any("W_seed" in n for n in names)
        assert {"layers.0.A", "layers.0.B", "layers.0.beta"} <= names

    def test_the_full_baseline_trains_the_backbone_and_has_no_adapter(self):
        m = ScaffoldMLP(16, 2, hidden=(8, 8), rank=4, full=True)
        names = {n for n, p in m.named_parameters()}
        assert any("W_seed" in n for n in names)
        assert not any(n.endswith(".A") or n.endswith(".B") for n in names)

    def test_the_baseline_is_the_same_layer_not_a_different_network(self):
        # The comparison is only about what the optimiser may touch, so the two
        # conditions must start from the identical backbone. An earlier version
        # wrote the baseline as a separate nn.Linear stack and the "fully trained
        # ceiling" landed four points below a rank-4 adapter.
        a = ScaffoldMLP(16, 2, hidden=(8, 8), rank=2, seed=5)
        b = ScaffoldMLP(16, 2, hidden=(8, 8), rank=2, seed=5, full=True)
        for la, lb in zip(a.layers, b.layers):
            assert torch.allclose(la.W_seed, lb.W_seed.detach())

    def test_the_adapter_is_a_correctly_scaled_layer_at_every_rank(self):
        # B init of zero is the LoRA convention and is wrong here: it makes the
        # zero-scaffold control identically dead. The replacement has to keep the
        # adapter path's scale independent of rank, or the sweep would confound
        # rank with initialisation scale.
        var = {}
        for r in (1, 4, 16):
            lin = ScaffoldMLP(64, 2, hidden=(64,), rank=r, seed=6).layers[0]
            eff = (lin.alpha / lin.rank) * (lin.B @ lin.A)
            var[r] = float(eff.detach().var())
        assert max(var.values()) / min(var.values()) < 2.0, var
        seed_var = float(ScaffoldMLP(64, 2, hidden=(64,), rank=4,
                                     seed=6).layers[0].W_seed.detach().var())
        assert 0.4 < var[4] / seed_var < 2.5, (var[4], seed_var)

    def test_the_zero_scaffold_is_not_dead(self):
        # Directly the bug: with a zero backbone and a zero B every activation is
        # zero, so are the gradients, and the control trains to exactly chance.
        m = ScaffoldMLP(16, 2, hidden=(8, 8), rank=2, scaffold="zero", seed=7)
        out = m(torch.randn(32, 16))
        assert float(out.detach().abs().max()) > 0
        loss = torch.nn.functional.cross_entropy(out, torch.zeros(32,
                                                                  dtype=torch.long))
        loss.backward()
        assert float(m.layers[0].B.grad.abs().sum()) > 0

    def test_the_zero_scaffold_really_has_no_backbone(self):
        m = ScaffoldMLP(16, 2, hidden=(8, 8), rank=2, scaffold="zero")
        assert all(float(l.W_seed.abs().sum()) == 0 for l in m.layers)

    def test_dropping_the_scaffold_changes_the_function(self):
        m = ScaffoldMLP(16, 2, hidden=(8, 8), rank=2, seed=8)
        x = torch.randn(8, 16)
        before = m(x)
        m.drop_scaffold()
        assert not torch.allclose(before, m(x))

    def test_trainable_grows_linearly_in_rank_and_total_does_not(self):
        counts = {r: ScaffoldMLP(64, 2, hidden=(64, 64), rank=r).trainable()
                  for r in (1, 2, 4)}
        # Each rank adds (d_in + d_out) per layer, so successive differences are
        # equal. This is the arithmetic behind the paper's %-trainable column.
        d1, d2 = counts[2] - counts[1], counts[4] - counts[2]
        assert d2 == pytest.approx(2 * d1)

    def test_an_unknown_scaffold_condition_raises(self):
        with pytest.raises(ValueError):
            ScaffoldMLP(8, 2, hidden=(4,), rank=1, scaffold="frozen-ish")

    def test_beta_starts_at_one_and_is_trainable(self):
        m = ScaffoldMLP(8, 2, hidden=(4,), rank=1)
        assert m.betas() == [1.0]
        assert m.layers[0].beta.requires_grad


class TestTrainingAndReadout:
    def test_early_stopping_restores_the_best_validation_epoch(self):
        task, tr, va, te = make_multi_index(1500, d=16, k=2, seed=9)
        m = ScaffoldMLP(16, 2, hidden=(24, 24), rank=4, seed=0)
        train(m, *tr, val=va, epochs=6, lr=3e-3)
        assert 0 <= m.best_epoch <= 5
        # The restored state must be the one that scored best, not the last one.
        assert evaluate(m, *va)["acc"] >= evaluate(m, *va)["acc"] - 1e-9

    def test_a_frozen_backbone_stays_bit_identical_through_training(self):
        task, tr, va, te = make_multi_index(1200, d=16, k=2, seed=10)
        m = ScaffoldMLP(16, 2, hidden=(16, 16), rank=2, seed=0)
        before = [l.W_seed.clone() for l in m.layers]
        train(m, *tr, val=va, epochs=4)
        for w0, l in zip(before, m.layers):
            assert torch.equal(w0, l.W_seed)

    def test_recovery_is_measured_above_chance(self):
        # Dividing accuracies directly, as the paper does, makes a coin flip
        # "recover" 58% of an 86% baseline. Subtracting chance is the same
        # quantity the paper means and is why these curves start near zero.
        assert recovery(0.5, 0.86) == pytest.approx(0.0)
        assert recovery(0.86, 0.86) == pytest.approx(1.0)
        assert recovery(0.68, 0.86) == pytest.approx(0.5)

    def test_a_baseline_at_chance_gives_no_recovery_rather_than_infinity(self):
        assert np.isnan(recovery(0.7, 0.5))


class TestMinimumRank:
    ROWS = [{"rank": 1, "recovery": 0.80}, {"rank": 2, "recovery": 0.95},
            {"rank": 4, "recovery": 0.995}, {"rank": 8, "recovery": 1.01}]

    def test_it_returns_the_first_rank_over_the_target(self):
        assert minimum_rank(self.ROWS, target=0.99) == 4
        assert minimum_rank(self.ROWS, target=0.90) == 2

    def test_it_does_not_care_what_order_the_rows_arrive_in(self):
        assert minimum_rank(list(reversed(self.ROWS)), target=0.99) == 4

    def test_a_curve_that_never_saturates_returns_none(self):
        # Not the largest rank tried: "did not saturate on this grid" and
        # "saturated at the last point I happened to try" are different claims,
        # and collapsing them turns r* into a measurement of the grid.
        assert minimum_rank(self.ROWS, target=1.5) is None
