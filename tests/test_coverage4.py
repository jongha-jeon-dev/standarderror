"""Coverage episode 4: calibration across training runs.

The trajectory file is what the episode quotes, so these tests do not take it
on trust: they reload the committed checkpoints, recompute the endpoints, and
check the file against them. The claims are pinned as inequalities, because
they are properties of these runs rather than theorems.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from standarderror.llm import tiny  # noqa: E402
from standarderror.uncertainty import trajectory as tj  # noqa: E402

ARMS = ("recipe", "long", "small")


@pytest.fixture(scope="module")
def traj():
    return tj.trajectory()


def _rows(traj, arm):
    return [r for r in traj["arms"][arm] if r["step"] > 0]


def _at(traj, arm, step):
    return next(r for r in traj["arms"][arm] if r["step"] == step)


class TestTheFileMatchesTheWeights:

    @pytest.mark.parametrize("name", ["recipe_final", "long_best",
                                      "long_final", "small_best",
                                      "small_final"])
    def test_a_committed_checkpoint_reproduces_its_row(self, traj, name):
        arm, _ = name.rsplit("_", 1)
        step = traj["checkpoints"][name]["step"]
        share = traj["config"]["arms"][arm]["share"]
        got = tj.evaluate(tj.load_checkpoint(name), share=share)
        row = _at(traj, arm, step)
        for split in ("val", "train"):
            assert got[split]["nll"] == pytest.approx(row[split]["nll"],
                                                      abs=1e-4)
            assert got[split]["fitted_t"] == pytest.approx(
                row[split]["fitted_t"], abs=2e-3)

    def test_a_swapped_checkpoint_is_refused(self, traj, tmp_path,
                                             monkeypatch):
        import shutil
        src = tj.data_dir()
        for f in src.iterdir():
            shutil.copy(f, tmp_path / f.name)
        shutil.copy(src / "long_final.pt", tmp_path / "small_final.pt")
        monkeypatch.setattr(tj, "data_dir", lambda: tmp_path)
        with pytest.raises(ValueError):
            tj.load_checkpoint("small_final")


class TestOverconfidenceArrivesEarly:

    def test_the_rerun_recipe_is_a_sibling_of_the_committed_model(self, traj):
        committed = tj.evaluate(tiny.load()["model"])["val"]
        rerun = _rows(traj, "recipe")[-1]["val"]
        assert rerun["nll"] == pytest.approx(committed["nll"], abs=0.02)
        assert rerun["fitted_t"] == pytest.approx(committed["fitted_t"],
                                                  abs=0.02)

    @pytest.mark.parametrize("arm", ARMS)
    def test_every_run_starts_underconfident(self, traj, arm):
        assert min(r["val"]["fitted_t"] for r in _rows(traj, arm)[:4]) < 1.0

    @pytest.mark.parametrize("arm", ["long", "small", "small_seed1"])
    def test_it_is_overconfident_before_the_minimum(self, traj, arm):
        rows = _rows(traj, arm)
        best = tj.turn(rows)
        onset = next(r for r in rows if r["val"]["fitted_t"] >= 1.05)
        assert onset["step"] < best["step"]
        assert _at(traj, arm, best["step"])["val"]["fitted_t"] > 1.15

    @pytest.mark.parametrize("arm", ARMS + ("small_seed1",))
    def test_but_never_on_its_own_training_text(self, traj, arm):
        assert max(r["train"]["fitted_t"] for r in _rows(traj, arm)) < 1.03


class TestPastTheMinimum:

    def test_the_loss_passes_the_uniform_guess_and_accuracy_does_not_move(
            self, traj):
        rows = _rows(traj, "small")
        best = _at(traj, "small", tj.turn(rows)["step"])["val"]
        end = rows[-1]["val"]
        assert end["nll"] > tiny.UNIFORM_LOSS
        assert abs(end["accuracy"] - best["accuracy"]) < 0.03

    def test_one_temperature_removes_most_of_it(self, traj):
        rows = _rows(traj, "small")
        end, best = rows[-1]["val"], tj.turn(rows)["value"]
        assert (end["nll"] - end["nll_at_fit"]) > 0.8 * (end["nll"] - best)

    def test_selecting_on_the_scaled_loss_picks_a_later_checkpoint(self, traj):
        for arm in ("long", "small"):
            rows = _rows(traj, arm)
            raw = min(rows, key=lambda r: r["val"]["nll"])
            cal = min(rows, key=lambda r: r["val"]["nll_at_fit"])
            assert cal["step"] > raw["step"]
            assert cal["val"]["nll_at_fit"] < raw["val"]["nll_at_fit"]
