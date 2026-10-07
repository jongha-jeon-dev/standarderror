"""Published time-series models behind the same interface as ours.

Kept in its own file because it needs the optional `chronos-forecasting`
package. It used to sit at the bottom of `test_tsfm.py` behind a module-level
`importorskip`, which skipped *every* test in that file -- all 49 -- whenever
the package was missing, not just these four.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("statsmodels")
pytest.importorskip("chronos")
from standarderror.tsfm import compress as cp  # noqa: E402
from standarderror.tsfm import evaluate as ev  # noqa: E402
from standarderror.tsfm import external as ex  # noqa: E402
from standarderror.tsfm import model as tm  # noqa: E402
from standarderror.tsfm import pool as tp  # noqa: E402


class TestPublishedModelAdapter:
    """Against a randomly initialised Chronos-Bolt built from the real
    classes, because the real weights are not reachable from here."""

    @pytest.fixture(scope="class")
    def bolt(self):
        return ex.tiny_bolt()

    def test_it_returns_our_nine_levels_sorted(self, bolt):
        q = ex.forecaster(bolt)([np.arange(40.0), np.arange(200.0)], 8, 1)
        assert q.shape == (2, 8, len(tm.QUANTILES))
        assert (np.diff(q, axis=-1) >= 0).all()

    def test_the_compressors_reach_its_weights(self, bolt):
        names = [n for n, _ in cp._targets(bolt["model"])]
        assert any("input_patch_embedding" in n for n in names)
        assert any("encoder" in n for n in names)

    def test_a_compressed_copy_scores_through_the_same_harness(self, bolt):
        small = ex.with_model(bolt, cp.quantize(bolt["model"], 4))
        series = [s for s in tp.pool() if s.freq == "Q"][:2]
        rows = ev.evaluate(ex.forecaster(small), series)
        assert rows and all(np.isfinite(r["coverage"]) for r in rows)
        assert small["model"] is not bolt["model"]

    def test_unreachable_weights_fail_with_a_plain_message(self):
        try:
            ex.load("chronos-bolt-tiny")
        except ex.Unreachable as e:
            assert "huggingface" in str(e).lower()
        else:
            pytest.skip("weights reachable here; nothing to check")


