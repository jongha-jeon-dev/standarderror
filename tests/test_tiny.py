"""The committed model, checked to be the one the published numbers used.

These are not tests of the model's quality -- there is nothing to assert about
a validation loss of 1.573 beyond that it is what the checkpoint achieved. They
are tests that an episode measuring `softmax` Jacobians inside *this* network is
measuring the network the prose describes.
"""

import pytest

torch = pytest.importorskip("torch")

from standarderror.llm import tiny  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return tiny.load()


def test_the_checkpoint_is_the_published_one(bundle):
    """`load` hashes the file, so reaching this line already proves it. Stated
    as a test anyway, because a silently swapped checkpoint is the failure that
    would invalidate three episodes without breaking anything."""
    assert bundle["parameters"] == 816_128
    assert bundle["vocab"] == 65


def test_the_hash_is_actually_checked():
    import hashlib
    got = hashlib.sha256(tiny.checkpoint_path().read_bytes()).hexdigest()
    assert got == tiny.CHECKPOINT_SHA256


def test_it_is_a_language_model_rather_than_a_guess(bundle):
    """Loss on held-out-looking text well below `ln(65)`. A weak assertion on
    purpose: the point is that the weights are trained, not that they are good."""
    model, stoi = bundle["model"], bundle["stoi"]
    text = ("To be, or not to be, that is the question:\n"
            "Whether tis nobler in the mind to suffer")[:tiny.BLOCK + 1]
    ids = torch.tensor([[stoi[c] for c in text]])
    with torch.no_grad():
        _, loss, _, _ = model(ids[:, :-1], ids[:, 1:])
    assert float(loss) < 0.6 * tiny.UNIFORM_LOSS, float(loss)


def test_it_hands_back_the_internals_the_episodes_need(bundle):
    model, stoi = bundle["model"], bundle["stoi"]
    ids = torch.tensor([[stoi[c] for c in "ROMEO: what light"]])
    with torch.no_grad():
        logits, _, atts, hidden = model(ids, want_attn=True, want_hidden=True)
    T = ids.shape[1]
    assert logits.shape == (1, T, bundle["vocab"])
    assert len(atts) == tiny.LAYERS and len(hidden) == tiny.LAYERS
    for a in atts:
        assert a.shape == (1, tiny.HEADS, T, T)
        # rows are probability distributions, and causally masked
        assert torch.allclose(a.sum(-1), torch.ones(1, tiny.HEADS, T),
                              atol=1e-5)
        assert float(a[0, 0].triu(1).abs().max()) == 0.0
    for h in hidden:
        assert h.shape == (1, T, tiny.WIDTH)


def test_a_missing_checkpoint_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setattr(tiny, "checkpoint_path", lambda: tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="committed to this repository"):
        tiny.load()
