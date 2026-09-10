r"""Where the derivative does not exist, and what autodiff returns instead.

`relu` is not differentiable at zero. Neither is `abs`, `clamp` at either
bound, `max`, `sign`, or a vector norm at the origin. Automatic
differentiation returns a float at all of them, and the float is a **choice**
made by whoever wrote the kernel.

The choices are not consistent with each other. Measured on torch:

    relu(x)          at 0   ->  0.0
    abs(x)           at 0   ->  0.0
    clamp(x, 0, 1)   at 0   ->  0.0
    norm(x)          at 0   ->  0.0
    maximum(x, 0)    at 0   ->  0.5     <- splits the tie instead
    minimum(x, 0)    at 0   ->  0.5
    max(stack(x, 0)) at 0   ->  0.5
    sqrt(x * x)      at 0   ->  nan     <- same function as abs(x)

Three different answers for one mathematical object, decided by which operation
you typed. Which has a consequence sharper than the inconsistency itself:
**autodiff is a function of the expression, not of the function it computes.**
The identity map, written three ways that agree at every point:

    f(x) = x                     ->  f'(0) = 1.0
    f(x) = relu(x) - relu(-x)    ->  f'(0) = 0.0
    f(x) = relu(x) + min(x, 0)   ->  f'(0) = 0.5

All three *are* the identity, differentiable everywhere with derivative one.
Two of the three return something else, and 0.0 is not even an element of the
subdifferential of the identity, which is the single point {1}. This is not a
bug report: it is the documented behaviour of composing chosen subgradients,
and Bolte and Pauwels' "conservative fields" are the formalism under which
the result is still usable for optimisation despite not being a subgradient.

**Then the part this module was written to find, and did not.** None of the
above happens in the transformer this series measures. `GELU` is smooth,
`LayerNorm` is smooth, and the attention softmax is smooth; the non-smoothness
in the training procedure -- gradient clipping, which is a `min` -- never
activates in a 600-step run, the largest gradient norm staying under the
threshold; and across 10.6 million unmasked attention probabilities, not one is
exactly 0 or exactly 1 in float32.

With exactly one exception, and the mask puts it there rather than training:
the attention row at position 0 is a softmax over a single element, so it is
the constant 1, so its gradient is **identically zero** -- 5,120 of 5,120 first
rows, every head, every layer, every sequence. That is not a kink and no choice
of subgradient reaches it. The function really is locally constant.

So the honest conclusion is that the kink stories are true, entertaining, and
almost irrelevant to a transformer; what actually stops gradient flow is
saturation, which is perfectly smooth, and that is the next episode.

References: Bolte and Pauwels, "A mathematical model for automatic
differentiation in machine learning", *NeurIPS* (2020), for why a chosen
subgradient composed through a network need not be a subgradient of anything;
Griewank and Walther, *Evaluating Derivatives* (2nd ed., 2008), for
non-smoothness in AD generally; Kakade and Lee, "Provably correct automatic
subdifferentiation for qualified programs", *NeurIPS* (2018), for when the
composition *is* correct.
"""

from __future__ import annotations

from dataclasses import dataclass


def _torch():
    import torch
    return torch


@dataclass(frozen=True)
class Kink:
    """One operation, one point, and what autodiff says the slope is there."""

    name: str
    point: float
    returned: float | str
    subdifferential: str
    #: True when `returned` lies in the subdifferential set. `nan` never does.
    admissible: bool


def _derivative(f, x0: float):
    torch = _torch()
    x = torch.tensor([float(x0)], requires_grad=True)
    f(x).sum().backward()
    return float(x.grad[0])


def catalogue() -> list[Kink]:
    """Every non-differentiable point this episode looks at, measured.

    Written as a measurement rather than a table of remembered values, because
    these are kernel choices and kernel choices change between versions. If a
    torch release moves one of them, this returns the new answer and the test
    that pins the *pattern* -- some return 0, some split the tie, one returns
    nan -- is what fails.
    """
    torch = _torch()
    F = torch.nn.functional
    zero = torch.zeros(1)

    specs = [
        ("relu(x)", lambda x: torch.relu(x), 0.0, "[0, 1]"),
        ("abs(x)", lambda x: x.abs(), 0.0, "[-1, 1]"),
        ("clamp(x, 0, 1)", lambda x: x.clamp(0.0, 1.0), 0.0, "[0, 1]"),
        ("clamp(x, 0, 1)", lambda x: x.clamp(0.0, 1.0), 1.0, "[0, 1]"),
        ("hardtanh(x)", lambda x: F.hardtanh(x), 1.0, "[0, 1]"),
        ("norm(x)", lambda x: torch.linalg.vector_norm(x), 0.0, "[-1, 1]"),
        ("maximum(x, 0)", lambda x: torch.maximum(x, zero), 0.0, "[0, 1]"),
        ("minimum(x, 0)", lambda x: torch.minimum(x, zero), 0.0, "[0, 1]"),
        ("max(stack(x, 0))", lambda x: torch.stack([x[0], zero[0]]).max(),
         0.0, "[0, 1]"),
        ("sqrt(x * x)", lambda x: torch.sqrt(x * x), 0.0, "[-1, 1]"),
    ]
    out = []
    for name, f, x0, sub in specs:
        got = _derivative(f, x0)
        lo, hi = (float(t) for t in sub.strip("[]").split(","))
        ok = got == got and lo <= got <= hi          # nan fails the first test
        out.append(Kink(name=name, point=float(x0), returned=got,
                        subdifferential=sub, admissible=bool(ok)))
    return out


#: The identity map, written three ways that agree at every real number.
IDENTITIES = ("x", "relu(x) - relu(-x)", "relu(x) + min(x, 0)")


def identity_three_ways(point: float = 0.0) -> list[dict]:
    """The same function, spelled three ways, differentiated at one point.

    The subdifferential of the identity is the single point {1}, so anything
    other than 1.0 here is not a subgradient of the function being computed --
    which is the whole claim, stated so that it can be checked rather than
    believed.
    """
    torch = _torch()
    zero = torch.zeros(1)
    forms = {
        "x": lambda x: x,
        "relu(x) - relu(-x)": lambda x: torch.relu(x) - torch.relu(-x),
        "relu(x) + min(x, 0)": lambda x: torch.relu(x)
        + torch.minimum(x, zero),
    }
    out = []
    for name, f in forms.items():
        value = float(f(torch.tensor([float(point)]))[0])
        got = _derivative(f, point)
        out.append({"form": name, "point": float(point), "value": value,
                    "derivative": got, "is_subgradient": got == 1.0})
    return out


def fully_masked_softmax() -> dict:
    """A softmax over an all -inf row, which is what a fully masked attention
    row is.

    The failure is in the **forward** pass, not the backward one: `exp` of
    -inf is 0 everywhere, the normaliser is 0, and 0/0 is nan. Worth knowing
    because the usual mental model puts NaNs in the gradient.
    """
    torch = _torch()
    F = torch.nn.functional
    logits = torch.tensor([[0.3, -0.2, 1.1]], requires_grad=True)
    p = F.softmax(logits.masked_fill(
        torch.ones(1, 3, dtype=torch.bool), float("-inf")), dim=-1)
    forward_nan = bool(torch.isnan(p).all())
    p.nan_to_num(0.0).sum().backward()
    return {"forward_all_nan": forward_nan,
            "gradient": [float(g) for g in logits.grad[0]]}


def attention_survey(*, count: int = 20, size: int = 16, seed: int = 0
                     ) -> dict:
    """Are any of the model's attention probabilities exactly 0 or exactly 1?

    Two answers, and they are different in kind. Away from position 0: none, in
    ten million of them, which is why the kink stories above do not describe
    this model. At position 0: all of them, because a softmax over one element
    is the constant 1, so the gradient there is identically zero and the mask
    rather than the training put it that way.
    """
    torch = _torch()
    import numpy as np

    from standarderror.llm import tiny

    bundle = tiny.load()
    model = bundle["model"]
    first_rows = first_onehot = 0
    exact_zero = exact_one = below_micro = live = 0
    maxima = []
    with torch.no_grad():
        for x, _ in tiny.batches(count=count, size=size, seed=seed):
            _, _, atts, _ = model(x, want_attn=True)
            for a in atts:
                T = a.shape[-1]
                first_rows += a.shape[0] * a.shape[1]
                first_onehot += int((a[:, :, 0, 0] == 1.0).sum())
                keep = torch.tril(torch.ones(T, T, dtype=torch.bool))
                keep[0] = False        # the structural row, counted separately
                p = a[..., keep]
                exact_zero += int((p == 0.0).sum())
                exact_one += int((p == 1.0).sum())
                below_micro += int((p < 1e-6).sum())
                live += p.numel()
                maxima.append(a[:, :, 1:, :].max(-1).values.reshape(-1)
                              .numpy())
    maxima = np.concatenate(maxima)
    return {
        "probabilities": int(live),
        "exact_zero": int(exact_zero),
        "exact_one": int(exact_one),
        "below_micro": int(below_micro),
        "below_micro_share": below_micro / live,
        "first_rows": int(first_rows),
        "first_rows_onehot": int(first_onehot),
        "median_max_p": float(np.median(maxima)),
        "share_above": {t: float((maxima > t).mean())
                        for t in (0.9, 0.99, 0.999, 0.9999)},
        "largest_max_p": float(maxima.max()),
    }


def clipping_survey(*, steps: int = 600, threshold: float = 1.0,
                    lr: float = 3e-3, batch: int = 32, seed: int = 0) -> dict:
    """How often gradient clipping's `min` is actually active during training.

    Clipping is the one unambiguous non-smoothness in the training procedure:
    the update is `g * min(1, c / ||g||)`, which has a kink at `||g|| = c`. It
    is also, on this model, never the branch taken -- which is the negative
    result this episode was written around.

    Trains a fresh model, so it costs a couple of minutes.
    """
    torch = _torch()
    import numpy as np

    from standarderror.llm import tiny

    torch.manual_seed(int(seed))
    text = tiny.corpus()
    chars = sorted(set(text))
    model = tiny.build(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=0.1)
    norms = []
    gen = tiny.batches("train", count=int(steps), size=int(batch), seed=seed)
    for x, y in gen:
        _, loss, _, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(threshold))))
        opt.step()
    norms = np.asarray(norms)
    clipped = norms > float(threshold)
    windows = []
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 400), (400, steps)):
        if hi <= len(norms):
            windows.append({"from": lo, "to": hi,
                            "clipped": float(clipped[lo:hi].mean()),
                            "median_norm": float(np.median(norms[lo:hi]))})
    return {"steps": int(steps), "threshold": float(threshold),
            "clipped": int(clipped.sum()),
            "clipped_share": float(clipped.mean()),
            "median_norm": float(np.median(norms)),
            "max_norm": float(norms.max()),
            "windows": windows}
