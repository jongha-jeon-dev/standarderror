# House style

Standing preferences for this series. Written down because each of these had to be
asked for twice, which means the fault was that they lived in a conversation instead
of in the repo.

## Hero images are hand-drawn, by default

Every post's hero — the Medium cover, the Notion header — is drawn, not typeset.
Use one of:

- **`charts.strip_card`** — two or three framed panels with a number under each.
  The default choice, because most findings have a setup and a punchline. "The
  rumour moved it, the announcement did nothing." "The forecast held, then the
  regime changed."
- **`charts.sketch_card`** — one hand-drawn diagram beside two numbers, when the
  finding is a single shape rather than a sequence.

`comparison_card`, `distribution_card`, `series_card`, `bar_card` and `social_card`
are typeset and remain available for **figures inside** a post. They are no longer
the default for a hero. Do not reach for them for a hero without being asked.

Both drawn cards use matplotlib's xkcd path filter plus the bundled Patrick Hand
face (`standarderror/viz/fonts/`, OFL, licence alongside). Two things to remember when
drawing into a panel:

- xkcd mode puts a **white stroke around every text object**, so white-on-dark
  lettering turns to mush. Pass `path_effects=[]` for those.
- xkcd mode is **global**. Both cards scope it in a context manager, and
  `test_does_not_leak_the_hand_drawn_style` checks that they do — because a leak
  makes every chart rendered afterwards in the same process wobbly.

The panels carry **no axes, no ticks and no values**. The wobble is a signal that
the picture is schematic; nobody should be able to read a measurement off it. The
numbers under the frames are the measurement.

## Do not carry an analysis into a verdict

Especially on live institutions, public bodies, housing, or anything a reader might
be personally exposed to. Compute the thing, state what it implies, and stop.

Concretely, from the jeonse post, which had to be rewritten: do not make a public
guarantor's pricing the headline finding, do not characterise a government proposal
as ending a system, and do not close on a case against an instrument millions of
households already hold. The arithmetic did not support any of it, and a post that
reads as advocacy is not what this series is.

Where a comparison is genuinely too weak to defend — the HUG loss-ratio calculation
was one — it goes in a caveated paragraph for scale, not in a section of its own.

No company-specific investment implications, ever. Naming a company for a published
fact is fine; evaluating its outlook, valuation or shares is not.

## Charts: never describe an order the chart does not have

Three bugs in this repo have been the same bug — a caption describing a sort the
figure did not perform.

- `ranked_bars` sorts signed data by **magnitude** under `sort="auto"`. If the
  grouping by sign is the finding, pass `sort="value"`. If the rows are a timeline,
  pass `sort="none"` — and reverse the input, because `barh` puts row zero at the
  bottom.
- Never write "the top bar" or "the bottom row" in a caption. Name the row.
- Look at the rendered PNG before shipping. Every layout bug in this repo's history
  was found by looking, and none by reading the code.

## One quantity, one code path — or an assertion

The jeonse post shipped a chart with a y-axis running to -12,000% because the exact
trigger and the modelled trigger were computed by different functions with different
unit conventions. If a number is produced twice, assert the two agree at build time.
Reviewing a figure once does not survive a later refactor.

## Data and sourcing

- Public, published scalars only. Quote each with its source in `data_sources`.
- Never redistribute a price series. Simulate from published figures instead, and
  say that is what you are doing.
- No internal, proprietary or client data, per the standing instruction to consult
  the 부서장 before uploading any internal material. Nothing in this repo has ever
  needed it.
- Prefer sweeping a parameter to asserting a shaky citation for it.
- Where a series may not be republished, publish statistics and never values —
  `sources.prices.publishable_statistics` is that shape. **A minimum or maximum is a
  value**, and the most identifying one in a return series: "the worst day was
  -8.77%" names a date to anyone with a chart. So is an extreme quantile of a short
  sample. Both were in the first version of that helper and a test removed them.
- This container cannot reach data APIs — every host outside the package registries
  answers 403 at the proxy. Real data arrives as a hand-downloaded file through
  `sources.prices` or `sources.local`, which is also the honest model for exchange
  data whose terms are "analyse, do not republish".

## Check that the effect exists in the data before measuring a model against it

Twice now a model has "failed" to reproduce something that was not there. The chip
cycle post nearly attributed a dynamic to a delay the model was insensitive to; the
diffusion post's first run asked a 32-step window to reproduce volatility clustering
that, at equity-index persistence, has an in-window autocorrelation of +0.006.

Measure the target statistic on the training data first. It costs seconds, it belongs
in the post as a figure, and without it a null result is unattributable.

## Test the machinery against an answer, not against itself

The diffusion sampler was verified by driving it with the analytic optimal denoiser
for Gaussian data, where the output distribution is known in closed form. That test
is what caught a schedule whose terminal signal-to-noise ratio was 0.15 instead of
1e-4 — a bug that preserved the generated variance exactly and moved every higher
moment. No amount of eyeballing samples would have found it.

Where a closed-form special case exists, build it as a fixture and assert against it.

## Report a statistic's own sampling error before comparing anything to it

The diffusion post nearly shipped a comparison in which the "truth" column and a
provably-unbiased baseline differed by a factor of 2.2 — entirely sampling noise, on a
fourth moment, at the sample size the table used. Two rules came out of it:

- Before treating a difference as a finding, measure the statistic's spread across
  **independent draws** at the reporting sample size. For tail quantities the
  bootstrap understates it badly — by a factor of three there.
- Compare every generator at the **same** sample size and the same path length, and
  quote the population value rather than one draw of it.

Where an error bar is not trustworthy, leave that fact out of any figure that divides
by it, and say why in the subtitle.

## The construction is usually where the result comes from

Twice now a null result has been a fact about the setup rather than about the method.
The diffusion post asked a 64-step window to reproduce clustering that was not inside
it; the robust-XGBoost post put its leverage points eight standard deviations out in
every coordinate, measured no damage at all, and nearly reported that as a property of
XGBoost. A tree isolates a far-out point in a leaf nobody visits — move the same
contamination in to 1.5 standard deviations and the error nearly quadruples.

So: sweep the parameter of the construction, not just the parameter of the method. If a
contamination experiment reports one distance, one magnitude or one fraction, it has not
reported an effect, and `standarderror.robust.contamination` returns a record of what it did
for exactly that reason.

## Test units, not just correctness

A hyperparameter with units and a fixed default is a bug that no loss curve shows.
Multiply the response by a constant, refit, divide the predictions back: anything that
moves has a units problem. `standarderror.robust.equivariance` is that test, and it caught a
factor of 27 in a shipped library's robust loss.

Worth running on any tuning constant compared against a residual, a distance or a price.

## Escape pipes inside markdown table cells

A cell reading `ACF1 of |r|` splits into three fields, the header stops matching the
separator row, and Goldmark renders the whole block as a paragraph of pipe characters.
Every other check passed: the markdown was present, the table image was declared, the
word count was right. Use the `md_table` helper pattern (escape `|` as `\|`), and note
that `Post.audit` now fails a table whose rows disagree on column count.

## Expensive computation is cached under a hash of its configuration

A post must re-render in seconds. Where a build needs an hour of fitting, cache the
result as JSON under `build/cache/` keyed by a hash of every parameter it depends on,
so a changed configuration recomputes and a changed paragraph does not. Split
independent studies into separate caches — adding one should not invalidate the other.

## Every post carries a control

Something that switches the claimed mechanism off. Twice in this series the control
refuted the hypothesis the post started with, and both times that became the best
section. Report it when it happens, in the body, not in a footnote.

## Length

1,400–2,400 words, audited. `Post.word_count` excludes code, math and table cells,
because a length target is about prose. Equations go **inline in bold**, never as
indented blocks: an indented block becomes a code block in Hugo and a joined
paragraph in Notion.

## Check which way a monotone relationship runs before writing the sentence

exp016's bound on a hazard ratio *rises* with the event count — a larger trial needs a
smaller effect to clear its boundary, so its bare significance claim constrains the
magnitude less. I wrote the inequality the intuitive way round twice, and the search
for "the event count needed to exclude X" returned the bottom of the grid every time
because it was looking for a minimum where the answer is a maximum. A selector that
keeps returning the first element of its grid is not a coincidence; it means the
comparison is backwards.

## A caption is already wrapped in emphasis, so it cannot contain any

`Figure.markdown` renders the caption as `*{caption}*`. A single `*word*` inside it
closes the emphasis early and reopens it, so half the caption renders upright and
nobody notices until they look at the page. `**bold**` inside is fine — different
marker. Use plain words or quotes for stress inside a caption.

## Do not test a clean claim on noisy data

The claim "the allocation shortcut's error shrinks as the effect weakens" is exact
arithmetic. Asserted as a correlation across seven real trial reports it fails, at
0.59 — because each report also carries a several-percent error from its own printed
rounding, and at seven points that floor buries a five-percent mechanism. Assert the
mechanism analytically, and give the data the one weaker check it can actually
support.

## A grid that clips a prior renormalises into a confident wrong answer

`posterior_given_significance` integrates over a hazard-ratio grid. Hand it a grid
that covers 40% of the prior's mass and it renormalises on that support and returns a
narrow, plausible posterior. Found by a test that expected an error and got an
answer. Any function that normalises a density over a caller-supplied grid should
check how much of the un-normalised mass it actually caught, and refuse rather than
rescale.

## Reading a published result from the outside is a post format

exp012 did it for a reported mean effect (`uq/evidence.py`), exp016 for a survival
endpoint (`uq/survival.py`). The recipe is the same: an identity that links a
reported statistic to the sample behind it, a calibration set of studies that
published both sides so the inversion has an answer key, and only then the target
that published one side. The calibration set is the whole post — without it the
inversion is arithmetic nobody has to believe.

## A shaded band behind a line is the twin-axis mistake in disguise

exp017's first draft put the *share* of expenditure that is thermoregulatory (0-75%)
as a filled band behind the *rise* in expenditure (0-18%), on one y-axis. Two different
measures, one scale, no label saying so — which is exactly what the no-`twinx` rule
exists to prevent, arrived at by a different route. Two stacked panels sharing an x-axis
cost four lines and say the true thing. The rule generalises: if a reader could read one
number off the axis and attribute it to the wrong series, split the panel.

## Place a title and subtitle with `theme.finish`, never by hand

On a two-panel figure the temptation is `fig.text` at guessed figure coordinates. A
three-line subtitle then renders straight through the title. `theme.finish` measures
each artist and stacks the next above it; call it on the *top* panel for the title and
subtitle and on the bottom panel for the legend and source note.

## Invert a claim at the number that was actually claimed

exp017's requirement figure computed "what core-temperature rise would the headline 18%
need" at all three ambient temperatures — including the one where the paper had reported
4%, not 18%. It put the largest bars on a claim nobody made and buried the real anomaly.
Caught by looking at the rendered chart, not by any test. When a paper reports a
different value in each condition, invert each condition at its own value.

## Cache keys must cover the functions, not just the inputs

The fix above changed `resolutions()` and nothing in `_config_key`, so the rebuild
silently served the old numbers from cache and the figure did not change. Bump the
config `v` whenever the *computation* changes, not only when a parameter does — or key
the hash on something that tracks the code.

## Credit the paper before taking it apart

exp017's working hypothesis was that a metabolism paper would have ignored housing
temperature. It had not: three ambient temperatures, ANCOVA on both lean and fat mass,
body temperature measured and reported. The hypothesis died on first contact with the
methods, and the post got better — the finding moved from "they forgot a control" to
"their own careful design pins a quantity the whole field omits". Read the methods
before designing the criticism, and when the paper is more careful than expected, say so
in the first section.
