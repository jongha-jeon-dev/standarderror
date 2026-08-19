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
face (`quantpost/viz/fonts/`, OFL, licence alongside). Two things to remember when
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

## Every post carries a control

Something that switches the claimed mechanism off. Twice in this series the control
refuted the hypothesis the post started with, and both times that became the best
section. Report it when it happens, in the body, not in a footnote.

## Length

1,400–2,400 words, audited. `Post.word_count` excludes code, math and table cells,
because a length target is about prose. Equations go **inline in bold**, never as
indented blocks: an indented block becomes a code block in Hugo and a joined
paragraph in Notion.
