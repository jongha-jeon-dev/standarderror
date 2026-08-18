# Bundled fonts

`PatrickHand-Regular.ttf` — SIL Open Font License 1.1, copyright Patrick
Wagesreiter. Licence text in `OFL-PatrickHand.txt`, which the OFL requires to
travel with the file.

It is here so that `charts.sketch_card` renders identically wherever the repo is
checked out. Everything else in `quantpost.viz` uses the system sans; this is the
only bundled face, and only the hand-drawn card uses it. If it is missing the card
falls back to the system sans and still renders — it just stops looking drawn.

Converted from the `@fontsource/patrick-hand` woff2 with
`fontTools.ttLib.woff2.decompress`, because matplotlib cannot read woff2.
