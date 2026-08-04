# quantpost

A pipeline for research-grade blog posts on financial ML: public data in, a
published page out — with the parts that usually get skipped made mandatory
rather than optional.

The design bias throughout is **make the honest thing the easy thing**.
Baselines are first-class objects, the split helpers physically cannot leak the
future, every fetch records provenance, and `Post.audit()` refuses to publish a
post whose figures have no alt text or whose forecasting claim never mentions
persistence.

```bash
pip install -e ".[dev]"
cp .env.example .env          # optional keys; most sources need none
quantpost doctor              # what works, what needs a key
quantpost run exp001_chaos_horizon --publish --medium
```

## Layout

| Package | What is in it |
|---|---|
| `quantpost.sources` | FRED, ECB, Bank of Korea ECOS, BIS, HMDA, market, local-file adapters |
| `quantpost.dynamics` | ODE / PDE / SDE generators with known ground truth, Lyapunov tooling |
| `quantpost.models` | Echo state networks, NG-RC, and the baselines you are obliged to beat |
| `quantpost.xai` | Attribution methods + reservoir-specific probes |
| `quantpost.viz` | One accessibility-validated chart style, light and dark |
| `quantpost.render` | `Post` → Hugo page, Medium crosspost, Notion page |
| `experiments/` | One file per post: `build() -> Post` |
| `site/` | Hugo scaffold and the GitHub Pages workflow |

## Data sources

Verified request specs as of August 2026. Most need no key.

| Source | Key | Notes |
|---|---|---|
| **FRED** | optional | `fred.get()` uses the keyless `fredgraph.csv` endpoint. A key unlocks the documented API and **vintages** (`realtime_start`) — which any honest macro backtest needs. Attribution line is mandatory. |
| **ECB Data Portal** | none | `data-api.ecb.europa.eu`; the old `sdw-wsrest` host is decommissioned. Yield curves, CISS. |
| **BIS Data Portal** | none | SDMX v2 at `stats.bis.org/api/v2`. Credit-to-GDP gap, property prices, plus bulk CSVs. |
| **Bank of Korea ECOS** | required | Key goes in the URL *path*. The literal key `sample` works for 10 rows without registering. |
| **HMDA (CFPB)** | none | Aggregations endpoint first — the CSV endpoints stream hundreds of MB with no pagination. |
| **Stooq / yfinance** | none | Both off the happy path. `stooq()` requires `accept_terms=True` because the endpoint is `robots.txt`-disallowed; yfinance is an unofficial scrape that rate-limits. Prefer FRED where it has the series. |
| **Freddie Mac SF loan-level** | login | No API. `local.load_freddie()` reads manually downloaded files and carries the licence obligation on the frame. |

Two licence traps the adapters surface automatically via
`sources.licence_warnings(df)`:

- ICE BofA OAS series on FRED (`BAMLH0A0HYM2`, `BAMLC0A4CBBB`) were cut to a
  rolling **3-year** window in April 2026 and are **internal-use only** — you
  cannot republish those values, and you can no longer get GFC-era credit
  spreads from FRED at all.
- Stooq and yfinance data carry no redistribution rights.

## Two bugs this repo exists to have already fixed

Both are silent, both look fine for a while, and both would have quietly
corrupted every number downstream. They are documented at length in the modules
because the debugging is more instructive than the fix.

**The Kuramoto–Sivashinsky integrator blew up at t ≈ 355** for every domain
length, grid size, timestep, and scheme — ETDRK4 and semi-implicit CNAB2 alike.
The tell was that the blow-up time never moved: it is set by the KS maximum
linear growth rate `max_k(k² − k⁴) = 0.25` and nothing else. Holding the state as
a full complex spectrum gives `2N` real degrees of freedom for an `N`-point real
field; the redundant non-Hermitian half is invisible to `real(ifft(v))`, so the
nonlinear term never constrains it while the linear operator amplifies it from
roundoff. `rfft` makes those modes unrepresentable. The same integration now runs
to `t = 20000`, and `test_dynamics.py` pins it.

**The Lyapunov exponent was 20% too high** — 1.087 against the literature's
0.9056 — because the tangent space was propagated with an Euler step. Using
`expm(J·dt)` gives 0.8965 and a Kaplan–Yorke dimension of 2.061 against 2.062.
Since every forecast horizon in the repo is quoted in Lyapunov times, that error
would have rescaled every headline number.

## Publishing

Own your archive, then syndicate:

1. `publish.hugo_page_bundle(post)` writes `site/content/posts/<slug>/index.md`
   with the images beside it. Push; the included Action deploys to GitHub Pages.
2. `publish.medium_bundle(post)` writes markdown with **absolute** image URLs and
   a canonical note. Import it at `medium.com/p/import`, which sets
   `rel="canonical"` back to your site, so search engines credit your domain.

`medium_bundle` refuses to run without a canonical URL. That is the whole point of
owning the site.

## Testing

```bash
pytest                                     # ~105 tests, no network
QUANTPOST_NETWORK_TESTS=1 pytest -m network # live API smoke tests
```

The tests check physics and statistics, not shapes: the Lyapunov spectrum against
the exact trace identity, KS against an independent implicit solver, Hawkes
intensity against `μ/(1−α/β)`, Heston's leverage sign, fBm's Hurst scaling, NG-RC
recovering the Lorenz right-hand side, and — the one most worth copying — an ESN
that **must not** beat persistence on white noise. That last test is the leakage
canary.

## Licence

MIT for the code. Data licences belong to their providers; see the table above and
check `sources.licence_warnings()` before you publish.
