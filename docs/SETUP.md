# Setup: single public repo, drafts stay drafts

One repo holds the code, the site and the posts. Unfinished posts live in it as
`draft: true` — their source is visible on GitHub, but they are absent from the
built site. Publishing is one flag.

The script does all of this. Read it if you prefer, it is ~200 lines of bash and
every step is guarded:

```bash
./scripts/setup_github.sh --check     # verify only, changes nothing
./scripts/setup_github.sh            # do it
```

It is idempotent — re-running after a failure is safe, nothing is force-pushed,
nothing is deleted. If you would rather do it by hand, the manual equivalent of
each step is below.

---

## 0. Prerequisites

| Tool | Needed for | Install |
|---|---|---|
| git | everything | already have it |
| Python ≥ 3.10 | the pipeline | already have it |
| **gh** (GitHub CLI) | creating the repo, enabling Pages | `brew install gh` / `apt install gh` / `winget install GitHub.cli` |
| **hugo-extended ≥ 0.112** | *local preview only* | `brew install hugo` / `apt install hugo` |

`gh` and `hugo` are both optional. Without `gh` you create the repo in the browser;
without `hugo` you lose local preview but CI still builds and deploys.

```bash
gh auth login          # once, if you have gh
```

---

## 1. Configure

```bash
git clone <this-repo> standarderror && cd standarderror
GH_USER=jongha-jeon-dev REPO=standarderror ./scripts/setup_github.sh
```

Three things must agree, and this is the single most common way the site ends up
broken:

| Setting | Where | Value |
|---|---|---|
| `baseURL` | `site/hugo.toml` | `https://<user>.github.io/<repo>/` — **trailing slash** |
| `SITE_BASE_URL` | `.env` | the same, **without** the trailing slash |
| `CODE_REPO_URL` | `.env` | `https://github.com/<user>/<repo>` |

`baseURL` needs the trailing slash or every asset 404s under the `/<repo>/` path
prefix. `SITE_BASE_URL` is what `medium_bundle()` builds absolute image URLs from.
`Post.audit()` rejects placeholder URLs, so a forgotten `YOURNAME` blocks
publication rather than shipping.

Manual equivalent:

```bash
cp .env.example .env && ${EDITOR:-nano} .env
${EDITOR:-nano} site/hugo.toml       # baseURL
```

---

## 2. Theme

```bash
git submodule add https://github.com/adityatelange/hugo-PaperMod.git \
    site/themes/PaperMod
git -C site/themes/PaperMod checkout v8.0
```

**Pin the tag.** PaperMod's `master` has already raised its minimum to Hugo
0.146, newer than most distributions package — tracking it means the site can
stop building because of a commit you never made. `v8.0` builds on Hugo 0.112+
and is verified here against 0.123.7. To move later:

```bash
git -C site/themes/PaperMod fetch --tags
git -C site/themes/PaperMod checkout v9.0     # then commit the submodule bump
```

`.github/workflows/pages.yml` already checks out submodules recursively.

---

## 3. Repo and Pages

```bash
python -m pip install -e ".[dev]"
pytest                                  # ~110 tests, no network
git branch -M main                      # the Pages workflow triggers on main
gh repo create jongha-jeon-dev/standarderror --public --source=. --remote=origin
git push -u origin main
gh api -X POST repos/jongha-jeon-dev/standarderror/pages -f build_type=workflow
```

Without `gh`: create the repo at <https://github.com/new> (public, **no** README
or .gitignore — you have both), then

```bash
git remote add origin https://github.com/jongha-jeon-dev/standarderror.git
git push -u origin main
```

and set **Settings → Pages → Source: GitHub Actions**. `build_type=workflow` is
the part people miss; the default tries to build Jekyll from a branch and ignores
the uploaded artifact.

First deploy takes a few minutes. Watch it with `gh run watch` or on the Actions
tab.

---

## 4. The writing loop

This is the part you repeat.

```bash
# 1. Draft. Writes site/content/posts/<slug>/index.md with draft: true.
standarderror run exp001_chaos_horizon --publish

# 2. Preview. -D shows drafts; without it the post is correctly invisible.
make serve                     # http://localhost:1313

# 3. Commit the draft freely — it cannot reach readers.
git add -A && git commit -m "Draft: how far ahead can you forecast chaos"
git push

# 4. Publish when you are happy.
standarderror run exp001_chaos_horizon --publish --live
git add -A && git commit -m "Publish: how far ahead can you forecast chaos"
git push                       # Action deploys in ~2 minutes

# 5. Crosspost, only after the site is live.
standarderror run exp001_chaos_horizon --publish --live --medium
# then import build/medium/<slug>.md at https://medium.com/p/import
```

Why step 5 comes last: Medium's importer fetches images from your live site by
absolute URL. `medium_bundle()` refuses to run on a draft for that reason —
building it early hands you URLs that 404.

`--live` is required for `--medium`; `standarderror run ... --publish --medium` on a
draft tells you so and skips the bundle rather than writing a broken one.

### What is committed

Committed: `site/content/posts/<slug>/index.md` and the figures beside it (Hugo
page bundles keep them together, so a slug change never breaks an image).

Ignored: `build/` (regenerable), `.cache/` (fetched data), `.env` (your keys).

The figures **are** in git. They are the published artefact and they need to be
in the deploy; a few hundred KB per post is the right trade.

---

## 5. Verifying before you push

```bash
make test                                       # the suite
standarderror audit build/*.manifest.json           # re-check a written post
cd site && hugo --gc --minify                   # does it actually build?
```

`Post.audit()` runs automatically inside `standarderror run` and returns a non-zero
exit code on any problem, so `--publish` cannot ship a post with an untitled
figure, missing alt text, a leftover TODO, a placeholder URL, or a forecasting
claim that never mentions a baseline. `docs/../references/audit-checklist.md` in
the skill has the judgement-call half of the list.

---

## 6. Recurring maintenance

```bash
git submodule update --remote --merge     # only if you want a newer theme; re-pin
python -m pip install -e ".[dev]" -U      # dependency refresh, then run pytest
```

CI runs `ruff` and `pytest` on every push (`.github/workflows/tests.yml`) with no
network access — the adapter tests use recorded payloads, so a provider outage can
never turn your build red. To check the live APIs deliberately:

```bash
SERR_NETWORK_TESTS=1 pytest -m network
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Site deploys but every image and stylesheet 404s | `baseURL` missing the trailing slash, or not matching `/<repo>/` |
| Post not on the site, no error | still `draft: true`. Use `--live`, or `hugo server -D` to confirm |
| Action succeeds, site shows the README | Pages source is not "GitHub Actions" (`build_type=workflow`) |
| `hugo v0.146.0 or greater is required` | PaperMod moved to master; re-pin: `git -C site/themes/PaperMod checkout v8.0` |
| `partial "head.html" not found` | the submodule is empty: `git submodule update --init --recursive` |
| Nothing deploys on push | you are on `master`; the workflow triggers on `main` |
| Medium import shows broken images | site not live yet, or `SITE_BASE_URL` wrong |
| `standarderror run` writes into the wrong clone | run it from inside the repo you mean, or set `SERR_ROOT` |
| `audit FAILED: placeholder 'YOURNAME'` | working as designed — set `CODE_REPO_URL` in `.env` |

## A note on what goes in a public repo

Everything here uses public sources only. Before drafting anything that touches
your organisation's data or analysis — the Korean household-debt post in the
backlog is the obvious case — clear the topic with your 부서장 first. The repo
being public is precisely why that matters: `git push` is not reversible in any
way that counts.
