#!/usr/bin/env bash
#
# One-time setup: put this repo on GitHub, wire up Hugo + Pages, publish.
#
#   ./scripts/setup_github.sh                       # interactive, safe defaults
#   GH_USER=jongha-jeon-dev REPO=standarderror ./scripts/setup_github.sh
#   ./scripts/setup_github.sh --check               # verify only, change nothing
#
# Idempotent: every step checks whether it is already done and skips if so, so
# re-running after a failure is safe. Nothing is force-pushed and nothing is
# deleted.
#
set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true)
red=$(tput setaf 1 2>/dev/null || true); grn=$(tput setaf 2 2>/dev/null || true)
ylw=$(tput setaf 3 2>/dev/null || true); rst=$(tput sgr0 2>/dev/null || true)

step() { echo; echo "${bold}==> $*${rst}"; }
ok()   { echo "  ${grn}ok${rst}    $*"; }
warn() { echo "  ${ylw}warn${rst}  $*"; }
die()  { echo "  ${red}fail${rst}  $*" >&2; exit 1; }
skip() { echo "  ${dim}skip${rst}  $*"; }
run()  { if [[ $CHECK_ONLY -eq 1 ]]; then echo "  ${dim}would run:${rst} $*";
         else eval "$@"; fi }

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

# ---------------------------------------------------------------- 1. tooling

step "1. Checking tools"
command -v git >/dev/null || die "git not found"
ok "git $(git --version | awk '{print $3}')"

command -v python3 >/dev/null || die "python3 not found"
ok "python $(python3 -V | awk '{print $2}')"

if command -v gh >/dev/null; then
  ok "gh $(gh --version | head -1 | awk '{print $3}')"
  if gh auth status >/dev/null 2>&1; then
    GH_READY=1; ok "gh is authenticated"
  else
    GH_READY=0
    warn "gh is installed but not logged in — run: gh auth login"
  fi
else
  GH_READY=0
  warn "gh (GitHub CLI) not found. Install it to automate repo + Pages setup:"
  echo "        macOS:  brew install gh"
  echo "        Ubuntu: sudo apt install gh"
  echo "        Windows: winget install GitHub.cli"
  echo "        Without it, create the repo by hand — see docs/SETUP.md step 3."
fi

HUGO_MIN="0.112.0"       # floor for PaperMod v8.0, the tag pinned below
if command -v hugo >/dev/null; then
  HUGO_VER=$(hugo version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  ok "hugo $HUGO_VER"
  if [[ "$(printf '%s\n%s\n' "$HUGO_MIN" "$HUGO_VER" | sort -V | head -1)" \
        != "$HUGO_MIN" ]]; then
    warn "hugo $HUGO_VER is older than $HUGO_MIN, which PaperMod v8.0 needs."
  fi
  hugo version | grep -q extended || warn \
    "this is not hugo-extended, which PaperMod needs for SCSS. CI uses extended,
        so deploys still work — only local preview breaks. brew install hugo
        gives you extended; on apt, 'hugo' in Ubuntu 24.04+ is extended."
else
  warn "hugo not found — needed only for local preview (make serve). CI builds
        the site regardless. Install: brew install hugo / apt install hugo"
fi

# ---------------------------------------------------------------- 2. identity

step "2. Repo identity"
GH_USER="${GH_USER:-}"
if [[ -z "$GH_USER" && $GH_READY -eq 1 ]]; then
  GH_USER=$(gh api user --jq .login 2>/dev/null || true)
fi
if [[ -z "$GH_USER" ]]; then
  read -rp "  GitHub username: " GH_USER
fi
[[ -n "$GH_USER" ]] || die "a GitHub username is required"
REPO="${REPO:-standarderror}"
ok "will use github.com/$GH_USER/$REPO"

SITE_URL="https://${GH_USER}.github.io/${REPO}/"
CODE_URL="https://github.com/${GH_USER}/${REPO}"
ok "site  $SITE_URL"

# ---------------------------------------------------------------- 3. config

step "3. Writing config that depends on your username"

# Portability: no `sed -i` (BSD sed on macOS needs `-i ''`, GNU does not) and no
# eval around a heredoc. Write to a temp file and move it.
rewrite() {  # rewrite <file> <sed-expr>...
  local f="$1"; shift
  local tmp; tmp=$(mktemp)
  sed "$@" "$f" > "$tmp" && mv "$tmp" "$f"
}

# hugo.toml baseURL must match the Pages URL exactly, trailing slash included, or
# every asset 404s under the /$REPO/ path prefix.
if grep -qF "baseURL = \"$SITE_URL\"" site/hugo.toml 2>/dev/null; then
  skip "site/hugo.toml baseURL already correct"
elif [[ $CHECK_ONLY -eq 1 ]]; then
  echo "  ${dim}would set${rst} site/hugo.toml baseURL -> $SITE_URL"
else
  rewrite site/hugo.toml \
    -e "s|^baseURL = .*|baseURL = \"$SITE_URL\"|" \
    -e "s|https://github.com/YOURNAME/standarderror|$CODE_URL|g" \
    -e "s|^  name = \"Jongha Jeon\"|  name = \"${POST_AUTHOR:-Jongha Jeon}\"|"
  ok "site/hugo.toml baseURL -> $SITE_URL"
fi

if [[ -f .env ]]; then
  skip ".env exists (not overwritten) — check SITE_BASE_URL and CODE_REPO_URL"
elif [[ $CHECK_ONLY -eq 1 ]]; then
  echo "  ${dim}would create${rst} .env from .env.example with your URLs"
else
  sed -e "s|^SITE_BASE_URL=.*|SITE_BASE_URL=${SITE_URL%/}|" \
      -e "s|^CODE_REPO_URL=.*|CODE_REPO_URL=$CODE_URL|" \
      .env.example > .env
  ok ".env created from .env.example with your URLs"
fi

# ---------------------------------------------------------------- 4. theme

step "4. Hugo theme (PaperMod, pinned)"
# Pinned to a tag, not master. PaperMod's master branch has already raised its
# minimum Hugo to 0.146, which is newer than what most distributions package —
# tracking it means the site can stop building because of a commit you never
# made. v8.0 builds on Hugo 0.112+ and was verified against 0.123.7.
PAPERMOD_TAG="v8.0"
if [[ -f site/themes/PaperMod/theme.toml ]]; then
  HAVE=$(git -C site/themes/PaperMod describe --tags --always 2>/dev/null || echo "?")
  skip "PaperMod already present at $HAVE"
elif [[ -e site/themes/PaperMod ]]; then
  warn "site/themes/PaperMod exists but is empty — run:
        git submodule update --init --recursive"
else
  run "git submodule add -q \
        https://github.com/adityatelange/hugo-PaperMod.git site/themes/PaperMod"
  run "git -C site/themes/PaperMod checkout -q $PAPERMOD_TAG"
  ok "PaperMod pinned at $PAPERMOD_TAG (the Pages workflow checks out submodules)"
  echo "        To move it later:  git -C site/themes/PaperMod fetch --tags &&"
  echo "                            git -C site/themes/PaperMod checkout vX.0"
fi

# ---------------------------------------------------------------- 5. python

step "5. Python package"
if python3 -c "import standarderror" 2>/dev/null; then
  skip "standarderror importable"
else
  run "python3 -m pip install -e '.[dev]'"
  ok "installed in editable mode"
fi

if [[ $CHECK_ONLY -eq 0 ]]; then
  if python3 -m pytest -q >/dev/null 2>&1; then ok "test suite passes"
  else warn "tests failed — run 'pytest' and look before publishing"; fi
fi

# ---------------------------------------------------------------- 6. commit

step "6. Local git state"
if [[ -d .git ]]; then
  ok "git repo initialised ($(git rev-list --count HEAD 2>/dev/null || echo 0) commits)"
else
  run "git init -q && git branch -M main"
  ok "git initialised on main"
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)
if [[ "$CURRENT_BRANCH" == "main" ]]; then
  ok "on main"
elif git remote get-url origin >/dev/null 2>&1; then
  # A remote already exists, so renaming could orphan the upstream branch.
  warn "on branch '$CURRENT_BRANCH' but the Pages workflow triggers on main.
        With a remote already set, rename deliberately:
          git branch -M main && git push -u origin main"
else
  # No remote yet: renaming is free and avoids a silent no-deploy later.
  run "git branch -M main"
  ok "renamed '$CURRENT_BRANCH' -> main (the Pages workflow triggers on main)"
  CURRENT_BRANCH=main
fi

if [[ -n "$(git status --porcelain)" ]]; then
  run "git add -A && git commit -q -m 'Configure site for GitHub Pages'"
  ok "committed configuration changes"
else
  skip "working tree clean"
fi

# ---------------------------------------------------------------- 7. remote

step "7. GitHub remote"
if git remote get-url origin >/dev/null 2>&1; then
  skip "origin already set to $(git remote get-url origin)"
elif [[ $GH_READY -eq 1 ]]; then
  if gh repo view "$GH_USER/$REPO" >/dev/null 2>&1; then
    run "git remote add origin '$CODE_URL.git'"
    ok "repo already existed on GitHub; origin added"
  else
    run "gh repo create '$GH_USER/$REPO' --public --source=. --remote=origin \
          --description 'The Standard Error - writes a phenomenon down as numbers or equations, then looks for the mistake everybody makes in it'"
    ok "created public repo $GH_USER/$REPO"
  fi
else
  warn "no gh CLI: create the repo at https://github.com/new (public, no README),"
  echo "        then: git remote add origin $CODE_URL.git"
fi

if git remote get-url origin >/dev/null 2>&1; then
  run "git push -u origin main"
  ok "pushed main"
fi

# ---------------------------------------------------------------- 8. pages

step "8. GitHub Pages"
if [[ $GH_READY -eq 1 ]] && git remote get-url origin >/dev/null 2>&1; then
  if gh api "repos/$GH_USER/$REPO/pages" >/dev/null 2>&1; then
    skip "Pages already configured"
  else
    # build_type=workflow tells Pages to serve the artifact our Action uploads,
    # rather than building Jekyll from a branch.
    if run "gh api -X POST 'repos/$GH_USER/$REPO/pages' \
              -f 'build_type=workflow' >/dev/null 2>&1"; then
      ok "Pages enabled with GitHub Actions as the source"
    else
      warn "could not enable Pages via API. Do it by hand:"
      echo "        Settings -> Pages -> Source: GitHub Actions"
    fi
  fi
  run "gh workflow run pages.yml >/dev/null 2>&1 || true"
  ok "requested a site build (watch: gh run watch)"
else
  warn "enable Pages by hand: Settings -> Pages -> Source: GitHub Actions"
fi

# ---------------------------------------------------------------- done

step "Done"
cat <<EOF
  Site        $SITE_URL
  Repo        $CODE_URL
  Actions     $CODE_URL/actions

  Next:
    1. standarderror doctor                              # sanity check
    2. standarderror run exp001_chaos_horizon --publish   # writes a DRAFT
    3. make serve                                    # preview at :1313 (-D shows drafts)
    4. standarderror run exp001_chaos_horizon --publish --live
    5. git add -A && git commit -m 'Publish: how far ahead can you forecast chaos'
       git push
    6. once the site is live:
       standarderror run exp001_chaos_horizon --publish --live --medium
       then import build/medium/<slug>.md at https://medium.com/p/import

  The first Pages deploy takes a few minutes. Full walkthrough: docs/SETUP.md
EOF
