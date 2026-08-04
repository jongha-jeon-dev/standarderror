# Site

Hugo + PaperMod. One-time setup:

```bash
cd site
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

Then set `baseURL` in `hugo.toml` to your Pages URL and `SITE_BASE_URL` in `.env`
to the same value — `publish.medium_bundle()` builds absolute image URLs from it,
and Medium's importer needs them to resolve.

`quantpost run <exp> --publish` writes `content/posts/<slug>/index.md` with the
figures beside it. Preview with `hugo server -D`; push to deploy.

Publish here **first**, then crosspost. Medium's importer sets `rel="canonical"`
to the original URL, so your domain gets the search credit instead of Medium's.
