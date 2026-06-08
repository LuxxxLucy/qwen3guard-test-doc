# Generative classifiers are just language models

Source for the blog post. Built with [dual-typst](https://github.com/) (vendored under `vendor/`), one Typst source compiling to both PDF and HTML.

## Build

```sh
./build.sh
```

Produces `main.pdf` and `index.html`. The HTML is self-contained (fonts via the vendored Roboto family, images embedded as data URIs), so `index.html` can be served as-is.

## Files

- `main.typ` — the post.
- `refs.bib` — references.
- `assets/images/` — figures (GPU latency sweeps).
- `vendor/dual-typst/` — vendored template engine (`src/` + Roboto fonts), so the repo builds standalone with no sibling checkout.

## Publish

This directory is its own git repo, published to GitHub Pages. Push to a Pages-enabled repo; `index.html` is served at the root. Then add a link entry on the `LuxxxLucy.github.io` index.
