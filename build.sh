#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

FONTS=vendor/dual-typst/assets/fonts

# Render the cetz stage-ladder figures to figures/output/*.svg first.
figures/build.sh

# PDF
typst compile --root . --font-path "$FONTS" \
    main.typ main.pdf &

# HTML, emitted as index.html so GitHub Pages serves it at the repo root.
typst compile --root . --font-path "$FONTS" \
    --features html --input target=html \
    main.typ index.html &

wait
echo "Built: $(pwd)/main.pdf $(pwd)/index.html"
