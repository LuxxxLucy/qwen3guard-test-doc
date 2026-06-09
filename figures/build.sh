#!/usr/bin/env bash
# Render the cetz stage-ladder figures to standalone SVGs in output/.
# The doc includes these via image("/figures/output/fig_lN.svg").
set -euo pipefail
cd "$(dirname "$0")"

FONTS=../vendor/dual-typst/assets/fonts
mkdir -p output

for f in l2 l3; do
    typst compile --root . --font-path "$FONTS" \
        --input fig="$f" render.typ "output/fig_$f.svg"
done
echo "Built figures: $(pwd)/output"
