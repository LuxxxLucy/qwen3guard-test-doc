// Harness: render one cetz figure to a tightly-cropped page.
// Pick the figure with `--input fig=l0` (l0 / l1 / l2 / l3).
#import "diagrams.typ" as d

#set page(width: auto, height: auto, margin: 4pt)
#set text(font: "Roboto", size: 9pt)

#let which = sys.inputs.at("fig", default: "l2")
#if which == "l2" [#d.fig-l2] else if which == "l3" [#d.fig-l3]
