// Figure set for the three-tricks section, drawn in cetz.
// Two model-pipeline snapshots:
//   fig-l2  the optimized read (forced prefix, one pass, LM-head trimming)
//   fig-l3  the optimized read with the fixed prefix's KV reused
// Rendered to figures/output/*.svg by build.sh and included via image().

#import "@preview/cetz:0.4.2"

#let _teal  = rgb("#2a7f7f")
#let _tealL = rgb("#9fcccc")
#let _grey  = rgb("#cccccc")
#let _greyT = rgb("#e6e6e6")
#let _greyD = rgb("#888888")
#let _dim   = rgb("#e2e2e2")
#let _dimT  = rgb("#b0b0b0")

#let dr = cetz.draw
#let _sd  = (paint: _greyD, thickness: 0.6pt)
#let _sdim = (paint: _dimT, thickness: 0.5pt, dash: "dashed")

#let cbox(cx, cy, body, w: 0.9, h: 0.5, fill: _grey, stroke: _sd, tcol: black, sz: 7pt) = {
  dr.rect((cx - w/2, cy - h/2), (cx + w/2, cy + h/2), fill: fill, stroke: stroke)
  dr.content((cx, cy), text(size: sz, fill: tcol)[#body])
}

#let lab(cx, cy, body, sz: 7.5pt, tcol: black, anchor: "center") = {
  dr.content((cx, cy), text(size: sz, fill: tcol)[#body], anchor: anchor)
}

#let arr(a, b, stroke: _sd) = dr.line(a, b, mark: (end: ">", fill: _greyD, scale: 0.4), stroke: stroke)

// input token centres, shared across snapshots
#let _sys  = (0.6, 1.55, 2.5, 3.45, 4.4)
#let _user = (5.5, 6.45, 7.4)
#let _forced = 8.7
#let _cx = 4.65   // pipeline centre line

#let pipeline(stage) = cetz.canvas(length: 1cm, {
  let original = stage == "l0"
  let forced   = not original
  let cached   = stage == "l3"
  let all = _sys + _user + (if forced { (_forced,) } else { () })
  let last = all.last()

  // ---- top caption ----
  if original { lab(0, 9.0, [model card `generate()`], tcol: _greyD, anchor: "west") }
  if stage == "l2" { lab(0, 9.0, [*the optimized read* — #text(fill: _teal)[L1 forced prefix + L2 LM-head trimming]], anchor: "west") }
  if stage == "l3" { lab(0, 9.0, [*the optimized read* + #text(fill: _teal)[L3 KV cache]], anchor: "west") }

  // ---- input token row ----
  let sys-fill = if cached { _dim } else { _grey }
  let sys-stk  = if cached { _sdim } else { _sd }
  for x in _sys { cbox(x, 8.1, [], fill: sys-fill, stroke: sys-stk) }
  for x in _user { cbox(x, 8.1, [], fill: _greyT) }
  if forced { cbox(_forced, 8.1, [Safety:], w: 1.3, fill: _teal, stroke: (paint: _teal, thickness: 0.6pt), tcol: white) }

  dr.line((_sys.first() - 0.45, 7.72), (_sys.last() + 0.45, 7.72), stroke: sys-stk)
  lab(2.5, 7.48, if cached [system prompt #text(fill: _teal)[(cached, KV reused)]] else [system prompt (fixed)], tcol: if cached { _dimT } else { black })
  dr.line((_user.first() - 0.45, 7.72), (_user.last() + 0.45, 7.72), stroke: _sd)
  lab(6.45, 7.48, [user text])
  if forced { lab(_forced, 7.48, [forced prefix], tcol: _teal) }

  // ---- transformer ----
  arr((_cx, 7.3), (_cx, 7.05))
  dr.rect((0.0, 6.55), (9.4, 7.05), fill: _grey, stroke: _sd)
  lab(_cx, 6.8, if cached [transformer layers #text(fill: _teal)[(suffix only)]] else [transformer layers])

  // ---- hidden states ----
  arr((_cx, 6.55), (_cx, 6.3))
  lab(-0.15, 6.1, [hidden states], anchor: "east", tcol: _greyD)
  for x in all {
    let dim = forced and x != last
    let f = if x == last and forced { _teal } else if (cached and _sys.contains(x)) or dim { _dim } else { _grey }
    dr.circle((x, 6.1), radius: 0.12, fill: f, stroke: if f == _dim { (paint: _dimT, thickness: 0.5pt) } else { _sd })
  }

  // ---- lm_head: one per position (original) or last position only (optimized) ----
  if original {
    for x in all {
      arr((x, 5.98), (x, 5.72))
      cbox(x, 5.45, [`lm_head`], w: 0.85, h: 0.5, fill: _tealL, stroke: (paint: _teal, thickness: 0.6pt), sz: 5pt)
    }
    lab(9.6, 5.45, [one `lm_head` #linebreak() per position], anchor: "west", tcol: _greyD)
  } else {
    // every non-last lm_head is skipped (greyed); only the last runs
    for x in all.slice(0, all.len() - 1) {
      cbox(x, 5.45, [], w: 0.85, h: 0.5, fill: none, stroke: _sdim)
    }
    lab(_sys.at(2), 5.45, [skipped], tcol: _dimT, sz: 6.5pt)
    arr((last, 5.98), (last, 5.72), stroke: (paint: _teal, thickness: 0.8pt))
    cbox(last, 5.45, [`lm_head`], w: 0.85, h: 0.5, fill: _tealL, stroke: (paint: _teal, thickness: 0.6pt), sz: 5pt)
    lab(last + 0.6, 5.45, [last position #linebreak() only], anchor: "west", tcol: _teal)
  }

  // ---- output ----
  if original {
    // the decode loop: the model writes a sequence of tokens, one per pass
    arr((_cx, 5.2), (3.6, 4.9))
    cbox(1.2, 4.55, [Safety:], w: 1.3)
    cbox(2.6, 4.55, [unsafe], w: 1.1, fill: _teal, stroke: (paint: _teal, thickness: 0.6pt), tcol: white)
    cbox(3.55, 4.55, sym.arrow.l.hook, w: 0.5)
    cbox(5.0, 4.55, [Categories:], w: 1.8)
    lab(6.1, 4.55, [#text(fill: _greyD)[...]], anchor: "west")
    lab(2.6, 4.98, [verdict], tcol: _teal, sz: 7pt)
    dr.line((0.55, 4.18), (6.0, 4.18), stroke: _sd)
    lab(3.1, 3.92, [decode loop — about nine forward passes, one token each], tcol: _greyD)
  } else {
    arr((last, 5.2), (last, 4.92), stroke: (paint: _teal, thickness: 0.8pt))
    cbox(last, 4.6, [unsafe], w: 1.3, fill: _teal, stroke: (paint: _teal, thickness: 0.6pt), tcol: white)
    lab(last + 0.75, 4.6, [verdict], tcol: _teal, sz: 7pt, anchor: "west")
    lab(_cx, 4.0, [one forward pass, read at the last position], tcol: _greyD)
  }
})

#let fig-l2 = pipeline("l2")
#let fig-l3 = pipeline("l3")
