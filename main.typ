#import "/vendor/dual-typst/src/lib.typ": (
  tufte, sidenote, marginnote, sidecite,
  main-figure, margin-figure, full-width-figure,
  epigraph, new-thought, full-width, sans, diagram,
)

// SEO metadata for the HTML page (see the homepage repo's AGENTS.md). One
// description string, reused across meta/og/twitter and JSON-LD. The og:image
// is the homepage cover, referenced absolutely so it resolves off-site.
#let _is-html = sys.inputs.at("target", default: "pdf") == "html"
#let _seo-title = "Some (too-)simple tricks for running a generative classifier (on CPU)"
#let _seo-desc = "Three serving tricks (forced prefix, LM-head trimming, KV cache) speed up the Qwen3Guard generative safety classifier 8.5x on CPU, at identical fp32 verdicts."
#let _seo-url = "https://luxxxlucy.github.io/qwen3guard-test-doc/"
#let _seo-img = "https://luxxxlucy.github.io/projects/2026_qwen3guard_classifier/cover.png"
#let _seo-date = "2026-06-08"
#let _seo-jsonld = "{\"@context\":\"https://schema.org\",\"@type\":\"BlogPosting\",\"headline\":\"" + _seo-title + "\",\"description\":\"" + _seo-desc + "\",\"url\":\"" + _seo-url + "\",\"image\":\"" + _seo-img + "\",\"datePublished\":\"" + _seo-date + "\",\"dateModified\":\"" + _seo-date + "\",\"author\":{\"@type\":\"Person\",\"@id\":\"https://luxxxlucy.github.io/#jialin-lu\",\"name\":\"Jialin Lu\",\"url\":\"https://luxxxlucy.github.io/\"}}"
#let _seo-head() = {
  html.elem("meta", attrs: (("name"): "description", ("content"): _seo-desc))[]
  html.elem("meta", attrs: (("name"): "author", ("content"): "Jialin Lu"))[]
  html.elem("link", attrs: (("rel"): "canonical", ("href"): _seo-url))[]
  html.elem("meta", attrs: (("property"): "og:type", ("content"): "article"))[]
  html.elem("meta", attrs: (("property"): "og:title", ("content"): _seo-title))[]
  html.elem("meta", attrs: (("property"): "og:description", ("content"): _seo-desc))[]
  html.elem("meta", attrs: (("property"): "og:url", ("content"): _seo-url))[]
  html.elem("meta", attrs: (("property"): "og:image", ("content"): _seo-img))[]
  html.elem("meta", attrs: (("property"): "article:published_time", ("content"): _seo-date))[]
  html.elem("meta", attrs: (("property"): "article:author", ("content"): "https://luxxxlucy.github.io/"))[]
  html.elem("meta", attrs: (("name"): "twitter:card", ("content"): "summary_large_image"))[]
  html.elem("meta", attrs: (("name"): "twitter:creator", ("content"): "@JIALIN_LU_1996"))[]
  html.elem("meta", attrs: (("name"): "twitter:title", ("content"): _seo-title))[]
  html.elem("meta", attrs: (("name"): "twitter:description", ("content"): _seo-desc))[]
  html.elem("meta", attrs: (("name"): "twitter:image", ("content"): _seo-img))[]
  html.elem("script", attrs: (("type"): "application/ld+json"))[#[#set smartquote(enabled: false); #_seo-jsonld]]
}

#show: tufte.with(
  title: [Some (too-)simple tricks for running a generative classifier (on CPU)],
  author: "Jialin Lu",
  date: "2026-06-08",
  abstract: [
  ],
  toc: false,
  style: "envision",
  head-extra: if _is-html { _seo-head() } else { none },
)

#marginnote[Code: #link("https://github.com/LuxxxLucy/qwen3guard-test")[github.com/LuxxxLucy/qwen3guard-test]]

*TL;DR*:
This blog happens as one of my colleagues was trying to assess the qwen3guard model#marginnote[Qwen3Guard is an (autoregressive) language model that outputs text, which is finetuned on safety corpus and reframed as a classifier.]
for whether it (as an instance of generative classifier family) is of practical merits for scanning LLM prompts (in our usecase).
The results on in-house evals seem relatively okay-ish,
but then he complained that this is just too slow to be useful
in our usecase#marginnote[He adapted the original inference snippets in the README (see #link("https://github.com/QwenLM/Qwen3Guard/blob/main/README.md#L68-L102")[here]).
Also we do need to mention that we do not have GPUs in our usecase.
In my experience, less than 0.8~1B model should actually run faster
on CPU anyway so it should be fine.
(caveat: not always, but mostly).
].

So I then start digging into it and found that there are some really easy opportunities,
like if you read some basics you shall really not miss it.#marginnote[I only started reading LLM inference recently (like 2 months ago, when I realized that even in serving the same open source models, somehow some providers can make it much faster than others).]
But anyway I will lay down these really simple and apparent tricks here,
and the results are pretty good: we can get about 8.5x speedup on CPU (across different runtimes and similarly on GPU as well) with the same full precision fp32 model,
and quantization can further halve the latency.

= A classifier of text

The task we consider is simple,
we read a string of text and we want to label it as one out of the $K$ class#marginnote[here $K = 3$, being the 3 categories of `safe`, `unsafe`, or `controversial`.].

In an oversimplified textbook setting,
this classifier would be defined as a function $f$ that takes an input $x$ and turns it into a vector of (unnormalized) scores $f(x)$ with the dimension being $K$.
A softmax turns the scores into a proxy of probability for each category, and the largest one becomes the prediction:

$ p(y = k | x) = exp(f(x)_k) / (sum_(j=1)^K exp(f(x)_j)) $ <eq-disc>


The case of a generative classifier is different though.
As the name suggests, it uses a *generative* model of text.
It instead re-uses an autoregressive language model,
which takes an input string and extends it via next-token prediction so we have more text generated and concatenated in the end, like `Scan Result: Safe`.
Once the generation is done, we search the generated text,
if the word "safe" existed in the generated text, then we label it as "safe".
Of course, this means the model would have good prompts and finetuning so that this particular behavior is encouraged.

We need to note that several models use this recipe, such as
Llama Guard#sidecite(<llamaguard>), ShieldGemma#sidecite(<shieldgemma>), and Qwen3Guard#sidecite(<qwen3guard>).
I think
the main reason for this design is that we can assume that the base model is already pretrained on a vast amount of language and world knowledge,
so it has the capacity to be reframed as a good and robust (i.e. generalizable) classifier with just a little tuning.

One additional advantage, that makes it particularly interesting, is the user experience:
Now we can write human language for what is considered unsafe, a.k.a in-context learning, as part of the prompt.
This makes especially the org administrator happy as now finally in all these years they can write policies in an easy way,
and that this policy can also change on the fly without retraining the model, which is a huge plus.

= Analysis, breakdowns and tricks

So we have an autoregressive language model, reframed as a classifier,
now let us take a closer look at what is really computed.
#marginnote[
  This is using the default code from the hugging face readme, to call `generate()` with a prompt that includes the system prompt and the chat template, and then parse the generated text to get the verdict.
]

#block(
  width: 100%,
  fill: luma(248),
  stroke: 0.5pt + luma(218),
  radius: 3pt,
  inset: (x: 6pt, y: 2pt),
)[
  #set text(size: 6.5pt)
  ```
  input
   → render chat template
   → prefill              1 forward pass over the whole prompt
   → decode × 9 steps     1 forward pass per token: "Safety: unsafe\nCategories: …"
   → regex-parse the text
   → verdict (+ categories)
  ```
]

If we look closely enough we will understand that something is off, there are redundancies.

== L1: forced prefix

Our first instinct is that the
model writes `Safety: ` before the verdict anyway, so generating it is wasted work.
In fact, this can be seen as an extreme version of constrained decoding, we already know (and actually finetuned the model to do) this.
This makes the additional decoding work for generating `Safety: ` really unnecessary.

We can fix it simply by enforcing this part of the text instead,
which we call prefix enforcing.
We can simply treat it as if it is part of the input: append `Safety: ` to the prompt, run one forward pass, and read the next-token distribution at the last position.
The entirety of the decoding step is removed and now it becomes part of the prefill.

If we only want the verdict we can stop here; the `Categories:` line never needs to be decoded.
We know in real traffic, the benign samples must outnumber the malicious samples by several magnitudes, so we can simply just check for `Safety: unsafe` and make the `Categories:` part conditional computed.
In this way,
even more of the decode loop is eliminated: we need now only one forward pass instead of about ten.#sidenote[ShieldGemma's card publishes the identical recipe; Llama Guard's is the first-token-logit variant. The model would write `Safety: unsafe`, then a `Categories:` line, about nine tokens, so `generate()` runs about ten forward passes.]

#figure(
  image("/figures/src/fig_timeline.svg", width: 100%),
  caption: [
    `generate()` runs a prefill pass and then about nine decode steps, one forward pass each; the forced-prefix path runs one prefill and reads the verdict, since the label is fixed by the end of the first step.
  ],
)


== L2: LM-head trimming

Here we refer to `lm_head` as the final feedforward layer that projects the final embedding into the token logit space. If we have a text of length $N+1$, then this means we need to project every one of the $N$ positions onto the full 150,000-token vocabulary, but really, this is not needed.

I am actually surprised to find that this is the default behavior, but then I understand that PyTorch is ultimately a framework for training and this is actually needed and makes sense.
But in inference this is not really needed in two perspectives:
1. first, we only care about the last position, so the projections at the other positions are wasted work. Only $1$ of the $N$ is needed#sidenote[In PyTorch this is `logits_to_keep=1`; in ONNX we need to prune the computation graph with a slice node; llama.cpp and some other runtimes already return only the last position.].
2. Even for that one last position, we are actually only interested in the three label tokens, so projecting onto the whole vocabulary is also wasted work. Only $3$ of the $150,000$ are needed.

I mean this seems really obvious and really small, but since it is a 0.6B model, relatively these combined could not and should not be ignored.

#figure(
  image("/figures/src/fig_cast.svg", width: 100%),
  caption: [
    At the last position the `lm_head` gives a distribution over the whole vocabulary logit, and we only care about the three of them and renormalize to get $P(y | x)$.
  ],
)

This means a much smaller multiplication and an updated version of
@eq-disc, the classification head really shall be three rows of the `lm_head` all along if we only care about the classification.

#figure(
  image("/figures/output/fig_l2.svg", width: 100%),
  caption: [
    The projections at every position other than the last one are skipped.
  ],
)

== L3: KV cache

Now this is the usual game, we can cache the KVs for much of the system prompt.
This should need no explanation. It does not even need one more data copy as just make a persistent shared copy in memory would work. #sidenote[The real layout is a system-prompt head, then the user text, then a system-prompt tail and the forced `Safety: `. Only the head is a constant prefix, so only it is cacheable; the tail sits after the variable user text and will be recomputed. The diagrams is a over-simplification.]

#figure(
  image("/figures/output/fig_l3.svg", width: 100%),
  caption: [
    The optimized read with L3: the fixed prefix is cached, so the transformer runs over the suffix only.
    L1 and L2 still hold, one forward pass and a single last-position `lm_head`.
  ],
)

= Results

First, correctness: the optimization should not introduce any errors, and it should be: on every sample the optimized path returns the same verdict, this is actually exact.

The tricks change how much computation is run.
That means they hold on any backend and in any precision,
and the savings are largest where each forward pass is expensive, which is often determined by the memory bandwidth for moving things in and out between the CPU cache and memory.
So as long as we are using the same machine, different runtime backends should have similar speedup (the overhead of each runtime assumed to be similar as some fixed ratio plus constant).

Here we test and present the results with Qwen3Guard-Gen-0.6B, batch one inference, on a 16 cores Kunpeng 920 server CPU#marginnote[The work per call is small, so going past 12 cores gives diminishing returns, but I just settled with 16 cores as it seems a good and reasonable number.].
The input is a few hundred tokens, and each number is the median of 100 timed calls after basic warmups.

#figure(
  image("/figures/src/fig_ladder.svg", width: 100%),
  caption: [
    The ladder on one backend (PyTorch fp32).
    The decode loop, removed by L1, is most of the cost; L2 and L3 take the rest.
  ],
)

Similar speedup can be seen across three different backends: PyTorch, ONNX#sidecite(<onnxruntime>), and llama.cpp#sidecite(<llamacpp>).

#figure(
  table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    table.header([backend], [L0 baseline], [+L1], [+L2], [+L3]),
    [PyTorch fp32],  [2148], [688], [555], [408],
    [ONNX fp32],     [1671], [598], [485], [253],
    [llama.cpp q8_0], [643], [261], [#sym.dash.en], [129],
  ),
  caption: [
    Per-call latency, p50 ms, walking the ladder on three backends.
    L0 is the model-card `generate()` path, the same code the Qwen3Guard-Gen card shows.
    llama.cpp returns only the last position by default, so L2 is already in its baseline; that row is also 8-bit quantized, which is why it starts lower.
  ],
) <tab-ladder>

On every backend trick 1 (L1) removes most of the time, then L2 and L3 take more off:
the three tricks bring PyTorch from 2148 to 408 ms, about five times faster.
If we keep the tricks and switch to a faster fp32 runtime,
ONNX reaches 253 ms, #strong[8.5 times faster] than the default setting PyTorch reference.

Quantization is a separate dimension orthogonal to the three tricks, and it is the obvious next thing to try.
Storing the weights in 8 bits instead of 32 shrinks the model, on top of the tricks already in place.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    table.header([config (with the tricks)], [p50 ms], [vs. reference]),
    [ONNX fp32 (no quant)], [253], [8.5#sym.times],
    [ONNX int8],            [164], [13#sym.times],
    [llama.cpp q8_0],       [129], [17#sym.times],
  ),
  caption: [
    Best fp32 path against two 8-bit paths, all with the tricks applied, p50 ms.
  ],
) <tab-quant>

The full results across every backend we measured are listed below.
Here we test with two system prompt templates:
original (the official 296-token system prompt from Qwen3Guard) and test-200 (a compressed and simplified policy prompt, about 130 tokens).

#full-width-figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, right),
    table.header([backend], [variant], [original (p50 / p99)], [test-200 (p50 / p99)]),
    [pytorch fp32], [L0], [2148.1 / 2830.1], [875.8 / 1230.0],
    [], [+L1 forced prefix], [687.6 / 790.8], [423.7 / 433.0],
    [], [+L2 LM-head trimming], [554.9 / 711.0], [352.8 / 360.1],
    [], [+L3 KV cache], [407.5 / 428.9], [310.6 / 330.1],
    [onnx fp32], [L0], [1670.6 / 1709.6], [1277.5 / 1308.7],
    [], [+L1 forced prefix], [598.4 / 620.2], [315.0 / 327.9],
    [], [+L2 LM-head trimming], [485.1 / 502.8], [239.5 / 254.8],
    [], [+L3 KV cache], [253.2 / 265.5], [147.6 / 160.7],
    [onnx int8], [L0], [2136.3 / 2155.9], [1842.4 / 1865.4],
    [], [+L1 forced prefix], [382.0 / 389.2], [209.1 / 215.8],
    [], [+L2 LM-head trimming], [280.1 / 286.7], [155.6 / 161.3],
    [], [+L3 KV cache], [163.7 / 167.9], [113.9 / 118.5],
    [llamacpp f32 (L2 baked)], [L0], [1589.7 / 1625.2], [1239.7 / 1282.7],
    [], [+L1 forced prefix], [719.6 / 750.6], [426.7 / 467.5],
    [], [+L3 KV cache], [434.0 / 458.9], [317.1 / 394.1],
    [llamacpp f32 +kernel-opt (L2 baked)], [L0], [1278.0 / 1292.5], [966.6 / 974.1],
    [], [+L1 forced prefix], [511.9 / 523.5], [237.5 / 268.3],
    [], [+L3 KV cache], [242.2 / 249.0], [147.8 / 156.7],
    [llamacpp f16 (L2 baked)], [L0], [1496.6 / 1527.9], [1156.0 / 1189.0],
    [], [+L1 forced prefix], [928.4 / 960.5], [619.9 / 652.4],
    [], [+L3 KV cache], [653.4 / 691.0], [542.6 / 571.8],
    [llamacpp q8_0 (L2 baked)], [L0], [643.1 / 650.9], [437.4 / 445.2],
    [], [+L1 forced prefix], [261.2 / 273.5], [111.5 / 115.5],
    [], [+L3 KV cache], [128.7 / 133.1], [69.8 / 74.1],
    [rust-candle fp32], [L0], [6149.1 / 6227.9], [5205.9 / 5252.7],
    [], [+L1 forced prefix], [1270.6 / 1346.8], [536.3 / 550.3],
    [], [+L3 KV cache], [726.5 / 769.8], [374.1 / 388.4],
    [ctranslate2 fp32 (L2 baked)], [L0], [#sym.dash.en], [#sym.dash.en],
    [], [+L1 forced prefix], [1718.3 / 1780.0], [973.2 / 991.1],
    [mnn-llm fp16 (L2 baked)], [L0], [1336.8 / 1431.6], [1037.8 / 1127.1],
    [], [+L1 forced prefix], [571.1 / 586.7], [287.9 / 301.4],
  ),
  caption: [
    The full result on Kunpeng 920 aarch64, p50 / p99 ms, batch one, 16 threads.
    Rows are cumulative within a backend; "(L2 baked)" means the backend L0 already returns only the last position, so it has no separate +L2 row.
  ],
) <tab-full>

= Wrap up

Here we listed three tricks for optimizing Qwen3Guard, a generative classifier,
that are quite simple and apparent once you look at it.
The three tricks (forced prefix, LM-head trimming, KV cache) take a two-second CPU call down to about 250 ms,
that is
8.5 times faster without quantization;
quantization would further halves it.

In fact, these tricks are so apparent that if we look at vLLM, perhaps all these tricks are already implemented anyway#sidenote[And indeed, vLLM implemented them].

#bibliography("refs.bib", style: "chicago-author-date")

= Appendix: GPU results

Additional results on GPU are presented here.
Our main hardware is an RTX 3090.

On an RTX 3090, Qwen3Guard-Gen-0.6B with L1 runs about 29 ms p50 against 237 ms for the default setting baseline at a comparable input length.
L2 and L3 do not help on the GPU,
where the vocabulary projection and prompt re-reading are cheap next to the per-call overhead.
On CPU the opposite holds, which is why the full ladder matters there.

The stage-by-stage cost at a representative input, about 369 tokens, on the 3090 makes this plain:
prefill is about 21 ms, then nine decode steps at roughly 17 ms each.
That decode loop is most of the default path's 237 ms, and the forced-prefix path drops it entirely,
leaving prefill and a sub-millisecond read, about 29 ms.

#figure(
  image("/figures/src/fig1_latency_0.6b.png", width: 100%),
  caption: [
    Qwen3Guard-Gen-0.6B on an RTX 3090: the model-card path against the forced-prefix path (L1), p99 latency across input lengths.
    The forced-prefix path stays under a 200 ms budget out to about 2048 input tokens.
  ],
)

The same construction works on the larger sizes:
with the forced-prefix path, 0.6B stays under a 200 ms p99 budget up to about 2048 input tokens, 4B up to about 256, and 8B up to 128.

#figure(
  image("/figures/src/fig2_sizes_optimized.png", width: 100%),
  caption: [
    The forced-prefix path across the three Qwen3Guard-Gen sizes (0.6B / 4B / 8B) on an RTX 3090.
    All three get the same trick; the 0.6B clears a 200 ms budget at the longest inputs, the larger two hit it sooner.
  ],
)
