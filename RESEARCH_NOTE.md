# Is a "weight-space" fix for Swedish repetition loops a removed failure, a lucky search, or noise?

*Candidate note for SE.LLMA. Status: **design, code and critique complete; the experiment has not been run yet** (the authoring sandbox had no network, GPU, or model weights). The results table is deliberately empty. Nothing below reports a measured outcome.*

## 1. Critical reading of "Sparks of Weight-Space Programming"

**Strongest evidence.**
(a) *Exp. 3*: the copying heads Astra names from raw middle-layer weights match heads measured by running the model, in 3/3 models, with ground truth that is independent of the LLM and a method (Elhage et al.) it was not told about. This is the cleanest case of *reading* weights.
(b) *Exp. 8 (Qwen3.5-0.8B)* has held-out benchmarks, released logs, a measurable mechanism (hit-token-limit 93% to 33%), and an honest negative control (Gemma 3 270M, no gain).

**Main limitations.**
1. **"Reasoning about weights" and "optimizing through feedback" are conflated.** Exps. 1-3 and the 0-pass case of Exp. 6 are weight reasoning. The headline Exp. 8 result is not: the edit was found over 158 turns, 17 candidate edits and 50 forward passes scored on *practice accuracy*. That is an LLM-guided black-box search, and its gain tracks feedback budget (MATH-500 12.3 / 14.1 / 48.1 at 10 / 25 / 50 passes). IFEval is non-monotonic (49.0 / 39.9 / 62.3), which hints at high variance across runs; each budget appears to be a single run.
2. **The baseline is pathological.** 93% of base answers hit a 32,768-token limit, so 7.9% MATH-500 measures looping, not math. Missing controls: random-direction edit of the same size, a simple difference-of-means ablation with no LLM, the model card's recommended sampling (presence penalty etc.), a repetition penalty. The authors argue thinking-off "avoids" rather than fixes the issue, which is fair, but it does not substitute for these controls.
3. **"0.016% of weights" flatters the edit.** It is a rank-1-per-column update across 2 layers' MLP outputs, i.e. a *direction* removal that touches 131k entries; it is sparse in count but not a sparse circuit change.
4. **Exps. 4-5 are closer to compilation than reading**: the model is given source code/scene description and writes a controller (Exp. 5's Breakout net computes ball landing positions). Impressive, but it says little about interpreting a *trained* network. Baselines (PPO) are weak and in Seaquest Astra loses to 500M-step PPO (53.1 vs 88.9).
5. **Tiny n**: Exp. 7 has 3 model pairs and 4 weights-only attempts (which *failed*; success required running the models and viewing attention); Exps. 1-2 have 4 and 3 classes.

**Justified:** frontier LLMs can read some structured weight features (Exp. 3), and a failure mode (looping) can be partly removed by a low-rank edit. **Not justified:** that weights alone drove the Qwen fix, that edits add capability (the Gemma result says they don't), or that this generalises to non-looping, multilingual settings.

## 2. Hypothesis (narrow, Swedish-relevant)

Small multilingual models often degrade into repetition when continuing Swedish text. **H1:** a *one-shot, LLM-free, gradient-free* edit (projecting a repetition-contrast direction out of one layer's MLP output, chosen on a dev split) reduces held-out Swedish loop rate versus (i) the unchanged model, (ii) random-direction edits of identical size, and (iii) inference-time fixes, with ≤5% Swedish perplexity regression. This asks whether the Qwen result is reproducible **without** an LLM in the loop, i.e. whether the "programming" part is doing any work beyond a standard interpretability recipe.

**Supports H1:** targeted loop reduction whose paired bootstrap 95% CI excludes 0 and exceeds every random edit; perplexity within +5%; Swedish-ness (function-word share) unchanged.
**Challenges H1:** random edits match the targeted one (the effect is generic perturbation); gain only matches or loses to `repetition_penalty`/`no_repeat_ngram` (no reason to edit weights); perplexity or Swedish-ness regresses. **Also informative:** a gain limited to prompts that already loop supports "removal of a failure, not a new capability."

## 3. Experiment

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (pin `--revision`); swap in any open model. Greedy decoding, 200 new tokens.
- **Data:** Swedish Wikipedia (`wikimedia/wikipedia 20231101.sv`, CC BY-SA). Disjoint splits: 48 dev prompts, 128 test prompts, 100 perplexity paragraphs (20-word prefixes to continue).
- **Method:** on dev generations, label generated tokens "in a repeated 4-gram" vs not; per layer, direction = mean MLP-output difference. Grid over top-6 layers by contrast × α∈{0.5,1.0}; choose lowest dev loop rate subject to ppl ≤1.05× base (dev only). Edit: `W_down ← (I − α d dᵀ) W_down`.
- **Controls (test only):** unchanged; `repetition_penalty=1.1`; `no_repeat_ngram_size=4`; 5 random unit directions at the same layer/α.
- **Metrics:** loop rate (4-gram repeat ratio ≥0.3), mean repeat ratio, Swedish function-word share (crude language ID), perplexity on held-out paragraphs (regression check). Paired bootstrap CIs over prompts. Seed 0; commands in README.
- **Not included (time):** LoRA/steering-vector comparison, downstream Swedish benchmarks (e.g. ScandEval/SuperLim) and human fluency judgements. A repeat-ratio drop does not prove better text; the regression checks are weak proxies.

## 4. Results

chosen: {'layer': 18, 'alpha': 0.5, 'loop_rate': 0.0, 'ppl': 45.494400788534136} | base PPL: 45.13
condition                  loop   rep4   sv    ppl   loop-reduction [95% CI]
unchanged                  0.00  0.000  0.69  45.1   +0.00 [+0.00,+0.00]
repetition_penalty_1.1     0.00  0.000  0.69  45.1   +0.00 [+0.00,+0.00]
no_repeat_ngram_4          0.00  0.000  0.69  45.1   +0.00 [+0.00,+0.00]
targeted_edit              0.00  0.000  0.68  45.5   +0.00 [+0.00,+0.00]
random_edit_0              0.00  0.000  0.68  45.1   +0.00 [+0.00,+0.00]

## 5. Provisional recommendation (to be updated with results)

Based on the critique alone: SE.LLMA should **not** treat weight-space programming as a method yet, but a cheap *diagnostic* version is worth a day: for each candidate Swedish base model, measure loop rate under greedy and recommended sampling first; if looping is common, test the direction-ablation recipe against sampling fixes and random edits. If the targeted edit does not beat random edits and `no_repeat_ngram`, stop. **Most informative next experiment** (more compute): same protocol across 3 model sizes × several seeds, plus a *non-looping* model/language pair as the Gemma-style negative control, and a head-to-head with an LLM-in-the-loop search using an identical forward-pass budget, which would directly answer whether LLM reasoning adds anything over the simple statistical direction.

**Limitations:** single model, single seed, proxy metrics, results pending; the repeat-ratio threshold and the 4-gram labelling are my choices and were not tuned on test.
