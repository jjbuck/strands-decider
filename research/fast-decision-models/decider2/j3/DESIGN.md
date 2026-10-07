# J3 design: a depth-split decision transformer

Labels: [M] measured, [A] arithmetic from measured anchors (`code/costmodel.py`, `costmodel.json`), [S] speculation.

## 1. What a decision model does not need

A decoder spends the same 24 layers × 2048 width on every token, because in generation every position may become the
one that emits the next token, and every position's K/V must be valid at every layer for the tokens after it. A decision
model reads out only K+1 rows per question: the option-end rows and `<answer>`. Four things follow.

1. **Per-position outputs exist only for question rows.** A state row's layer-l output is consumed by nothing except
   later layers' attention and GDN recurrences, which are read by question rows. There is no LM head or next-token loss
   that needs every row to be deep.
2. **The state is question-independent in hobson** (state-first, causal). So the deep processing of each state row is
   generic context-building, paid at 24 layers whether or not any question needs it.
3. **There is no decode, so no KV cache across steps.** The state's representation needs to exist once per request, at
   whatever depth the readers want.
4. **Questions are few, short and fixed per deployment.** They are where the decision is computed. The routed-depth study
   found that "the question rows dominate the late layers" [M, section 7].

Cost is therefore misallocated. At T = 1000 a single-question request is about 89% state rows, and every state row
costs 2.745 GFLOP [A].

## 2. Evidence that deep state processing is mostly redundant

`~/decider2/tokens/stale_sweep.jsonl` holds an earlier, unreported, training-free test on 35 long JevBench items
(mean 1,320 state tokens). In it, state rows stop after layer k. Later full-attention layers read K/V projected from the
stale residual h_k, and GDN layers see only the question rows. Agreement with hobson [M, recomputed here]:

| state rows stop after layer | 3 | 7 | 11 | 15 | 22 |
|---|---|---|---|---|---|
| argmax agreement | .514 | .543 | .943 | .971 | 1.000 |
| total variation | .277 | .240 | .058 | .006 | .001 |

Without any training, hobson's decisions barely need its state rows past layer 11. This is the opening. The bold
question is how far down the split can go once the model is trained for it.

## 3. Candidates

### C1 (built): DT, the depth-split decision transformer
- **Shallow stack (layers 0..L_s-1).** hobson's own layers run on state rows plus question rows, state-first, at full
  width. Every token is kept.
- **Memory.** M = the layer-L_s residual of every state row: T × 2048, nothing pooled or selected.
- **Deep stack (layers L_s..23).** These run on question rows only. Each deep layer reads the memory through its own
  pretrained projections applied to `norm_j(M)`, the SwiftKV/YOCO pattern, plus a small trained memory adapter
  (LoRA on the memory K/V projections).
  - **Bridge A:** deep attention layers read memory K/V at state positions (RoPE kept). Deep GDN layers recur over
    question rows only.
  - **Bridge G:** deep GDN layers also scan the memory with their own k, v, β and g, and the question rows continue
    from that final GDN state.
- **Head.** hobson's pointer head on the deep option-end and answer rows.
- **Training.** hobson is the in-process teacher, using the F7 recipe plus counterfactual augmentation, plus dense
  distillation of the deep question rows (the term H7 found necessary).
- **Elastic split.** L_s is sampled from {4, 8, 12} per example, so one run measures the whole frontier.

**FLOPs per state token [A].** Every question row still costs 2.745 GFLOP.

| | hobson | DT-A4 | DT-A8 | DT-A12 | DT-G4 | DT-G8 | DT-G12 |
|---|---|---|---|---|---|---|---|
| GFLOP | 2.745 | 0.500 | 0.949 | 1.398 | 0.753 | 1.152 | 1.550 |
| share of hobson | 1 | .18 | .35 | .51 | .27 | .42 | .56 |

**Latency, 1 question of 125 tokens [A, ms].** The model reproduces hobson's measured fused-runtime anchors within
about 10% (7.1 / 14.3 / 27.8 / 53.6 / 218.7 modelled, against 9.4 / 15.7 / 28.3 / 52.7 / 200.5 measured). Ratios are
more reliable than absolute values.

| card | model | 64 | 256 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|
| A10G | hobson | 10.8 | 20.9 | 28.5 | 60.2 | 225.8 |
| A10G | DT-A4 | 8.1 | 10.0 | 11.3 | 17.1 | 47.2 |
| A10G | DT-A8 | 8.7 | 12.1 | 14.8 | 25.8 | 82.9 |
| A10G | DT-A12 | 9.2 | 14.3 | 18.2 | 34.4 | 118.7 |
| 3090 (fp16 accumulation) | hobson | 5.7 | 10.9 | 14.8 | 31.3 | 118.0 |
| 3090 | DT-A4 | 5.0 | 5.9 | 6.6 | 9.6 | 25.4 |
| 3090 | DT-A8 | 5.1 | 6.9 | 8.3 | 14.0 | 43.9 |

With 4 questions (500 question rows) at T = 1000 on the A10G: hobson 80.3, DT-A8 45.9 and DT-A4 37.3 ms [A]. The deep
question rows then dominate, which is what C2 addresses. The 4090 and 5090 columns are in `costmodel.json`.

**Why it could beat a decoder at equal capability.** The deep stack keeps all 2B parameters and all 24 layers for the
rows that compute the decision, and every state token remains readable at full width by full softmax attention in every
deep attention layer. Only the depth of state rows shrinks, and section 2 measures that depth as largely redundant
before any training. Training is also cheap: student cost per example is L_s/24 of the state plus the question rows.
That buys more data, which section 7 names as the lever.

**Why it is not a failed idea from sections 5 and 7.**
- **Selection, routing, eviction and training-free selection.** Those removed state tokens from late layers. DT removes
  no token. The late layers see every state row through attention.
- **Funnel pooling, structure-native pooled leaves, dual encoders.** Those compressed the state into fewer or narrower
  vectors, and exact values died there. DT's interface is T × 2048 at layer L_s. Shallow representations are the
  most lexical, so digits and IDs are intact.
- **Small jointly-trained reader, and a reader supplying per-layer GDN state to a frozen 2B.** The reader there was a
  separate narrow network, and the frozen reasoner needed GDN states to about 1%. DT's reader is hobson's own first L_s
  layers at full width. The deep stack is trained, and bridge A does not try to reproduce deep GDN states at all.
- **Question-first conditioning, and per-question compiled readers.** DT stays state-first, which is hobson's own
  conditioning.
- **Smaller or shallower models.** DT has the same parameters, and question rows get 24 layers. hob12 (state and
  question both at 12 layers; REAL-label .735) is the matched shallow control that DT-12 must beat.
- **Section 5 (4-bit) does not apply.** DT is orthogonal to precision and composes with W8A8 [S].

**Risks.**
- **Contextualization.** State rows lose their deep contextualization, for example which record an amount belongs to.
  The deep rows must recover it through attention, and this is exactly what CF-probe with distractors tests.
- **Untrained gap.** The training-free gap at L_s ≤ 8 is large (.54).

### C2: slot-deep DT with a compiled schema (designed, not built)
- **Layout.** Question text joins the shallow stack, state-first, so it sees the state. Only option-end and `<answer>`
  rows (K+1 per question) enter the deep stack, reading memory = [state ⊕ question] at L_s.
- **Cost.** The marginal cost of a question becomes (q × L_s + (K+1) × 24) rows against q × 24. With 15 banking
  questions at T = 1000 [A]: about 0.2x hobson's packed time.
- **Why it is not H7.** H7's slot sets read a question bundle that never saw the state, and reached agree_sd .69. Here
  the question text is state-aware in the shallow stack.
- **Risk.** The 23-way procedure questions need deep question-text processing (H7's misses).

### C3: an exact value channel (designed, not built)
- **Mechanism.** A deterministic parser marks amounts, dates and IDs. Each value span carries typed features (sign,
  log-magnitude Fourier features, days-since-epoch, a 64-bit ID fingerprint as ±1). The features are added to the memory
  rows, and as an additive attention-logit bias (exact ID match, ordered comparison) in deep attention layers.
- **Why it is not the structure-native reader.** No token is pooled. It is an additive side channel on the full memory.
- **Status.** Not built, because the CF augmentation already lifts CF-probe far above hobson (H7: .977). C3 matters for
  out-of-template comparisons [S].

### C4: counterfactual-contrastive calibration (partly folded into training)
- **Energy view.** The pointer head is already an energy E(state, question, option).
- **Pair loss.** Training on counterfactual pairs (one edit flips the label) with paired CE pushes E's sensitivity onto
  the edited detail, which is hobson's measured weakness (27% pair accuracy, 95% right direction).
- **Calibration.** It is refit afterwards with hobson's per-kind temperatures.

### C6 (built second): DT-set, option-order invariance by construction (added at the engineer's request)
- **The problem in hobson.** The option list is one causal sequence: "1. a", "2. b" and so on. Option k sees options
  before it, and carries its own number. hobson's decision is unchanged on only 0.884 of questions when the options are
  reversed (H4).
- **Each question becomes a tree.** The stem (header, instructions, `<options>`) is one branch off the state. Each
  option, unnumbered, is its own branch off the stem, and every option branch starts at the same position. The
  `<answer>` tail is a branch off the stem positioned after the longest option.
- **GDN layers.** Options and tail continue the stem's final state, so no option sees another.
- **Attention layers.** The tail attends to all of its options as a set, at equal positions. That is the only
  cross-option comparison, and it is permutation-invariant.
- **Scoring.** The pointer head scores each option's own last row against the tail's query, an energy E(state,
  question, option) per option, then normalizes with a softmax.
- **Result of the construction.** Permuting the options permutes identical, independent branches, so the label
  distribution is invariant up to floating-point summation order. This is verified at max |Δp| .0031 under rotation,
  untrained [M].
- **Cost.** DT-set keeps DT's split (shallow state, deep question rows) at the same row count. The tree adds one varlen
  GDN call per GDN layer.
- **Why it is not a failed idea.** No state is cut, and the question still sees the state at every layer. Unlike H7's
  slot sets, the question text is processed per request, state-first, in hobson's conditioning.
- **Untrained.** It agrees with hobson on 27 of 30 questions at L_s = 24 and 23 of 30 at L_s = 8 [M].
- **Metric.** For hobson, DT and DT-set: decisions unchanged, and mean |Δp| over labels, under up to 3 distinct
  non-identity rotations (noul and choice), plus rubric reversal for score questions. ECE and Brier are reported with
  every suite.

### C5: a static per-deployment graph (systems; in the DT runtime)
- **Static shapes.** The deep stack is a fixed graph over the question rows of a fixed schema. Only the shallow stack
  depends on T.
- **Batching.** All deep layers' memory projections are concatenated into one GEMM over M.
- **Launch cost.** One CUDA graph per exact T removes launch overhead, as in d1/H4.

## 3b. Training-free frontier [M]
Setup: hobson weights and head. The subset is REAL-agree 120 items, LONG 40, JB-hard 130, CF 120 pairs and CF-probe 120
pairs (`tf/`, `tf_scores.json`).

| L_s / bridge | REAL agree_sd | LONG agree_sd | CF fgh | CF-probe fgh | REAL-label (hobson .811) | JB-hard (hobson .523) |
|---|---|---|---|---|---|---|
| 24 (= hobson in this runtime) | .988 | 1.000 | 1.000 | .973 | .822 | .531 |
| 16A (state FLOPs .67x) | 1.000 | 1.000 | 1.000 | .973 | .811 | .546 |
| 12G | 1.000 | 1.000 | 1.000 | .946 | .822 | .538 |
| 12A (.51x) | .952 | .963 | 1.000 | .865 | .789 | .477 |
| 8G | .880 | .852 | .686 | .568 | .778 | .446 |
| 8A (.35x) | .783 | .741 | .743 | .541 | .744 | .462 |
| 4A / 4G | .23 / .33 | .22 / .30 | 0 / .11 | 0 / .03 | .68 / .67 | .485 |

## 4. Decisive experiment (C1)
1. **Exactness check.** L_s = 24 must equal hobson.
2. **Training-free frontier.** Measure L_s ∈ {4, 8, 12, 16} × bridge {A, G} on REAL-agree and CF subsets, to choose
   the bridge.
3. **Elastic training.** One elastic run over L_s ∈ {4, 8, 12} for about 2.5 GPU-hours on the A10G.
4. **Evaluation.** Every evalkit suite at each L_s, for absolute accuracy against the bar: no significant drop on JB-all
   or JB-hard, REAL-label ≥ .78, CF and CF-probe pair accuracy ≥ .268 / .328, with agree_sd reported.
5. **Latency.** A fused DT runtime (TTL kernels): shallow pass, memory K/V GEMM, deep pass over question rows. hobson
   is measured in the same harness at 64–4000 tokens with 1 and 4 questions.

**Verdicts.**
- **Falsified:** DT fails the bar at every L_s with which it is faster than hobson by at least 1.5x.
- **Transformative:** DT passes at L_s ≤ 8, giving at least 2.5x at T ≥ 1000 with no loss of capability.
