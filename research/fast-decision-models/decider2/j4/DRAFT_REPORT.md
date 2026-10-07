# J4: decision pretraining at equal size (box j4, A10G)

Labels: **[M]** measured, **[V]** verified from a fetched source, **[S]** speculation. Files: `~/decider2/j4/` (NOTES, code, preds, `results.json`, `seeds.json`, `curves.png`, `lat_j4.json`).

## 1. Hypothesis, from first principles
- **[V] ELECTRA (arXiv 2003.10555).**
  - ELECTRA-Small scores 79.9 GLUE, against BERT-Small at 75.1 (same FLOPs) and GPT at 78.8 (29x the FLOPs).
  - ELECTRA-400K matches RoBERTa-500K (89.0 against 88.9) at 1/4.5 the compute.
  - Its ablation, Table 5: ELECTRA 85.0, All-Tokens MLM 84.3, ELECTRA-15% 82.4, BERT 82.2. The gain is the loss on *every* token; being discriminative adds about 0.7 points.
- **[S] So replaced-token detection should not beat NTP**, which is already a loss on every token.
- **[M] What is sparse is the *decision* signal.**
  - hobson's fine-tune mix gives one label per 346 tokens.
  - The no-state baseline agrees with hobson on 0.68 of real questions, so most labels do not require reading the state.
- **Hypothesis.** Dense, self-supervised supervision at the pointer head ("decision pretraining", DP) buys detail reading per parameter.
- **Why it is not marginal [S].** DP needs no teacher or labels; any store of states is its corpus. If it worked, an equal-cost model would read like a larger one.

## 2. What I built [M]
- **DP corpus (`gen_dp.py`).** Questions are generated from each state's own text, with labels exact by construction.
  - **Verify:** an exact span, or the same span with one digit, number, date or same-type word replaced. This is replaced-span detection posed as a decision; half come as contrast pairs.
  - **Lookup:** the value after an anchor, against distractors of the same type.
  - **Order, author, count, number comparison, date comparison.**
  - 5,000 train-split states.
  - CF edit kinds are held out (far transfer); CF-probe's skills overlap DP's (near transfer).
- **Packing (`j4lib.py`).** Up to 40 questions branch off one state pass, each starting from the state's GDN state and conv tail.
  - Verified equal to the plain forward. With hobson's weights it gives JB-all .727, agreeing with hobson's references on .996.
  - That is one decision per 112 tokens, 3.1x the fine-tune's density, and every decision depends on the state.
- **Setup.**
  - Torso: hobson's own (Qwen3.5-2B-Base, the PEFT unload of v19), with LoRA r16 and the pointer head.
  - Fine-tune: the F7 recipe. Gold rows get CE plus KL(hobson); 5.5% are real train-split states with KL only.
  - Constant learning rate, so checkpoints form the curve. The trainer asserts zero eval tasks.
- **Arms:**
  - (a) fine-tune only;
  - (b) DP on 4.40M tokens, merged, then the same fine-tune; **bf** is b with a fresh head;
  - (c) (a) continued to b's total compute: 24.4k rows, 8.56M tokens;
  - (d) NTP on the **same** states and tokens, then the fine-tune;
  - (e) b with one DP state mixed into every fine-tune update.

## 3. Results [M]
All numbers are absolute; hobson is v19, trained on 115k rows.

**Sample efficiency: mean over seeds, range in brackets**

| | a@1k (2 seeds) | b@1k (2) | a@4k (3) | b@4k (3) | bf@4k | d@4k (NTP) | a@12k | b@12k |
|---|---|---|---|---|---|---|---|---|
| JB-all | .574 | **.626** | .632 [.62–.65] | .657 [.65–.67] | .680 | .654 | .688 | .671 |
| JB-hard | .396 | .412 | .410 | .444 | .469 | .438 | .492 | .454 |
| REAL-label | **.722** | .690 | .725 [.70–.74] | .720 [.70–.75] | .745 | .693 | .740 | .753 |
| CF flip | .268 | **.443** | .347 [.26–.46] | .365 [.32–.41] | .463 | .431 | .434 | .463 |
| CF-probe flip | .219 | **.484** | .264 [.22–.29] | **.483** [.38–.55] | .584 | .322 | .300 | .391 |
| SHUF both right | .252 | **.472** | .332 | .350 | .439 | .411 | .411 | .442 |

**End of budget: every suite, one seed**

| | hobson | a 12k | b 12k | c 24.4k (= b's compute) | e 12k |
|---|---|---|---|---|---|
| JB-all | .723 | .688 | .671 | **.727** | .684 |
| JB-hard (McNemar p against hobson) | .523 | .492 (.58) | .454 (.18) | **.546** (.66) | .485 (.53) |
| REAL-label (p against hobson) | **.785** | .740 (.005) | .753 (.053) | .753 (.041) | .715 (.0002) |
| REAL / LONG agree_sd | 1 / 1 | .844 / .897 | .835 / .873 | .896 / .873 | .809 / .788 |
| CF acc / flip | .594 / .268 | .670 / .434 | .682 / .463 | .636 / .367 | **.730 / .559** |
| CF-probe acc / flip | .653 / .328 | .645 / .300 | **.694 / .391** | .681 / .363 | .691 / .384 |
| Brier JB / REAL-label | .348 / .347 | .406 / .402 | .412 / .380 | .355 / .388 | .422 / .400 |

**What survives the noise**
- **Seed noise is large.** Two seeds of a at 4k differ on CF pairs by 71 against 17. REAL-label moves 4.5 points between seeds.
- **Robust: CF-probe about doubles at every budget up to 4k.** It reaches .48 against .22–.26 (threshold-free dmean .345 against .176), above hobson's .328. Amount-against-limit goes from .05 to .77.
- **At 1k rows (two seeds each), DP also lifts:**
  - far-transfer CF, .443 against .268 (human_insert .29 → .69);
  - SHUF, .47 against .25;
  - JB-all, by 5 points.
  - REAL-label falls 3 points.
- **At 4k rows (three seeds), most of that is gone.** CF (dmean .251 against .235) and REAL-label are equal. JB-all (+2.5) and JB-hard (+3.4) sit at the edge of the seed range.
- **The gain lives in the torso.** bf (fresh head) is at least as good as b at 4k.
- **The objective matters.** NTP on the same tokens gives the same JB gain, but b beats d on CF-probe (81/8), and NTP costs real traffic: REAL-label .543 at 1k and .693 at 4k (b ahead 93/38 and 33/11).
- **At equal compute, DP loses on general accuracy.** b against c, paired counts: JB-all 10/23 (p .035), JB-hard 9/21 (p .043), REAL-label 22/22. c ties hobson on JB using 21% of hobson's rows.
- **The fine-tune erodes DP.**
  - b's CF-probe falls from .55 to .39 between 4k and 12k rows; amount-against-limit falls from .79 to .05. KL to hobson pulls the student toward a teacher that cannot read these details.
  - Mixing DP into every update (e) raises sensitivity (CF .559, SHUF .545) but still loses the probe skill (.384) and costs REAL-label (.715).
- **Never moved:** identity, procedure_intent and insists (about 0), date_order (≤ .16).

## 4. Latency [M]
- **Model measured:** b, with every LoRA merged, which is exactly hobson's shapes.
- **Setup:** TTL fused runtime in hobson's layout (the state pass, then each question as a cached branch), CUDA graph, exclusive GPU, fresh real states, n = 20. p95 is within 0.13 ms of the median.

| state tokens | 64 | 128 | 256 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|
| 1 question (ms) | 19.0 | 19.4 | 25.8 | 37.4 | 63.1 | 212.1 |
| 4 questions (ms) | 48.6 | 48.9 | 55.6 | 67.3 | 93.4 | 243.9 |

- The same harness measured hobson at 64.0 ms (1000 tokens, 1 question); DP changes weights, not FLOPs.

## 5. Projections (arithmetic)
- The cost is identical to hobson's: about 27 ms on a 3090 for the bf16 state pass at 1000 tokens, 14 ms on a 4090 and 10 ms on a 5090; with W8A8, about 18, 11 and 7 ms.
- DP pays only via a smaller model [S].

## 6. Verdict
- **Partly confirmed, not transformative [M].** Decision-shaped self-supervision teaches near-transfer detail reading: CF-probe doubles and beats hobson with 1/30 of its labelled rows. Below about 1k labelled rows it is a strong head start.
- **The objective, not the data, does it:** NTP on the same states does not, and costs real traffic.
- **It does not raise general decision accuracy at a realistic budget.** At equal compute, ordinary fine-tuning wins on JB and ties on REAL-label, and distillation toward hobson erodes DP's skills.
- **No arm meets the bar.** REAL-label ≥ .78 fails everywhere; the best is .773, c at 16k rows.
- **Capability per parameter is not shown.** DP is a detail-reading add-on, not a substitute for decision data.

## 7. Single decisive next step
- Run the full 115k-row hobson recipe at **0.8B and 2B**, three seeds, with DP as a constant auxiliary loss.
- Replace KL(hobson) on detail-bearing rows with construction-label CE, keeping KL only where hobson is reliable.
- Kill it unless the 0.8B arm matches hobson-2B on JB-hard and REAL-label (≥ .78) with CF-probe ≥ 1.5x hobson. That is the only outcome that buys a cheaper model.
