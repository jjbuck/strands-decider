# J11: reinventing the transformer for typed decisions, from scratch, at matched compute

Labels: **[M]** measured on A10G box j11; **[V]** verified by an explicit check; **[S]** speculation or projection. Data is in `~/decider2/j11/` (`results.json`, `runs/*/`, `lat_*.json`, `NOTES.md`); code is in `code/`.

## 1. Hypothesis
A decoder spends about 2N FLOPs on every token at every depth. A decision reads only K+1 rows, needs a few exact facts whose location depends on the question, and uses questions fixed per deployment. A decision-native network should:
- put depth on the options, not on the tokens;
- retrieve exact values;
- compile the question, so the state never depends on it;
- treat options as a set.

This changes the cost class: per-token depth L/3, decision depth O(K), nearly free extra questions, and order invariance by construction.

Every design that failed in sections 5 and 7 sent less of the state through the model. None of these drops, pools or routes a state row: every token stays addressable by every decision layer.

## 2. What I built
All arms are trained from scratch with a pointer readout, and non-embedding parameters are equal within a scale.
- **dec (control):** a causal decoder in hobson's layout.
- **slot:**
  - L/3 bidirectional layers read the state with no question in view;
  - the question stem and each option (numbering removed) are encoded separately;
  - a deep stack of slot layers (answer, K options, 4 scratch slots) attends jointly over [state; stem; options; slots].
  - It is not the failed reader retrofit: there is no frozen model behind an interface.
- **vslot2:** slot plus exact value channels.
  - A typed parse gives date ordinals, fp32 amounts, canonical hashes for ids and phones, and field and record bindings.
  - fp32 pointer retrieval with equality, signed-difference, magnitude and count features.
  - Binding priors favour state tokens equal to the slot's own literals, and that record.
  - Nothing is pooled, and there is no planner. vslot (v1) has the same operators without the priors.
- **belief:** a weight-tied slot block run for 3 rounds, with option beliefs fed back. A round costs O(K), not O(T).
- **decv:** the decoder plus vslot's input channels, without its operators (attribution).
- **Rejected on paper:** late-interaction energy scoring, which is the dual encoder that already failed.

**Protocol [M].**
- **Budget.** One deterministic step stream and an equal analytic training-FLOP budget per arm (3x forward, attention and auxiliary losses included). Cheaper arms therefore see 1.5–1.9x more decisions.
- **Corpus:**
  - 70% exact-label synthetic decisions over conversations plus tool-result JSON (8 families);
  - 15% train_v5 (gold + hobson KL);
  - 15% real train-split states with hobson KL (6,000 from g3 plus 6,747 I labelled).
- **Held-out families:** 3-condition policy, 3-hop chains, long distractor states, an unseen schema.
- **Scales** (non-embedding parameters, budget): XS d448/L8 20M at 5 PF; S d512/L12 38M at 10 PF; M d768/L16 114M at 16 PF.

**The trap, hit and fixed [M].** My first control learned nothing, and the slot arms could not memorize 64 examples. Applying the same fixes to every arm made all of them learn:
- a training-only dense auxiliary loss (next-token for the decoder, masked-token for the encoders), with its FLOPs counted;
- qk-norm;
- unit-scale scratch slots;
- compact synthetic states.

## 3. Results [M]
- **Synth** is accuracy on 2,400 iid / 1,200 held-out items (SE about 1 point, chance about .38).
- **|dp|** is the mean |Δp| under 3 option rotations.
- **GF / ms** is inference GFLOPs and A10G latency at 1,000 tokens.

| run | synth | JB-all | JB-hard | REAL agree_sd | CF / probe flip | REAL-label | ECE (CF) | \|dp\| | GF / ms |
|---|---|---|---|---|---|---|---|---|---|
| hobson | – | .723 | .523 | 1 | .268 / .328 | .785 | .114 | .884 unch. | ~2740 / 52.7 |
| XS_dec | .461 / .374 | .342 | .415 | .350 | .000 / .003 | .675 | .215 | .012 | 53 / 3.94 |
| XS_slot | .463 / .411 | .372 | .431 | .410 | .000 / .034 | .698 | .196 | .001 | 15 / 1.71 |
| XS_vslot2 | .475 / .391 | .342 | .415 | .344 | .042 / .044 | .625 | .177 | .001 | 15 / 2.58 |
| S_dec | .466 / .394 | .316 | .369 | .457 | .002 / .019 | .710 | .209 | .011 | 100 / 6.94 |
| S_decv | .475 / .401 | .303 | .338 | .382 | .002 / .013 | .677 | .180 | .012 | 100 |
| S_slot | .459 / .386 | .359 | .415 | .390 | .012 / .016 | .675 | .181 | .001 | 36 / 3.17 |
| S_belief | .449 / .388 | .325 | .400 | .442 | .000 / .031 | .695 | .211 | .001 | 38 / 6.17 |
| S_vslot2 | .496 / .406 | .338 | .392 | .327 | .074 / .056 | .650 | .144 | .002 | 37 / 4.37 |
| M_dec | .442 / .394 | .299 | .362 | .382 | .000 / .025 | .645 | .212 | .010 | 286 / 13.3 |
| M_slot | .450 / .395 | .333 | .408 | .358 | .015 / .019 | .672 | .227 | .001 | 93 / 5.53 |
| M_vslot2 | .452 / .399 | .368 | .446 | .384 | .005 / .069 | .677 | .168 | .002 | 94 / 7.37 |

- **Against hobson.** McNemar on JB-all gives p ≤ 1e-12 for every run; JB-hard gives p = .006–.27 (n = 130, underpowered). CF and CF-probe accuracy are .44–.51 (hobson .594 / .653), and LONG agree_sd is .41–.63. Nothing passes the bar: these are 20–115M models trained on 13–35M tokens. vslot v1 sits within 2 points of slot except REAL agree_sd (.303).
- **Families.**
  - vslot2 on argmax: .59 / .67 / .64 against dec's .41 / .43 / .42 at XS / S / M (z ≈ 6). At S it reached .60 by 1.9 PF, so it is more than 5x as compute-efficient there.
  - decv on argmax is .45, so the gain comes from the binding operator, not from the channels.
  - vslot2 gains +7 to +9 on 2-hop chains and loses −3 to −11 on status. Every arm is at chance on 3-hop chains.
- **Order [V].** In fp32 at batch 1, S_slot gives max |Δp| 6.3e-7 over 322 rotation pairs, with 100% of decisions unchanged: exactly invariant. On confident decisions (margin > .1), dec is .997 unchanged as well. Raw unchanged rates are dominated by near-uniform ties.

## 4. The slope (deliverable)
- **The control does not scale here.** Synthetic accuracy is .461 / .466 / .442 at XS / S / M. Learning curves are flat after 2–4 PF, and M_dec (615 steps, about 20M tokens) is token-starved. Every arm's scaling exponent is about 0, so a law comparison is impossible at ≤16 PF from scratch. That is the measured reason.
- **Margins over dec at equal training FLOPs** (points, XS / S / M):
  - slot on synthetic: +0.2 / −0.7 / +0.8;
  - vslot2 on synthetic: +1.4 / +3.0 / +1.0;
  - slot on REAL agree_sd: +6.0 / −6.7 / −2.4.

  None is a factor, and none grows monotonically.
- **At equal accuracy,** the slot arms match the decoder at 0.28–0.37x its inference FLOPs (XS_slot at 15 GF is at least as good as XS_dec at 53 GF on every suite). That is a 2.7–3.6x factor, but in cost, not capability. At equal inference FLOPs, M_vslot2 against S_dec is mixed: JB-hard +7.7, REAL agree_sd −7.3.
- **belief** recovers most of slot's real-traffic deficit at S (agree_sd .442 against .390; dec .457), but it is latency-bound.

## 5. Latency [M], projections [S]
A10G, batch 1, CUDA graph, exact lengths, fresh inputs, 20 warm reps; p95 is within 0.1 ms everywhere. Cells are median ms with 1 question (4).

| model | 64 | 256 | 1000 | 4000 | 8000 |
|---|---|---|---|---|---|
| dec S | 2.91 (3.97) | 3.59 (4.69) | 6.94 (8.26) | 23.9 (26.2) | 50.4 (54.6) |
| slot S | 1.47 (1.55) | 1.82 (1.92) | 3.17 (3.49) | 11.4 (12.3) | 25.4 (27.1) |
| dec M | 4.64 (7.15) | 5.81 (8.42) | 13.3 (16.4) | 48.9 (55.2) | 103 (114) |
| slot M | 2.22 (2.38) | 2.77 (2.98) | 5.53 (6.04) | 20.4 (22.0) | 46.4 (49.4) |

- The slot design is 2.0–2.4x faster at every length.
- Four questions add +5–10%, against +19–54% for the decoder's shared-prefix path.
- At 4,000 tokens the FLOP ratio rises to 0.41–0.47x (the encoder's bidirectional attention).
- vslot's pointer ops add 0.9–1.8 ms (4.37 ms at S); belief's 27 sequential slot layers cost 6.2 ms.
- **Projection [S]** (GEMMs scaled by peak, the rest by bandwidth): M_slot at 1,000 tokens is 3.2 / 2.5 / 1.5 ms on a 3090 / 4090 / 5090, against M_dec's 7.6 / 5.2 / 3.3.

## 6. Verdict
- **Cost class: survives [M][V].** Moving decision depth from tokens to option slots loses nothing on the synthetic aggregate at equal training compute, at three scales. It needs about 3x fewer inference FLOPs, runs 2.0–2.4x faster, makes extra questions nearly free and makes option order exactly irrelevant.
- **"Beats the decoder by a factor": negative [M].** Exact operators give large, scale-stable gains only where the task is the operator (argmax). Elsewhere they move results a few points either way.
- **"Margin grows with scale": not supported [M].** At ≤16 PF from scratch every arm is limited by the number of decisions seen and plateaus (synthetic .44–.50; REAL-label .63–.71 against a no-state baseline of .635).
- **Lost time.** A box outage (RAM exhaustion during prep) and the trap cost about 3 hours.

## 7. Decisive next step
Rerun S_dec, S_slot and S_vslot2 at 10x the budget (about 100 PF, 2.5 A10G-hours each) on the same stream to leave the plateau.
- **If slot stays on the decoder's curve** at 0.35x inference FLOPs, the cost class is confirmed, and pretrained 2B (J3) is the deployment path.
- **If the decoder pulls away on real traffic,** the slots belong on top of a deep state encoder.
