# M2: decision-native layout — isolated state segments, shallow state rows, global question readers (box m2, A10G)

Labels: [M] measured, [V] verified, [A] arithmetic. Files: `~/decider2/m2/` (NOTES.md, ARCH.md, code/, res/).

## What I built
- **Segmenter** (`m2seg.py`). Cuts a state into natural segments: header sections, messages, tool calls, tool results, KB documents,
  hook notes, and paragraphs of free text. A document's rank and score become their own tiny segment, so its tokens are the same in
  every request. 0 mismatches in 750 states [V].
- **Isolated forward on hobson** (`m2lib.py`).
  - State rows attend only to U (`<state>\n`, a 4-token sink) and their own segment, at segment-local positions.
  - GDN conv and recurrence run per segment, each starting from U's state.
  - Question rows read every state row in every layer: attention over all keys, and GDN from an exact composition of the segment
    states (J9's affine fold).
  - Options: per-layer isolation; state rows frozen after layer k (J3's bridge G, so M3 with a shallow state); top-k reads (M5).
  - Checks [V]: a token change in one segment moves another segment's rows by exactly 0.0; with every flag off the forward is hobson.
- **Fused runtime** (`m2lat.py`, d1/H4 kernels, CUDA graphs). U and documents are compiled once. Matches m2lib with compiled documents
  on 16/16 decisions; with the fused question pass 15/16, the 16th a tie at p .51 [V].
- **Distillation** (`m2train.py`): the brief's recipe, with depth k ∈ {8, 12} trained jointly. Two layouts:
  - **N:** every segment isolated.
  - **C:** only documents and hook notes isolated; the dynamic text is one stream that reads them first (J9's layout R, made exact).

## Training-free measurements (untrained weights; 966 eval items, all 400 REAL-label) [M]
| layout | REAL agree / state-dependent | REAL-label | CF / CF-probe retention |
|---|---|---|---|
| hobson (this runtime) | .995 / .990 | .787 | 1.00 / 1.00 |
| every segment isolated, all layers | .855 / .817 | .723 | .77 / .41 |
| isolated in layers 0-7 only | .879 / .817 | .743 | .77 / .43 |
| isolated in layers 8-23 / 16-23 only | .955 / .938 ; .996 / .995 | .787 ; .785 | .97 / .84 ; 1.00 / 1.00 |
| only documents + hook notes isolated | .944 / .889 | .785 | .97 / .84 |
| state frozen after 12 / 8 | .995 / 1.00 ; .904 / .803 | .782 ; .755 | 1.00 / .95 ; .71 / .57 |
| order-free sum instead of exact composition | .663 / .452 | .568 | .14 / .00 |
| top-16 / 64 / 256 keys for question rows (M5) | .939 / .875 ; .972 / .909 ; .991 / .971 | .777 ; .787 ; .787 | .97/.84 ; .97/.87 ; 1.00/.95 |

- **hobson does its cross-segment mixing in layers 0-7.** Isolating those layers costs 17 points of state-dependent agreement;
  isolating 16-23 costs nothing.
- **Segment-local positions cost nothing:** max|dp| .027 against native positions.
- **Only the exact composition works.**
- **Not done:** M4 summary slots were not built (time). M5 was measured untrained only.

## Trained layouts against both bars (all 3,227 questions; McNemar against hobson) [M]
| | hobson | N k8, 2.6 h | N k12, 2.6 h | N k12, 4.1 h | C k8, 2.3 h | C k12, 2.3 h |
|---|---|---|---|---|---|---|
| JB-all (p) | .723 | .649 (.02) | .675 (.12) | .693 (.35) | .654 (.01) | .697 (.26) |
| JB-hard (p) | .523 | .431 (.08) | .454 (.19) | .477 (.42) | .423 (.03) | .485 (.33) |
| REAL-label (p) | .785 | .792 (.70) | .780 (.86) | .787 (1.0) | .780 (.85) | .770 (.35) |
| REAL agree / state-dependent | 1 / 1 | .924 / .844 | .925 / .864 | .940 / .870 | .923 / .853 | .924 / .870 |
| CF / CF-probe pair accuracy | .268 / .328 | .704 / .553 | .697 / .541 | .719 / .537 | .714 / .981 | .702 / .981 |
| Brier, REAL-label | .347 | .348 | .352 | .355 | .344 | .349 |
| RuleTaker d3 / d5 | .860 / .720 | .713 / .580 | .740 / .607 | .807 / .620 | .747 / .633 | .800 / .687 |
| strict bar | | fails | **passes** | **passes** | fails | fails (REAL-label) |
| relaxed bar | | fails | fails (JB-all) | fails by 1 JB item | fails (JB-all) | **passes** |

- **At k = 12 every layout passes one bar.** N at 4.1 h passes strict, and misses relaxed by one JevBench item (160 of 231 right,
  .6926 against ≥ .693). C at 2.3 h passes relaxed and misses strict on REAL-label (.770 against ≥ .78; not significant against hobson).
- **At k = 8 both lose JevBench significantly.** Training N from 2.6 to 4.1 h raised its state-dependent agreement (.844 to .861) and
  lowered JevBench (.649 to .636).
- **Isolating the dynamic segments costs one skill: binding a record to its lookup.** On CF-probe items with a distractor record, N gets
  0.0 on id and status pairs (hobson .38 / .25); C gets .88-1.0, because its dynamic text still mixes.
- **Learning curves** (dev state-dependent agreement, k = 8): N went .75 → .91 at 2.6 h → .89 at 4.1 h; C went .85 → .91 at 2.3 h.

## Precompiled documents are exact [M]
On 60 banking requests (228 questions), every document and note segment (61% of state rows) was computed standalone and spliced in:
- hobson's own layout (A6's error): 7 decisions change, max|dp| .26;
- trained N and C (k = 8): 228/228 unchanged, max|dp| .0057 / .0076 (bf16 rounding).

## Latency against hobson [M]
A10G, bf16, CUDA graph per exact shape, 15 warm reps, fresh inputs. Medians in ms; p95 within 0.1 ms. One question, with the question
as one fused cached pass (as hobson's own). "c" = share of the state compiled.

| T | hobson | N k12, c 0 | N k12, c .55 | C k12, c .55 | N k8, c .55 | C k8, c .55 | J3 depth-8 (no isolation) |
|---|---|---|---|---|---|---|---|
| 64 | 15.3 | 18.0 | - | - | - | - | 15.8 |
| 256 | 22.9 | 22.1 | - | - | - | - | 18.4 |
| 1000 | 57.1 | 49.4 | 34.8 | 33.1 | 29.8 | 28.7 | 36.3 |
| 4000 | 204.7 | 156.3 | 89.9 | 87.0 | 75.0 | 73.0 | 110.1 |

- **Four questions** (J3's question pass): hobson 96.1 / 246.8 ms at 1,000 / 4,000 tokens; N k12 at c .55 68.1 / 126.3.
- **J9's 84 real requests**, mean ms (A6 = J9's runtime re-run on this box: 98.4 against J9's 98.8):

  | | hobson | A6 | N k12 | C k12 | C k8 |
  |---|---|---|---|---|---|
  | all 84 | 160.1 | 98.4 | 94.0 | 88.0 | 75.1 |
  | 59 banking | 178.8 | 97.0 | 94.3 | 88.8 | 76.3 |
  | 25 retail | 116.0 | 101.8 | 93.5 | 86.2 | 72.2 |

  On banking that is 2.01x for C k12 (per-request median 2.05x) and 1.90x for N k12, against A6's 1.84x, which is inexact.
- **Isolation glue** (gathers, the composition scan, unfused) costs 5 ms at 1,000 tokens and 18 ms at 4,000 against J3's depth split
  at the same depth. That is why N k12 without documents is 1.16x faster than hobson at
  1,000 tokens, against 1.33x for J3's 12-layer split.

## Projections (J9's method) [A]
- **1,000 tokens, N k12 at c .55:** 18.4 / 9.8 / 7.0 ms on 3090 / 4090 / 5090, against hobson's 29.6 / 14.6 / 10.7.
- **4,000 tokens:** 46.1 / 21.6 / 16.3, against 103.8 / 46.2 / 35.8.
- **8-bit runtime:** W8A8's measured 1.57x end to end on hobson (P1) would multiply these.

## What it means
- **The premise holds in hobson's late layers.** Untrained, state rows frozen after layer 12, or isolated in layers 16-23, change
  nothing; question rows do the reading.
- **It does not hold in layers 0-7.** hobson needs early cross-segment mixing. Distillation recovers it at k = 12 but not at k = 8,
  and fully isolated dynamic segments lose record binding.
- **What works now is constants isolated and dynamic text mixed (C).** It is exact precompilation (A6 without its error) plus a
  12-layer state: 2.0x on real banking traffic, with relaxed-bar accuracy after 2.3 h.

## Single most valuable next step
Train C at k = 12 for the full 6-hour budget, then run it in the 8-bit runtime.
- **Why C:** it misses the strict bar by one REAL-label point (.770 against .78; not significant against hobson's .785).
- **What to watch:** JevBench. That is where k = 8 failed, and where N at k = 8 fell as it trained longer (.649 to .636).
- **Payoff [A]:** exact compiled documents × a 12-layer state × W8A8 ≈ 3x on banking traffic.
- **For N1:** ARCH.md has the shapes. Per-block GDN turns a 4,000-token state's one dependent chain into 16 independent
  256-token chains [A]; N1 is timing that on inf2.
