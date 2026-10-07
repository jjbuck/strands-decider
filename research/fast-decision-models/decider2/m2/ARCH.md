# M2 layout for timing (v2, 2026-10-07 06:45 PDT; agent M2)

**Status (measured on the A10G, full evalkit):** trained 2.3-2.6 h. With every segment isolated (layout N) and state depth k = 12 the
model passes the strict accuracy bar; with only documents isolated (layout C, the dynamic text one causal stream reading the documents
first) and k = 12 it passes the relaxed bar. k = 8 fails JevBench in both. So time **k = 12 and k = 8**, blocks as below. For layout C,
the live state rows are one causal stream (keys [U | compiled documents | live rows], lower-right causal), not blocks; its GDN is one
scan per layer from the documents' folded state.

This is the layout M2 is training and timing on the A10G. N1: random weights of these shapes are enough for latency.
hobson-v19 shapes throughout (24 layers, d = 2048; GDN layers 16 heads x 128; attention layers 3, 7, 11, 15, 19, 23 with 8 query heads x 256
and 2 KV heads x 256; SwiGLU 6144). No new weight matrices except small LoRA-style adapters that merge into existing GEMMs.

## Rows
- **U**: the 4 tokens of `<state>\n`. A constant: compiled once (its K/V per attention layer, its GDN state S_U and conv tail per GDN layer).
- **State rows**: the state's content tokens, split into **segments**. For timing use fixed **B = 256-token blocks** (B = 128 as the
  alternative); the last block may be partial. (Real requests use natural segments: messages, tool outputs, KB documents, hook notes;
  mean 180 tokens on real traffic, 117 on CF-probe.)
- **Question rows**: per question, `</state>\n` (4 tokens) + the question tokens (85-125 tokens; options; `<answer>` last).

## State rows: layers 0 .. k-1 only (k = 8 or 12), then frozen
Per state layer, per block (all blocks independent and identical in shape -> one batched op):
- Projections, MLP: the usual GEMMs over all live state rows (M = number of live state rows).
- Attention layers (3, 7 for k = 8; 3, 7, 11 for k = 12): each block's queries attend to [U (4 keys) + the block's own keys], causal
  within the block. RoPE at **block-local positions** (U at 0..3, block rows at 4..4+B-1). Shape per block: Q [8, B, 256], K/V [2, B+4, 256].
- GDN layers: causal conv (kernel 4) with history = U's last 3 rows for every block; delta-rule recurrence per block, every block
  starting from S_U (the same [16, 128, 128] fp32 state). So a 4000-token state is 16 independent 256-token recurrences (2-4 chunks each),
  not one 63-chunk chain. Outputs: per block its rows' outputs.
- Extra per GDN layer for the question read: the state's composed GDN state = one native-order scan over all live state rows starting
  from S0 (= S_U, or the fold of compiled documents, see below). Only the final state [16, 128, 128] is needed.

## Deep layers k .. 23: state rows frozen (J3's bridge G)
- One GEMM over the frozen state rows (layer-k residual) for every deep attention layer's K/V: [M x 2048] x [2048 x 1024 * n_deep_attn].
- Per deep GDN layer: one GEMM [M x 2048] x [2048 x 6176] (qkv | b | a), conv, one scan over the rows (final state only).
- No Wo / MLP for state rows in deep layers.

## Question rows: all 24 layers
- Every question is a varlen branch (J3's q_pass), causal within itself.
- Attention layers: queries [8, R, 256] attend to ALL state keys (U + compiled documents + live rows, at their global positions) plus
  their own branch. Keys = state length T (+ R).
- GDN layers: conv with the state's last 3 rows as history; recurrence from the composed state.
- Pointer head on the option rows and the `<answer>` row.

## Compiled documents (the precompilation benefit)
KB documents and hook notes are isolated segments, so their rows depend only on [U + their own tokens]: compiled once per deployment.
Per compiled document: per attention layer K (block-local RoPE; re-rotated by one scalar offset at request time) and V; per GDN layer its
transfer A_d and end state E_d from S_U ([16, 128, 128] each). Per request: S0 = fold over the request's documents
(S <- A_d (S - S_U) + E_d), a few small batched matmuls; the documents' rows skip every GEMM. 46.5% of real gate-state tokens are such
constants (J9); with this layout they are exact, not approximate.

## Timing grid (what M2 measures on the A10G; please match on inf2)
T (state tokens) = 64, 256, 1000, 4000; 1 and 4 questions (125 / ~460 question tokens); k = 8 and 12 (and 24 = isolation only);
compiled share c = 0 and 0.55 (the first 55% of the state compiled, in whole blocks).
