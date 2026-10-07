# K2 notes (runtime and latency for the combined stack)

## 2026-10-06 19:15 PDT
- Read BRIEF9.md and IDEAS_EXPLORED.md. Box k2 not ready yet (no ~/decider2/boxes/k2.ready). k1/ is empty (no stacklib.py yet).
- Plan: read QRT (h2), J5 kernels, J9 cache, J6 deltas, J3 depth split, J7 super-tokens, J15 exit, J14 state-first runtime; then write stackrt.py parts that do not depend on stacklib.

## 19:40 design (before K1's stacklib exists)
- Base: H2 QRT on J5 kernels (j5rt.make: 'bf16' = FoldSK with sk GEMMs; 'w8a8:precmap_w8a8_b8' = IntSK, CUTLASS small-M + sk, per-shape choice).
  Own layer loop (stackrt.StackRT) so that the row set can change at layer 8 (D) and stop at 16 (X); same kernels as QRT (_conv_k, _gnorm_k, _agate_k, _aprep_k, _swiglu_k, _addq_k).
- Rows per request: [live state rows][question branches]. Branch = in-context question tokens (causal within, sees state; GDN from the state's final state)
  or a deployed question's K+1 slot rows (Q) with its stacked LoRA delta (shared r16 + own r8) on Win/Wo/Wd of slot rows only.
- C: prefix = compiled rows (frame + blocks). Per request, inside the graph: gather the request's rows from the deployment library (post-norm pre-RoPE K, V),
  RoPE at runtime positions, write into the K/V buffer ahead of the live rows. GDN: K/V-only library (blocks GDN-invisible, per J9 final) -> live state starts
  from the frame's state, conv tail from the last compiled rows. Will switch to whatever K1's stacklib does.
- D: state rows run layers 0-7; one memory GEMM over the state rows' layer-8 input (folded norms, bf16 or int8 on the same quantized codes as layer 8's Win input)
  gives K/V for deep attention layers (11,15,19,23); deep layers run branch rows only; deep GDN from zero state (bridge A).
- X: graph A = layers 0-15 + exit head (J15 EH, r512); graph B = layers 16-23 (+ deep memory K/V of 19/23 under D) for the questions that did not exit.
  Timed cascade = A + D2H + host margin + (B if any question of the request did not exit).
- W8A8 LoRA deltas (Q) read the dequantized int8 input codes of their GEMM, rotations folded into A/B; Win delta injected into the fp16 alpha-scaled output rows.
