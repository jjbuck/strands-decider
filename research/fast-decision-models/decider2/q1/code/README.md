# Q1 code (box q1: ~/work/q1). APIs for other agents (copy, never edit in place)

Box layout: ~/work/g2/g2lib.py + subsets.json (from ~/decider2/recovered/g2/g2), ~/work/h1/ (H1 code; Hessians in ~/work/h1/hess from
`h1calib.py --n 64`), ~/work/q1/ (this folder), ~/work/evalkit (bundle).

## q1lib.py
- `Q1(grad=True, **h1opts)`: H1's lean hobson-v19 (merged LoRA, fused weights). `grad=True` clones weights out of inference mode.
  - `g.fwdg(ids, gfrom=0, q0=None)`: differentiable forward; layers < gfrom run without grad, the residual becomes a leaf at gfrom.
    Returns the final normed hidden [T, 2048]. `g.fwd(ids, q0=...)` (H1, no grad) also goes through the same per-GEMM dispatch.
  - `g.logits_g(h, pr)`: differentiable pointer-head logits (temperature applied, first n_slots). `g.logits`/`g.pdict`: H1 no-grad versions.
  - Per-GEMM dispatch in `g.lin(i, k, x, xn)`: `g.qfn[(i,k)] = f(g, i, k, x, xn) -> y` replaces the GEMM (x = bf16 dense input,
    xn = fp32 gain-free RMSNorm input for Win/Wgu, None for Wo/Wd). `g.fwd_fn(i, k, x, xn, y)` is called after every GEMM;
    `g.hook_fn(i, k, dL/dy)` on backward (hooks are always registered when y requires grad). `g.track` = set of (i,k) or None (all).
  - `g._q0`: state/question row boundary used by formats (set by fwdg/fwd q0=).
- `req_sets(nA=256, nB=64)`: train-split requests (eval tasks excluded), one random question each; A and B from disjoint tau tasks.
- `dev_set(n, skip_rids)`: more train-split requests from B's tasks. `fisher_dirs(lg)`: softmax-Fisher eigenpairs of the logits.
- GEMM names: Win (GDN in_proj qkvzba [8224] / attn q+gate,k,v [5120]), Wo [2048], Wgu (gate+up) [12288], Wd [2048, K 6144].

## q1spec.py (B1.1 spectrum) -> ~/work/q1/spec/
- `L{i}_{k}.pt`: dict keyed (side, role) with side in {'out','in'}, role in {'s','q','all'}: 'ev' (all eigenvalues, descending),
  'V' (top-1024 eigenvectors, unrotated basis; present for every input side, and for output side when N = 2048 or role 'all'),
  'G' (full input-side matrix for K = 2048, roles s and q). G is a per-request mean of sum over rows and Fisher directions.
- `grp{g}.json`: per-GEMM summaries, row-energy shares, per-request meta. `heldout.json`: capture of set B's trace by A's eigenvectors.
- Laptop summary: `q1spec_summary.py` -> ~/decider2/q1/sensitivity.json.

## q1fmt.py (format emulation)
`Fmt(g, spec)`; `fm.install(g)` sets g.qfn for all 96 GEMMs. Spec keys (see docstring): a_s/a_q (4|8|16 bits for state/question rows),
wq ('gptq'|'rtn'), qw8 ('w8'|'w4'), map (per-GEMM override), clip4, group (g=64 group-scaled), corr (B1 low-rank correction),
split (ResQ-style or sensitivity subspace at int8), tc (B5 transform-coding plan tag), dither ('sr'|'tpdf', B2), proto (B3), b9 (tile
sampling), wonly/aonly (diagnostics), ba16. GPTQ codes cached in ~/work/q1/codes. `fm.work()` gives MAC shares per row role.

## Other scripts
- `q1cal.py`: per-GEMM input mean / second moment by role, crest factors, row samples -> ~/work/q1/cal/.
- `q1b5.py`: B5 planner (bases, {8,4,0}-bit slices) -> ~/work/q1/b5/{tag}/P_{i}_{k}.pt.
- `q1fo.py`: first-order pass for one format (predictor, coherence, certificate, token classes, positions, B10) -> ~/work/q1/fo/{tag}.json.
- `q1hd.py`: decision-weighted input Hessians and MLP-neuron saliency (written before B11/B12 moved to Q5) -> ~/work/q1/hd/.
- `q1eval.py`: all 3,227 evalkit questions for configs in cfgs.json -> preds JSON. `q1score.py` (laptop): bar check.
- `runq.sh` + `queue.txt`: sequential GPU job queue on the box.
