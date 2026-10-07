# Q2 FORMATS: the exact arithmetic of each GEMM kernel (v2, 2026-10-06 22:00 PDT)

Q1 and Q3 emulate these. Notation for every GEMM: input rows x_t (t = 0..M-1, K values each, fp32 inside the producing kernel), weight W [N, K] (out x in, the rotated, gain-folded matrix of H1 FORMAT v1), output y = x W^T [M, N]. "fp32(a*b)" means one IEEE multiply rounded to nearest even; no fused multiply-add anywhere unless written "fma". bf16() and fp16() are round-to-nearest-even casts. All integer accumulation is exact int32 (order does not matter). Status: [built] / [planned]; the kernels and their torch references are in `~/decider2/q2/code/` (box: `~/work/q2/`).

**What is bit-exact.** Everything integer, every scale, every rounding listed here is reproducible in torch with float32 tensors and eager ops (do not use torch.compile, addcmul or matmul for steps marked fp32 elementwise; they may fuse or reorder). The only order-dependent parts are fp32 sums computed on tensor cores (marked "fp32 tensor-core sum"): emulate them with an fp32 matmul; expect relative differences of ~1e-6 of that term.

## 0. The activation quantizer A(y; qmax, clip, mode), used by every format

For one row y (K or a group of g values, fp32):
1. amax = max |y_k|
2. s = fp32( fp32( max(amax, 1e-8) / qmax ) * clip )        (division correctly rounded)
3. v_k = fp32( y_k / s )                                      (correctly rounded)
4. codes:
   - RTN: q_k = clamp( rint(v_k), -qmax, qmax ), rint = round half to even (= torch.round)
   - SR (B2): q_k = clamp( floor( fp32(v_k + u_k) ), -qmax, qmax ), u_k from section 4
5. dequantized value: s * q_k. The scale s is stored as fp32.

Defaults: int8 qmax 127, clip 1.0; int4 qmax 7, clip 0.9 (H1/H2). torch: `s = (amax.clamp_min(1e-8) / torch.full_like(amax, qmax)) * clip; q = torch.round(y / s[:, None]).clamp(-qmax, qmax)`.

**Pitfall (measured 21:20):** `amax / 7.0` with a Python-scalar divisor is NOT an IEEE division in torch on CUDA (torch multiplies by the reciprocal), and it gives a 1-ulp different scale on some rows, which flips codes at rounding boundaries. Divide by a tensor (`torch.full_like`) as above. With that, the Triton prologues and every q2gemm variant match the torch references bit for bit (`q2/code/test_q2.py`, `bench_pro.py test`).

Weights: static, symmetric, per output channel n: codes w_nk in [-qmax, qmax], fp32 scale sw_n (GPTQ or RTN; offline, supplied by Q1/Q3). Group-scaled formats use sw_{n,g}.

## 1. The deployed H2 kernels (CUTLASS), as they run in QRT today [built, H2]

F0 W8A8 per token, F1 W4A4 per token (and the b8 / k48 maps that mix them with bf16 GEMMs):
1. acc_tn = sum_k q_tk w_nk (int32).
2. GEMM output fp16: C16_tn = fp16( alpha * float(acc_tn) ), alpha a power of two: s4 0.5 (K 2048), 0.125 (K 6144); s8 2^-10 (K 2048), 2^-11 (K 6144). (alpha = largest 2^-j with K*qmax_a*qmax_w*alpha <= 60000.)
3. The consumer kernel dequantizes: d_tn = fp32( fp32( float(C16_tn) * s_t ) * csa_n ), csa_n = sw_n / alpha (exact), then casts to bf16 where the bf16 model would (residual add: x = bf16(x + bf16(d)); conv / gates: bf16(d)).
4. gate_up (EVT epilogue, no fp16 stage): v = fp32( fp32( float(acc) * s_t ) * sw_n ); g = bf16(v_gate), u = bf16(v_up); m = bf16( bf16( g / (1 + __expf(-g)) ) * u ). __expf is the fast approximate exp, so this one step is not bit-exact in torch (use torch.sigmoid; differences are ~1 bf16 ulp on a few values).

The GEMM inputs y themselves come from H2's Triton glue (tl.rsqrt RMSNorm, online Hadamard with fp16 dots on the row-prescaled vector); emulations of y are close but not bit-exact. Everything after y is exact as written.

## 2. The Q2 kernel family: one output convention for all new formats

All Q2 kernels write the dequantized output directly (no fp16 stage):
- integer part: f_tn = fp32( fp32( float(acc_tn) * s_t ) * sw_n )
- then the format's extra terms (sections 3-8) are added to f in fp32, in the order given;
- output: bf16(f_tn). For gate_up the SwiGLU epilogue is m = bf16( bf16( g / (1 + expf(-g)) ) * u ) with g = bf16(f_gate), u = bf16(f_up), expf accurate (matches torch F.silu on fp32).

F2 = W4A4 per token in this convention [built when marked in the table at the end]. F2 differs from F1 only by the absence of the fp16 rounding of alpha*acc.

## 3. B1: int4 GEMM plus a low-rank correction (a "tail") [built: tail; Z by cuBLAS or q2gemm bf16]

The kernel computes, for any K x r' matrix A and r' x N matrix B_tail supplied per GEMM:

    Z = dX A  (+ diag(s) Q A_Q, optional)       Z [M, r'], fp32 tensor-core sum, stored as bf16(Z)
    y = bf16( f + Z B_tail )                    the tail is an fp32 tensor-core sum of bf16 x bf16 products added to f

- dX_tk = bf16( fp32( y_tk - fp32(s_t * q_tk) ) ): the per-token int4 rounding (and clipping) error of the GEMM input, written by the producing (quantize) kernel as bf16. Q is the int4 code matrix, s the per-token scales.
- Output-side subspace P [N, r] (orthonormal columns): A = W^T P (K x r), B_tail = P^T (r x N). Optional weight-error term: A_Q = (W - Wdeq)^T P with Wdeq_nk = sw_n * w_nk. Then y = int4 result + (E P) P^T with E = x W^T - (s Q)(Wdeq)^T, exactly the brief's formula.
- Input-side subspace P_in [K, r]: A = P_in, B_tail = P_in^T W^T. With weight error: r' = 2r, Z = [dX P_in | diag(s) Q P_in], B_tail = [P_in^T W^T ; P_in^T (W - Wdeq)^T].
- A, A_Q and B_tail are stored as bf16. Row-role variants (P per row role) use separate (A, B_tail) for state and question rows.
- Tail in int8 instead of bf16 (half the tail cost): Z quantized per row with A(Z; 127, 1.0, RTN) and B_tail per output column with absmax/127; f += fp32( fp32( float(acc_tail) * s_Z,t ) * s_B,n ). Reported separately as "int8 tail".

Cost: the tail adds M*r'*N multiply-adds inside the int4 GEMM (at bf16 rate, 4r'/K of the int4 main loop); Z adds M*K*r' outside it.

## 4. B2: stochastic rounding (dither) in the quantize prologue [built]

Codes use mode SR of section 0 with u_k from a counter hash (all arithmetic on uint32, wrapping):

    h = seed_g ^ (t * 0x9E3779B1) ^ (k * 0x85EBCA77)
    h ^= h >> 16;  h *= 0x85EBCA6B;  h ^= h >> 13;  h *= 0xC2B2AE35;  h ^= h >> 16
    u = float(h >> 8) * 2^-24            (in [0, 1), exact in fp32)

- t is the row index within the forward pass (0..M-1), k the input column (0..K-1, in the rotated basis that is quantized).
- seed_g = (seed_request * 0x9E3779B1 + gemm_id * 0x7F4A7C15) mod 2^32, gemm_id = 4*layer + {Win 0, Wo 1, Wgu 2, Wd 3}. seed_request is an input to the forward (a per-request integer).
- torch: use int64 tensors and `& 0xFFFFFFFF` after every multiply and xor; multiplication may wrap in int64, the low 32 bits stay correct.
- Rounding with uniform dither in (-1/2, 1/2) is the same distribution as SR, so SR is the one dither mode. It is unbiased (E[s q] = y for unclipped values) with error variance up to s^2/4 (RTN: s^2/12). Recommended clip 1.0 with SR so no value is clamped.
- Applies per row: the row-role format (section 8) uses SR on state rows and RTN int8 on question rows.

## 5. B3: prototype subtraction [built]

Per GEMM input: a codebook of Kc prototypes c_j in R^K (bf16 values), with tables supplied offline:
- int8 search codes: sc_j = fp32( max(amax(c_j), 1e-8) / 127 ), cq_jk = clamp(rint(c_jk / sc_j), -127, 127)
- hn_j = fp32 value of 0.5 * ||c_j||^2 (any offline computation; it is a table, the kernel only reads it)
- T_jn = bf16( c_j . W_n ) with the exact (unquantized) folded weight: the c W table.

Per row x_t (the producing kernel stores xb_t = bf16(x_t)):
1. Search: q8 = A(xb_t; 127, 1.0, RTN) with scale s8_t; dot_tj = sum_k q8_tk cq_jk (int32);
   score_tj = fp32( hn_j - fp32( fp32(s8_t * sc_j) * float(dot_tj) ) ); idx_t = argmin_j score_tj (ties: smallest j).
2. Residual: r_tk = fp32( float(xb_tk) - float(c_idx,k) ); codes and scale by A(r_t; 7, clip, RTN).
3. GEMM: f = fp32( fp32( float(acc) * s_t ) * sw_n ) on the residual codes; f = fp32( f + float(T_idx_t,n) ); y = bf16(f).

## 6. ResQ-style inner-dimension split (baseline) [built]

The input after the format's rotation is split x = [x_hi (first r columns) | x_lo (other K - r)]; the weight columns are split the same way.
- codes: x_hi by A(.; 127, 1.0, RTN) with scale s8_t; x_lo by A(.; 7, 0.9, RTN) with scale s4_t; weights: int8 per-channel scales sw8_n on the hi columns, int4 sw4_n on the lo columns.
- f = fp32( fp32( fp32(float(acc4) * s4_t) * sw4_n ) + fp32( fp32(float(acc8) * s8_t) * sw8_n ) ); y = bf16(f). (acc4 over the lo columns, acc8 over the hi columns.)
- r = 128 is the brief's comparison point; the kernel takes any r that is a multiple of 64.

## 7. Group-scaled int4 (baseline) [built]

g = 64 (also 128). Activations: A(.; 7, clip, RTN) applied to each group of g consecutive input columns of a row: scale sx_{t,G}. Weights: codes with scales sw_{n,G} per output channel and group G.
- acc_{tnG} = sum over the group's k (int32)
- p = fp32( sx_{t,G} * sw_{n,G} ); f = fp32( f + fp32( float(acc_{tnG}) * p ) ), for G = 0, 1, ..., K/g - 1 in increasing order, starting from f = 0
- y = bf16(f). acc_{tnG} is at most 64*49 in magnitude, so any fp32 matmul over one group's slice reproduces it exactly.

## 8. Row-role grouped GEMM (baseline and carrier for B1/B2 on state rows) [built]

Rows t < q0 (state) and rows t >= q0 (every question / answer / option row; in hobson's layout q0 = number of state tokens) use different formats in one launch:
- state rows: any of F2 (W4A4 RTN), SR (section 4), the B1 tail (section 3), group-scaled (section 7);
- question rows: W8A8 per token in the section 2 convention: f = fp32( fp32( float(acc8) * s_t ) * sw8_n ), y = bf16(f).
- Weights: one int4 copy and one int8 copy per GEMM (two copies). A one-copy variant derives the int4 weight from the int8 codes in the kernel: w4 = clamp( (w8 + 8) >> 4, -7, 7 ) (arithmetic shift, i.e. floor((w8+8)/16)) with sw4_n = 16 * sw8_n. Reported only if it is built.


### 8.1 Row-role as deployed in the runtime (q2rt.py)
Two runtime variants exist; both quantize question rows with A(.; 127, 1.0, RTN) and state rows with A(.; 7, 0.9, RTN) inside H2's glue kernels (qk2.py).
- **QRT2** (q2gemm only): every GEMM is one two-problem q2gemm launch; f exactly as in section 8, y = bf16(f) (SwiGLU epilogue for gate_up). Consumers multiply by unit scales (exact).
- **QRT2C** (the fast one): state rows run H2's CUTLASS W4A4 kernel, i.e. exactly H1/H2's deployed W4A4 arithmetic of section 1 (fp16(alpha4 * acc4), then the consumer's ((C16 * s_t) * csa4_n)). Question rows run q2gemm int8 and are written in the same fp16 units: C16_tn = fp16( fp32( float(acc8_tn) * g_n ) ), g_n = fp32( fp32( sw8_n * alpha4 ) / fp32( sw4_n * 32 ) ), and the glue stores 32 * s8_t as their activation scale; the consumer then computes ((C16 * 32 s8_t) * csa4_n). Relative to section 8 this adds one fp16 rounding (2^-11 relative) to question rows, the same kind of rounding H2's deployed int8 path has. gate_up question rows: q2gemm SwiGLU epilogue with f = fp32( fp32( float(acc8) * 32 s8_t ) * fp32(sw8_n / 32) ), which equals section 8's f exactly (powers of two).
- Per-GEMM maps: any GEMM can instead be 'w8' (all rows W8A8; QRT2C runs these on H2's CUTLASS s8 / EVT, section 1 arithmetic) or 'w4' (all rows W4A4); `q2/code/q2map_k{24,48,64}rr.json` put H1/H6's most sensitive 24/48/64 GEMMs at 'w8' and the rest row-role.
- **Measured through these kernels (all 3227 questions, 22:24):** k64rr passes the BRIEF10 fidelity bar (REAL flips 0.65%, McNemar vs bf16 runtime p .51, CF 1.000, CF-probe .962, JB-hard 0/0, REAL-label .790, TV .0058); k48rr 1.29% / CF-probe .914 and plain row-role 3.88% / CF-probe .762 do not. Scores: `q2/res/scores.json`.

### 8.2 What B5's basis change costs at run time (arithmetic, for Q1)
**Read this before scoring tc* formats as fast (22:47).** q1fmt.qgemm_tc computes z = (x - mu) @ UR8 / UR4 with dense K x n8 / K x n4 matrices for every GEMM input. In the runtime that is one extra dense GEMM per GEMM input of size M x K x (n8 + n4) ~ M x K x K, in bf16 unless it is itself quantized. Measured in a CUDA graph at M = 1,125 (`q2/res/res_basis_cost.json`): a dense 2048 x 2048 projection takes 170 us in bf16 (cuBLAS) and 90 us in int8 (q2gemm); 6144 x 6144 takes 1,417 us / 712 us [M]. Per request at T = 1000 that is 24 x (3 x 170 + 1,417) us = 46 ms in bf16, 24 ms in int8 [A] for the Win, Wo, Wgu and Wd inputs: more than b8's whole 36 ms. 'MAC time vs W4A4 = 1.12' counts only the quantized GEMMs, not this projection. The B5 kernel (q2gemm s4_ts8: int8 slice + int4 slice + skipped slice) is cheap (0.613x b8 GEMM time at M = 1,125 for 256 int8 + 1,280 int4 columns [M]); the basis is what costs. Deployable variants: see (a)-(c) below.
The kernels only see the transformed input; the transform itself must be applied somewhere. In hobson every GEMM input is produced by a non-linear or shared step: Win/Wgu read RMSNorm of the residual stream (shared by all later layers, so a per-GEMM basis cannot be folded into the residual), Wo reads gate * attention-or-GDN output (elementwise gate), Wd reads SwiGLU (elementwise). So a dense K x K basis per GEMM costs an extra K x K multiply per row: 2048^2 per row for Win/Wgu/Wo (= a whole extra Wo GEMM, +25% of a layer's GEMM work if done in int8) and 6144^2 for Wd (+150% of Wd). Bases that are free or nearly free: (a) the existing Hadamards (R1, R2, R4) and any orthogonal transform shared by all readers of the residual (fold into R1); (b) block-diagonal bases with small blocks (64-256) applied in the glue kernel like the per-head Hadamard (cost ~ block/K of a GEMM; 128-blocks at K = 2048 are ~6% of a Wo GEMM in bf16); (c) permutations and per-channel scales (free). B5 allocations restricted to (a)-(c) keep the K-shrink gain.

## 9. Strassen-Winograd formats (moved to Q4; arithmetic kept for reference)

One level splits M, K and N in half; the sums of two blocks must fit the integer type, and the two summed blocks belong to different rows (t and t + M/2) and different output channels (n and n + N/2), so those pairs must share one scale. Integer Strassen is exact, so the result equals a dense GEMM on these codes:
- int8, one level ("W7A7p"): activation codes qmax 63, clip 1.0; rows t and t + Mh share s = fp32( fp32( max(amax_t, amax_{t+Mh}, 1e-8) / 63 ) * clip ), Mh = ceil(M/2) (rows >= M are zero padding). Weights: qmax 63; channels n and n + N/2 share one scale (max of the pair's absmax / 63, or a GPTQ scale chosen for the pair). Result = section 2 integer part.
- int4, one level ("W3A3p"): the same with qmax 3 (clip as chosen; 0.9 default).
- two levels: groups of four rows {t, t+M/4, t+M/2, t+3M/4} and four channels share a scale; int8 qmax 31, int4 qmax 1 (ternary).
- bf16, one level: operand sums rounded to bf16 (activation sums in the producer; weight sums offline, also bf16), seven fp32 tensor-core products of size (M/2, K/2, N/2), output combination in fp32, then bf16. Not bit-exact; emulate with the same bf16 roundings of the sums.


## 10. Addendum formats (BRIEF10 B5, B6, B8, B9)

### 10.1 B5: contiguous K slices at 8, 4 and 0 bits [built]
After the format's basis change (folded offline; the kernel sees only the result), the input columns are ordered [8-bit slice: K8 columns | 4-bit slice: K4 columns | 0-bit slice: the rest]. K8 must be a multiple of 64 and K4 a multiple of 128 (pad with zero columns otherwise).
- codes: x8 = A(x[:, :K8]; 127, 1.0, RTN) with scale s8_t; x4 = A(x[:, K8:K8+K4]; 7, clip, RTN) with scale s4_t; the 0-bit columns are not read.
- weights: int8 codes with per-channel scale sw8_n for the first K8 columns, int4 with sw4_n for the next K4.
- f = fp32( fp32( fp32(float(acc4) * s4_t) * sw4_n ) + fp32( fp32(float(acc8) * s8_t) * sw8_n ) ); then f = fp32( f + bias_n ) with bias_n = sum over the 0-bit columns of mu_k * W_nk (fp32, offline; mu is the calibration mean of those columns); y = bf16(f).
- ResQ (section 6) is the special case with no 0-bit slice. The kernel is q2gemm variant s4_ts8 with the int8 slice as the tail segment.

### 10.2 B6: the certificate statistic [built: prologue form]
- Prologue form (no subspace): stat_g = sum_t w_t * sum_k (y_tk - s_t q_tk)^2, accumulated with fp32 atomics into 64 partial slots (index t mod 64) and summed; order of the atomic adds is not deterministic, so emulate in fp32 and expect ~1e-6 relative differences. w_t is a per-row weight (default 1).
- Subspace form (with B1): stat_g = sum_t w_t * ||Z_t||^2 with Z = dX A from section 3, accumulated the same way in the Z kernel. [planned]

### 10.3 B8: arbitrary row partition [built]
Any subset of rows can be int8 (W8A8, section 8 arithmetic) with the rest int4: the quantize prologue writes int8 rows and int4 rows into two compact buffers (a position map POS[t] gives the slot), and the grouped GEMM scatters each output row back to its original row index. The arithmetic per row is unchanged, so the format is defined by the subset alone.

### 10.4 B9: unbiased sampled K tiles [built]
For the int4 (state-row) problem only. The K dimension is cut into blocks of 128 columns (nkb = K/128 blocks); output rows into blocks of 128 (counted within the state-row problem, mb = floor(t/128)); output columns into blocks of 128 (nb = floor(n/128)). For each (mb, nb) the kernel keeps exactly nkeep blocks:
- h_kb = fmix32( base ^ (kb * 0xC2B2AE35) ), base = seed_g ^ (mb * 0x9E3779B1) ^ (nb * 0x85EBCA77) (uint32 wrapping; fmix32 as in section 4);
- keep the nkeep blocks with the smallest (h_kb, kb) (ties broken by the smaller kb);
- acc = sum over kept blocks (int32); f = fp32( fp32( fp32( float(acc) * s_t ) * sw_n ) * kscale ), kscale = fp32(nkb / nkeep); y = bf16(f) (+ B1 tail if present).
- nkeep = round(f * nkb): f = 0.75 keeps 12 of 16 blocks at K = 2048 and 36 of 48 at K = 6144. `q2k.skip_mask` is the torch reference.

## Status of each kernel

| format | section | kernel | built | measured |
|---|---|---|---|---|
| F0 W8A8, F1 W4A4 (deployed) | 1 | H2 CUTLASS + Triton glue | yes (H2) | H2, J5, J15 |
| F2 W4A4, bf16-direct | 2 | q2gemm s4 | yes, bit-exact vs torch | yes |
| B1 tail r = 32-256 | 3 | q2gemm s4_tbf16 / s4_ts8 (+ Z kernel) | tail yes (fp32 order only); fused Z planned | tail yes |
| B2 SR | 4 | Triton quantize prologue (q2pro mode 1) | yes, bit-exact vs torch hash | prologue cost |
| B3 prototypes | 5 | Triton search (q2pro._proto_k) + q2gemm table epilogue | yes, bit-exact (idx, codes, scales) | search cost |
| ResQ split, B5 slices | 6, 10.1 | q2gemm s4_ts8 | yes, bit-exact | yes |
| group-scaled int4 | 7 | q2gemm s4g64 / s4g128 | yes, bit-exact | yes |
| row-role, B8 partition | 8, 8.1, 10.3 | q2gemm two-problem launch + rowmap; runtime q2rt QRT2 / QRT2C | yes, bit-exact (QRT2C: one extra fp16 rounding on question rows) | yes, GEMM and end to end |
| B9 sampled K tiles | 10.4 | q2gemm s4skip | yes, bit-exact | yes |
| B6 statistic | 10.2 | Triton prologue mode 6; q2gemm epilogue stat | yes | prologue cost |
| Strassen | 9 | moved to Q4 (Addendum 3) | - | - |
