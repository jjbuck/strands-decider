"""registry.py: the configurations plotted in the figures, keyed by the IDs in docs/IDEAS_EXPLORED.md.

Each entry:
  id      the table ID (a letter suffix for a variant)
  name    short description for tooltips and the CSVs
  run     metrics.json run id (catalog.py naming), or None
  manual  scores transcribed from a DRAFT_REPORT when no prediction file exists ({suite: {metric: value}})
  lat     latency_grid curve key (A10G, measured), or None
  size    '2B' (hobson's size class) or 'small' (a smaller model: a control or a from-scratch test)
  note    caveat shown in the tooltip

Multi-question entries (curve q > 1) are plotted at their measured multi-question speedup and drawn hollow.
"""

E = [
    dict(id='hobson', name='hobson-v19, bf16 fused runtime', run='ref/hobson', lat='j15/bf16', size='2B'),
    # precision
    dict(id='P1', name='W8A8 + rotation + GPTQ, 8 GEMMs bf16', run='h2/preds_precmap_w8a8_b8_gptq_w8', lat='j15/b8', size='2B',
         note='original GPTQ draw; re-drawn calibrations flip 0.65-1.11%'),
    dict(id='P2', name='W4A4 + rotation + GPTQ', run='h2/preds_w4a4_codes_gptq_w4', lat='j15/w4a4', size='2B'),
    dict(id='P2a', name='W4A4 with 48 GEMMs at W8A8', run='h2/preds_precmap_w4a4_k48_gptq_w4_w8', lat='j15/k48', size='2B'),
    dict(id='P2b', name='W4A4, 64 GEMMs W8A8, question rows bf16 (H6)', run='h6/plainqb_k64_ba16', lat='h6/k64qb', size='2B'),
    dict(id='P3', name='W4A16 weight-only, GPTQ g128', run='j5/preds_w4g128', lat='j5/w4a16', size='2B',
         note='latency measured at 32-400 state tokens only'),
    dict(id='P9', name='natively ternary BitNet 2B, W1.58A8', run=None, lat='j10/ternary_a8', size='2B',
         manual={'JB-all': dict(acc=.619, brier=.464), 'JB-hard': dict(acc=.385), 'REAL-label': dict(acc=.733, brier=.392),
                 'REAL-agree': dict(agree_sd=.803), 'CF': dict(flip=.254), 'CF-probe': dict(flip=.203)}),
    # reducing work per token
    dict(id='R1', name='W8A8 + always exit at layer 16', run='j15/exit_L16_eval', lat='j15/exit16', size='2B',
         note='every decision exits at layer 16; the confidence cascade changes 0 of 6,216 real decisions at 0.688x W8A8'),
    dict(id='R1a', name='W8A8 + always exit at layer 8', run='j15/exit_L8_eval', lat='j15/exit8', size='2B'),
    dict(id='R2a', name='depth split DT-A8: state rows 8 of 24 layers', run='j3/a812_L8', lat='j3/DT-A8', size='2B'),
    dict(id='R2b', name='depth split DT-set12: 12 layers, options as a set', run='j3/set812_L12', lat='j3/DT-set12', size='2B'),
    dict(id='R2c', name='depth split at 16 layers, untrained', run='j3/tf_full_16A', lat='j3/DT-A16', size='2B'),
    dict(id='R6', name='every state row thin (rank 64) from layer 13', run=None, lat='j13/cut13', size='2B',
         manual={'JB-hard': dict(acc=.531), 'REAL-label': dict(acc=.792), 'REAL-agree': dict(agree_sd=.991),
                 'LONG': dict(agree_sd=.933), 'CF': dict(flip=.288, flip_given_hobson=1.0), 'CF-probe': dict(flip=.322, flip_given_hobson=.952)}),
    dict(id='R10a', name="hobson's first 12 layers, distilled (control)", run=None, lat='f7/hob12', size='small',
         manual={'JB-hard': dict(acc=.546), 'REAL-label': dict(acc=.735), 'REAL-agree': dict(agree_sd=.884), 'LONG': dict(agree_sd=.885),
                 'CF': dict(flip_given_hobson=.780), 'CF-probe': dict(flip_given_hobson=.571)}),
    dict(id='R10b', name='Qwen3.5-0.8B decider (control)', run=None, lat='f7/q08', size='small',
         manual={'JB-hard': dict(acc=.469), 'REAL-label': dict(acc=.767), 'REAL-agree': dict(agree_sd=.824), 'LONG': dict(agree_sd=.867),
                 'CF': dict(flip_given_hobson=.661), 'CF-probe': dict(flip_given_hobson=.543)}),
    # architecture
    dict(id='A1', name='T5Gemma 2B encoder, masked state cache (e1b)', run='j1/preds_A', lat='j1/encoder', size='2B'),
    dict(id='A1a', name='T5Gemma 2B encoder, state sees question (e1a)', run='j1/preds_B', lat='j1/encoder', size='2B',
         note='1-question latency; needs one state pass per question'),
    dict(id='A2', name='bidirectional GDN, masked state cache', run='j2/qag_all', lat='j2/bi_all', size='2B'),
    dict(id='A2a', name='bidirectional GDN, state sees question', run='j2/qa_all', lat='j2/bi_all', size='2B',
         note='1-question latency; needs one state pass per question'),
    dict(id='A3', name='question-first compiled schema (H7)', run='h7/sets_s900', lat=None, size='2B',
         manual_lat=dict(speedup=237.4 / 56.6, q=15, note='15 questions at 1000 tokens, bf16 (H2)')),
    dict(id='A4', name='questions compiled into weights, late (J6)', run='j6/preds_late:student', lat='j6/q4_late', size='2B',
         note='deployed questions only; JevBench questions stay in context, so JB equals hobson'),
    dict(id='A5', name='compiled question rows, 92% compiled (J14 A)', run='j14/A150:oa+sfx:replay:frozen', lat='j14/oa_q15', size='2B'),
    dict(id='A5a', name='compiled question rows, 69% compiled (J14 C)', run='j14/C400:ob@8+ok4@8+sfx:replay:frozen', lat='j14/ob_q15', size='2B'),
    dict(id='A5b', name='compiled question headers only, 12% compiled (J14 F)', run='j14/F150:ob+q32:replay:frozen', lat='j14/F_q15', size='2B',
         note='the only J14 arm that meets every bar'),
    dict(id='A6', name='precompiled deployment documents + compile adapter (J9 s400)', run='j9/t4_R_affine', lat='j9/c55', size='2B',
         note='latency at 55% compiled share; measured 1.90x on real banking requests'),
    dict(id='A6a', name='precompiled deployment documents, untrained', run='j9/u_R_affine', lat='j9/c55', size='2B'),
    dict(id='A7', name='64k super-token vocabulary, 1,100 updates (J7 TL)', run='j7/TL64k', lat='j7/super64k', size='2B'),
    dict(id='A7a', name='16k super-token vocabulary, 400 updates (J7 T16k)', run='j7/T16k', lat='j7/super16k', size='2B'),
    dict(id='A8', name='value-identity and structure channels (J7 V)', run='j7/V', lat='j7/hobson', size='2B',
         manual_lat=dict(speedup=57.06 / 57.35, q=1)),
    dict(id='A9', name='exact-relation module (J12 xr)', run='j12/preds_xr', lat='j12/xr', size='2B'),
    dict(id='A10', name='decision pretraining + fine-tune, 12k rows (J4 b)', run='j4/b_r12000', lat=None, size='2B',
         manual_lat=dict(speedup=1.0, q=1, note='same shapes as hobson')),
    dict(id='A11', name='slot model, 114M from scratch (J11 M_slot)', run='j11/runs/M_slot/preds_kit', lat='j11/M_slot', size='small',
         manual_lat=dict(speedup=57.1 / 5.53, q=1, note='against full hobson; 2.41x against its own 114M decoder')),
    dict(id='A11a', name='decoder, 114M from scratch (J11 M_dec)', run='j11/runs/M_dec/preds_kit', lat='j11/M_dec', size='small',
         manual_lat=dict(speedup=57.1 / 13.3, q=1, note='against full hobson')),
    # 4-bit program (BRIEF10) and decision-native layouts (BRIEF11), 2026-10-06/07
    dict(id='P10', name='row role: state rows W4A4, question rows W8A8 (Q2 deployed kernels)', run='q2/preds_q2_w4q8c', lat='q2/rowrole', size='2B'),
    dict(id='P11', name='k64rr: 64 most sensitive GEMMs W8A8, other 32 row role (Q2 deployed kernels)', run='q2/preds_q2_ck64rr', lat='q2/k64rr', size='2B'),
    dict(id='B4', name='W4A4 + decision-level scales + 20M-token QAT (Q3)', run='q3/qrt_w4a4_qad_t20M', lat='j15/w4a4', size='2B'),
    dict(id='B11a', name='W8A8 + decision-aware int4 2:4 on state rows, layers 14-23 (Q5)', run='q5/res/b8_s24_L14-23_sgptd_int4p_s:b8+s24.L14-23.sgptd.int4p.s', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 31.7, q=1, est=True, note='end-to-end latency estimated from Q4 GEMM time 0.83x of W8A8')),
    dict(id='B11b', name='W8A8 + decision-aware int8 2:4 on state rows, layers 12-22 (Q5)', run='q5/res/b8_s24_L12-22_sgptd_int8n05_s:b8+s24.L12-22.sgptd.int8n0.5.s', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 32.35, q=1, note='measured end to end by Q4 with B13 (0.89x W8A8)')),
    dict(id='B12', name='W8A8 + 60% of layer 12-23 MLP neurons removed by decision saliency (Q5)', run='q5/res/b8_nr_G12-23_dsalc_f06:b8+nr.G12-23.dsalc.f0.6', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 30.7, q=1, est=True, note='end-to-end latency estimated (with B13) from GEMM time 0.79x of W8A8')),
    dict(id='B13', name='W8A8 + exact eliminations (layer 23 K/V only, layer-0 table) (Q4)', run='q4/b8gptq_b13', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 34.80, q=1, note='measured end to end')),
    dict(id='M1', name='all-attention hobson, 2.3M tokens of training (M1 d3)', run='m1/d3small', lat='m1/allattn', size='2B'),
    dict(id='M2', name='segment-isolated state, depth 12, global question readers, 4.1 h training (M2 N k12)', run='m2/res/full/N2_k12', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 34.8, q=1, note='A10G bf16 at 1000 tokens with 55% of the state compiled; 1.70x on 84 real requests')),
    dict(id='M2a', name='constants isolated, depth 12 (M2 C k12)', run='m2/res/full/C_k12', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 33.1, q=1, note='A10G bf16 at 1000 tokens, 55% compiled; 1.82x on 84 real requests, 2.01x on banking')),
    dict(id='M2b', name='constants isolated, depth 8 (M2 C k8)', run='m2/res/full/C_k8', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 28.7, q=1, note='A10G bf16 at 1000 tokens, 55% compiled; 2.13x on 84 real requests')),
    dict(id='M2c', name='segment-isolated state, depth 8 (M2 N k8, 2.6 h)', run='m2/res/full/N_k8', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 29.8, q=1, note='A10G bf16 at 1000 tokens, 55% compiled; 2.02x on 84 real requests')),
    dict(id='C1', name='k64rr + layer-16 exit cascade (Q2, box q2b, deployed kernels)', run='q2/preds_q2b_k64x16', lat=None, size='2B',
         manual_lat=dict(speedup=57.1 / 23.77, q=1, est=True, note='1000-token latency combines measured segment times with the measured 94% exit share; 0.635x W8A8 measured on 120 real requests')),
    dict(id='HW3', name='hobson on Inferentia2, pipelined NKI GDN kernel (N1)', run=None, lat='n1/inf2', size='2B', plot_acc=False),
    # hardware (exact hobson function)
    dict(id='HW1', name='Inferentia2, one NeuronCore', run=None, lat='j8/inf2', size='2B', plot_acc=False),
    dict(id='HW1a', name='Sapphire Rapids AMX, 16 cores', run='j8/cpreds_bf16all', lat='j8/spr16', size='2B'),
    # data and evaluation
    dict(id='E1', name='hobson + detail augmentation, Qwen tokens (J7 C)', run='j7/C', lat=None, size='2B',
         manual_lat=dict(speedup=1.0, q=1, note='same shapes as hobson')),
    dict(id='E1a', name='hobson + detail augmentation, LoRA (J12 ctl)', run='j12/preds_ctl', lat=None, size='2B',
         manual_lat=dict(speedup=1.0, q=1, note='same shapes as hobson')),
    dict(id='E2', name='this-that-model-1.0', run='h4/this-that-model-1.0_multi', lat=None, size='2B',
         manual_lat=dict(speedup=1.0, q=1, note='same shapes as hobson')),
]
BY_ID = {e['id']: e for e in E}
