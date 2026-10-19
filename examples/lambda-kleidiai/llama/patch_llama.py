"""Patches for llama.cpp f74e2fa6a (qwen35) used by the Strands decider's prefill-only CPU engine.

  python patch_llama.py /path/to/llama.cpp [--patches prefill,gdn,kskip,sme]   (default: all four)

prefill  With embeddings on and no pooling, output only the rows the batch marks, and skip the qwen35 LM head. The
         decider reads hidden states at a handful of positions and never needs logits; upstream ran the last layer
         and the 248k-vocabulary LM head for every token in this mode (18-20% of a forward).
gdn      One pass over each Gated DeltaNet head's state per token instead of four (scalar gate, single snapshot).
kskip    GGML_KLEIDIAI_SKIP=a,b keeps weights whose name contains a or b off KleidiAI (an ablation switch).
sme      Drop the two Linux SME CPU variants from GGML_CPU_ALL_VARIANTS (AL2023's GCC 11 cannot build them).
"""
import argparse
from pathlib import Path


def patch_prefill(root):
    """Rows the batch marks only, and no LM head, in embeddings mode."""
    ctx = root / "src/llama-context.cpp"
    s = ctx.read_text()
    old = "    const bool output_all   = cparams.embeddings;\n"
    new = ("    // strands-decider prefill patch: with no pooling, honour the batch's output flags\n"
           "    const bool output_all   = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;\n")
    assert s.count(old) == 1, "llama-context.cpp anchor not found"
    ctx.write_text(s.replace(old, new))

    q = root / "src/models/qwen35.cpp"
    s = q.read_text()
    old = ("    // LM head\n    cur = build_lora_mm(model.output, cur, model.output_s);\n\n"
           "    cb(cur, \"result_output\", -1);\n    res->t_logits = cur;\n}\n")
    new = ("    if (cparams.embeddings) {  // strands-decider prefill patch: hidden states only, no LM head\n"
           "        ggml_build_forward_expand(gf, res->t_embd);\n        return;\n    }\n\n" + old)
    assert s.count(old) == 1, "qwen35.cpp LM-head anchor not found"
    q.write_text(s.replace(old, new, 1))


def patch_kskip(root):
    """GGML_KLEIDIAI_SKIP name filter in KleidiAI's supports_op."""
    k = root / "ggml/src/ggml-cpu/kleidiai/kleidiai.cpp"
    s = k.read_text()
    old = ("        // GET_ROWS needs the original weights in a native buffer.\n"
           "        if (op->op != GGML_OP_MUL_MAT) {\n            return false;\n        }\n")
    new = old + (
        "        // strands-decider prefill patch: GGML_KLEIDIAI_SKIP=a,b keeps weights whose name contains a or b on ggml's\n"
        "        // own path (block-32 activation quantisation) instead of KleidiAI's per-row int8 activations.\n"
        "        if (const char * skip = getenv(\"GGML_KLEIDIAI_SKIP\")) {\n"
        "            const std::string list(skip), name(op->src[0]->name);\n"
        "            for (size_t a = 0; a <= list.size();) {\n"
        "                size_t b = list.find(',', a);\n"
        "                if (b == std::string::npos) b = list.size();\n"
        "                if (b > a && name.find(list.substr(a, b - a)) != std::string::npos) return false;\n"
        "                a = b + 1;\n"
        "            }\n"
        "        }\n")
    assert s.count(old) == 1, "kleidiai.cpp supports_op anchor not found"
    k.write_text(s.replace(old, new))


def patch_sme(root):
    """Drop the Linux SME variants (armv9.2_1, armv9.2_2)."""
    c = root / "ggml/src/CMakeLists.txt"
    s = c.read_text()
    # AL2023's GCC 11 cannot target armv9.2-a+sme, and no Graviton has SME: drop the two Linux SME variants from
    # GGML_CPU_ALL_VARIANTS (Graviton2 -> armv8.2_1, Graviton3 -> armv8.6_1, Graviton4 -> armv8.6_2).
    for v in ("armv9.2_1", "armv9.2_2"):
        lines = [l for l in s.splitlines(True) if f"ggml_add_cpu_backend_variant({v} " in l]
        assert len(lines) == 1, v
        s = s.replace(lines[0], "")
    c.write_text(s)


def patch_gdn(root):
    """Single-pass Gated DeltaNet kernel for scalar-gate, single-snapshot heads."""
    o = root / "ggml/src/ggml-cpu/ops.cpp"
    s = o.read_text()
    old = """        for (int64_t t = 0; t < n_tokens; t++) {
                const float * q_d = (const float *)((const char *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);"""
    new = """        // strands-decider prefill patch: scalar-gate, single-snapshot heads in one pass over the state per token. Each row's
            // update S <- e^g S + k delta^T also produces the output S^T q and the next token's S^T k, instead of four
            // separate passes (scale, S^T k, rank-1 update, S^T q). Same arithmetic, fewer trips through the 64 KB state.
            if (!kda && K == 1 && S_v <= 512) {
                float dotk[512];
                {
                    const float * k0 = (const float *)((const char *)src_k->data + ik3 * nbk3 + ik1 * nbk1);
                    for (int64_t j = 0; j < S_v; ++j) {
                        float sum = 0.0f;
                        ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k0, 0, 1);
                        dotk[j] = sum;
                    }
                }
                for (int64_t t = 0; t < n_tokens; t++) {
                    const float * q_d = (const float *)((const char *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
                    const float * k_d = (const float *)((const char *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
                    const float * v_d = (const float *)((const char *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);
                    const float * kn  = t + 1 < n_tokens
                        ? (const float *)((const char *)src_k->data + ik3 * nbk3 + (t + 1) * nbk2 + ik1 * nbk1) : nullptr;
                    const float beta_val = *(const float *)((const char *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
                    const float eg = expf(*(const float *)((const char *)src_g->data + iv3 * nbg3 + t * nbg2 + iv1 * nbg1));
                    for (int64_t j = 0; j < S_v; ++j) {
                        float * row = &s_out[j * S_v];
                        const float dj = (v_d[j] - eg * dotk[j]) * beta_val;
                        float acc = 0.0f, accn = 0.0f;
                        int64_t i = 0;
    #if defined(__ARM_NEON)
                        const float32x4_t veg = vdupq_n_f32(eg), vdj = vdupq_n_f32(dj);
                        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, b0 = a0, b1 = a0;
                        for (; i + 8 <= S_v; i += 8) {
                            const float32x4_t r0 = vfmaq_f32(vmulq_f32(veg, vld1q_f32(row + i)), vdj, vld1q_f32(k_d + i));
                            const float32x4_t r1 = vfmaq_f32(vmulq_f32(veg, vld1q_f32(row + i + 4)), vdj, vld1q_f32(k_d + i + 4));
                            vst1q_f32(row + i, r0);
                            vst1q_f32(row + i + 4, r1);
                            a0 = vfmaq_f32(a0, r0, vld1q_f32(q_d + i));
                            a1 = vfmaq_f32(a1, r1, vld1q_f32(q_d + i + 4));
                            if (kn) {
                                b0 = vfmaq_f32(b0, r0, vld1q_f32(kn + i));
                                b1 = vfmaq_f32(b1, r1, vld1q_f32(kn + i + 4));
                            }
                        }
                        acc = vaddvq_f32(vaddq_f32(a0, a1));
                        accn = vaddvq_f32(vaddq_f32(b0, b1));
    #endif
                        for (; i < S_v; ++i) {
                            const float r = eg * row[i] + dj * k_d[i];
                            row[i] = r;
                            acc += r * q_d[i];
                            if (kn) {
                                accn += r * kn[i];
                            }
                        }
                        attn_data[j] = acc * scale;
                        dotk[j] = accn;
                    }
                    attn_data += S_v * H;
                }
                continue;
            }

            for (int64_t t = 0; t < n_tokens; t++) {
                const float * q_d = (const float *)((const char *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);"""
    # the triple-quoted texts above sit one indent deeper inside this function than in the source: undo that
    old, new = (t.replace("\n    ", "\n") for t in (old, new))
    assert s.count(old) == 1, "gated_delta_net anchor not found"
    o.write_text(s.replace(old, new))


PATCHES = {"prefill": patch_prefill, "gdn": patch_gdn, "kskip": patch_kskip, "sme": patch_sme}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--patches", default=",".join(PATCHES))
    a = ap.parse_args()
    for name in a.patches.split(","):
        if name:
            PATCHES[name](Path(a.root))
    print("patched", a.root, a.patches)
