// decider_hidden.c
//
// Flat C shim over libllama for the Lambda app. A ctypes-friendly API (no
// struct-by-value, no header layout to reproduce in Python): load a GGUF once,
// then decode exact token ids with embeddings=true and pooling NONE, and copy
// the per-token post-final-norm hidden states into a caller buffer. Those are
// the torso's last_hidden_state, which the pointer head reads.
//
// Built by container_build.sh against the pinned llama.cpp:
//   gcc -O3 -fPIC -shared -o libdecider_hidden.so decider_hidden.c \
//       -I<src>/include -I<src>/ggml/include -L<libs> -lllama

#include "llama.h"
#include <stdlib.h>
#include <string.h>

#ifndef DH_UBATCH
#define DH_UBATCH 512
#endif

// Opaque handle so Python holds a single void* and never sees a struct.
typedef struct {
    struct llama_model * model;
    int                  n_embd;
    int                  n_threads;
} dh_handle;

// Load a GGUF from a filesystem path (a /proc/self/fd/<n> path backed by a memfd
// works -- llama mmaps it). Returns NULL on failure. n_embd_out is set to the
// model's embedding-output dimension (2048 for this torso).
void * dh_load(const char * model_path, int n_threads, int * n_embd_out) {
    llama_backend_init();
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) return NULL;
    dh_handle * h = (dh_handle *) calloc(1, sizeof(dh_handle));
    h->model     = model;
    h->n_embd    = llama_model_n_embd_out(model);
    h->n_threads = n_threads > 0 ? n_threads : 1;
    if (n_embd_out) *n_embd_out = h->n_embd;
    return h;
}

int dh_n_embd(void * handle) {
    return handle ? ((dh_handle *) handle)->n_embd : -1;
}

// Run one forward pass over `n_tokens` ids and write [n_tokens, n_embd] float32
// row-major into `out` (caller-allocated). A fresh context is created and freed
// per call so each request is independent (no KV bleed between requests).
// The model weights stay resident in the handle across calls.
// Returns 0 on success, non-zero on failure.
int dh_forward(void * handle, const int * ids, int n_tokens, float * out) {
    if (!handle || n_tokens <= 0) return 1;
    dh_handle * h = (dh_handle *) handle;

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_tokens + 8;
    cparams.n_batch         = n_tokens + 8;
    cparams.n_ubatch        = (n_tokens + 8) < DH_UBATCH ? (n_tokens + 8) : DH_UBATCH;
    cparams.n_threads       = h->n_threads;
    cparams.n_threads_batch = h->n_threads;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf         = true;

    struct llama_context * ctx = llama_init_from_model(h->model, cparams);
    if (!ctx) return 2;

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = (llama_token) ids[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }

    int rc = llama_decode(ctx, batch);
    if (rc != 0) { llama_batch_free(batch); llama_free(ctx); return 3; }

    const int n_embd = h->n_embd;
    for (int i = 0; i < n_tokens; i++) {
        const float * e = llama_get_embeddings_ith(ctx, i);
        if (!e) { llama_batch_free(batch); llama_free(ctx); return 4; }
        memcpy(out + (size_t) i * n_embd, e, sizeof(float) * n_embd);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    return 0;
}

// Shared-prefix forward: decode the state prefix ONCE on seq 0 (no outputs),
// copy its memory (attention KV + linear-attention recurrent state) to seqs
// 1..n_q, then decode every question suffix on its own seq. Writes the hidden
// states of the SUFFIX tokens only, concatenated in question order, into `out`
// ([sum(q_lens), n_embd]). The same idea as SystemOneEngine's prefix cache.
// `sequential` != 0 decodes the suffixes one llama_decode each (fallback).
int dh_forward_shared(void * handle, const int * state_ids, int n_state,
                      const int * q_ids, const int * q_lens, int n_q,
                      int sequential, float * out) {
    if (!handle || n_state <= 0 || n_q <= 0) return 1;
    dh_handle * h = (dh_handle *) handle;
    int n_suffix = 0, max_q = 0;
    for (int s = 0; s < n_q; s++) { n_suffix += q_lens[s]; if (q_lens[s] > max_q) max_q = q_lens[s]; }
    const int n_big = n_state > n_suffix ? n_state : n_suffix;

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_state + n_suffix + 16;
    cparams.n_batch         = n_big + 8;
    // Cap the micro-batch: compute buffers scale ~quadratically with n_ubatch
    // (4096-token pages OOMed at 10 GB). llama splits n_batch into ubatches and
    // carries KV + recurrent state across them.
    cparams.n_ubatch        = (n_big + 8) < DH_UBATCH ? (n_big + 8) : DH_UBATCH;
    cparams.n_seq_max       = n_q + 1;
    cparams.kv_unified      = true;   // prefix cells are shared, not copied
    cparams.n_threads       = h->n_threads;
    cparams.n_threads_batch = h->n_threads;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf         = true;

    struct llama_context * ctx = llama_init_from_model(h->model, cparams);
    if (!ctx) return 2;

    // 1) state prefix on seq 0, no outputs needed.
    struct llama_batch b = llama_batch_init(n_big, 0, 1);
    b.n_tokens = n_state;
    for (int i = 0; i < n_state; i++) {
        b.token[i] = (llama_token) state_ids[i]; b.pos[i] = i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0;
    }
    if (llama_decode(ctx, b) != 0) { llama_batch_free(b); llama_free(ctx); return 3; }

    // 2) fork the prefix memory into one seq per question.
    llama_memory_t mem = llama_get_memory(ctx);
    for (int s = 0; s < n_q; s++) llama_memory_seq_cp(mem, 0, s + 1, -1, -1);

    // 3) suffixes (all in one batch, or one decode per question).
    const int n_embd = h->n_embd;
    int off = 0, row = 0;
    for (int s = 0; s < n_q; s++) {
        if (sequential || s == 0) b.n_tokens = 0;
        for (int j = 0; j < q_lens[s]; j++) {
            int k = b.n_tokens++;
            b.token[k] = (llama_token) q_ids[off + j]; b.pos[k] = n_state + j;
            b.n_seq_id[k] = 1; b.seq_id[k][0] = s + 1; b.logits[k] = 1;
        }
        off += q_lens[s];
        if (sequential || s == n_q - 1) {
            if (llama_decode(ctx, b) != 0) { llama_batch_free(b); llama_free(ctx); return 4; }
            for (int i = 0; i < b.n_tokens; i++, row++) {
                const float * e = llama_get_embeddings_ith(ctx, i);
                if (!e) { llama_batch_free(b); llama_free(ctx); return 5; }
                memcpy(out + (size_t) row * n_embd, e, sizeof(float) * n_embd);
            }
        }
    }
    llama_batch_free(b);
    llama_free(ctx);
    return 0;
}
