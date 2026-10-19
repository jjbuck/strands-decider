"""The Strands decider with its torso on llama.cpp (CPU), everything else as in the torch engine.

The decider never generates: one forward over the prompt, then a pointer head reads the final hidden states at the
`<answer>` token and at the last token of each option's line. `decshim.c` exposes exactly that from llama.cpp
(embeddings on, no pooling, outputs only at the requested positions). As in `mlx_engine.MLXEngine`, only the torso
moves: rendering, tokenisation, truncation, option positions, temperatures, the masked softmax and the fp32 head are
`SystemOneEngine`'s own code. The LoRA adapter is folded into the weights before conversion (box/merge_export.py).
"""
from __future__ import annotations

import ctypes
import threading
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import (StrandsDeciderConfig, apply_temperature, build_head, checkpoint_dir,
                                      config_path, load_head_state, masked_log_softmax)

_I32 = ctypes.POINTER(ctypes.c_int32)
_F32 = ctypes.POINTER(ctypes.c_float)


class Torso:
    """A GGUF torso behind decshim: hidden(ids, positions) -> [len(positions), d] float32."""

    def __init__(self, lib: str, gguf: str, threads: int, n_ctx: int = 4224, mmap: bool = True,
                 verbose: bool = False, n_seq: int = 1) -> None:
        self.lib = ctypes.CDLL(lib)
        self.lib.dec_load.restype = ctypes.c_void_p
        self.lib.dec_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.dec_eval.restype = ctypes.c_int
        self.lib.dec_eval.argtypes = [ctypes.c_void_p, _I32, ctypes.c_int, _I32, ctypes.c_int, _F32]
        self.lib.dec_n_embd.argtypes = [ctypes.c_void_p]
        self.lib.dec_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.dec_load_seq.restype = ctypes.c_void_p
        self.lib.dec_load_seq.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int]
        self.lib.dec_eval_fork.restype = ctypes.c_int
        self.lib.dec_eval_fork.argtypes = [ctypes.c_void_p, _I32, ctypes.c_int, _I32, _I32, ctypes.c_int, _I32, _I32,
                                           _F32]
        self.n_seq = n_seq
        self.h = self.lib.dec_load_seq(gguf.encode(), threads, n_ctx, int(mmap), int(verbose), n_seq)
        if not self.h:
            raise RuntimeError(f"llama.cpp could not load {gguf}")
        self.dim = self.lib.dec_n_embd(self.h)
        self.n_ctx = n_ctx

    def set_threads(self, n: int) -> None:
        self.lib.dec_set_threads(self.h, n)

    def hidden(self, ids: list[int], positions: list[int]) -> np.ndarray:
        t = np.ascontiguousarray(ids, dtype=np.int32)
        p = np.ascontiguousarray(positions, dtype=np.int32)
        out = np.empty((len(p), self.dim), dtype=np.float32)
        rc = self.lib.dec_eval(self.h, t.ctypes.data_as(_I32), len(t), p.ctypes.data_as(_I32), len(p),
                               out.ctypes.data_as(_F32))
        if rc != 0:
            raise RuntimeError(f"dec_eval failed ({rc}) on {len(t)} tokens")
        return out


    def fork(self, state: list[int], questions: list[list[int]], positions: list[list[int]]) -> np.ndarray:
        """State once, then every question from a copy of it: hidden states at each question's (question-relative)
        positions, concatenated in order."""
        st = np.ascontiguousarray(state, dtype=np.int32)
        qt = np.ascontiguousarray([t for q in questions for t in q], dtype=np.int32)
        ql = np.ascontiguousarray([len(q) for q in questions], dtype=np.int32)
        op = np.ascontiguousarray([p for ps in positions for p in ps], dtype=np.int32)
        no = np.ascontiguousarray([len(ps) for ps in positions], dtype=np.int32)
        out = np.empty((len(op), self.dim), dtype=np.float32)
        rc = self.lib.dec_eval_fork(self.h, st.ctypes.data_as(_I32), len(st), qt.ctypes.data_as(_I32),
                                    ql.ctypes.data_as(_I32), len(ql), op.ctypes.data_as(_I32), no.ctypes.data_as(_I32),
                                    out.ctypes.data_as(_F32))
        if rc != 0:
            raise RuntimeError(f"dec_eval_fork failed ({rc}): state {len(st)} tokens, {len(ql)} questions")
        return out


class _Readout:
    def __init__(self, config: StrandsDeciderConfig, tokenizer: Any, head: torch.nn.Module) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.head = head


class LlamaCppEngine(SystemOneEngine):
    """`SystemOneEngine` with the torso forward on llama.cpp. Build one with `load_llamacpp_engine`."""

    def __init__(self, torso: Torso, readout: _Readout, config: EngineConfig) -> None:
        # Not super().__init__(): there is no torch torso to move. These are the attributes the inherited helpers read.
        self.cfg = config
        self.model = readout
        self.tok = readout.tokenizer
        self.device = "cpu"
        self.torso = torso
        self._lock = threading.Lock()  # one llama.cpp context, one forward at a time
        self._prefix_cache_unforkable = False
        self._tok_lock = threading.Lock()
        self._quadratic_attn = False

    def evaluate(self, request: Any) -> Any:
        with self._lock:
            return super().evaluate(request)

    def _slot_probs_batched(self, s: list[int], q: list[list[int]], n_slots: list[int], kinds: list[str],
                            offsets: list[list[tuple[int, int]]], rendered: Any = None) -> tuple[torch.Tensor, int]:
        rows = [s + qi for qi in q]
        pointer = self.model.config.head_type == "pointer"
        opt = self._option_idx(rendered, len(s), offsets).tolist() if pointer else [[] for _ in rows]
        width = max(len(o) for o in opt) if pointer else 0
        pooled = np.empty((len(rows), self.torso.dim), dtype=np.float32)
        options = np.zeros((len(rows), max(width, 1), self.torso.dim), dtype=np.float32)
        for i, row in enumerate(rows):
            real = [p for p in opt[i] if p >= 0]
            h = self.torso.hidden(row, [len(row) - 1] + real)
            pooled[i] = h[0]
            options[i, : len(real)] = h[1:]  # padded slots stay 0; masked_log_softmax drops them
        with torch.inference_mode():
            p = torch.from_numpy(pooled)
            raw = self.model.head(p, torch.from_numpy(options)) if pointer else self.model.head(p)
            logits = apply_temperature(raw, self._temperatures(kinds))
            probs = masked_log_softmax(logits, torch.tensor(n_slots)).exp()
        return probs, sum(len(r) for r in rows)

    def _slot_probs_shared_prefix(self, s: list[int], q: list[list[int]], n_slots: list[int], kinds: list[str],
                                  offsets: list[list[tuple[int, int]]], rendered: Any = None) -> tuple[torch.Tensor, int]:
        """The state is encoded once and forked across the questions (dec_eval_fork), as the torch engine's shared-prefix
        path does with its KV cache. Option positions are question-relative here (base 0)."""
        pointer = self.model.config.head_type == "pointer"
        opt = self._option_idx(rendered, 0, offsets).tolist() if pointer else [[] for _ in q]
        width = max(len(o) for o in opt) if pointer else 0
        pooled = np.empty((len(q), self.torso.dim), dtype=np.float32)
        options = np.zeros((len(q), max(width, 1), self.torso.dim), dtype=np.float32)
        group = max(1, self.torso.n_seq - 1)
        for g0 in range(0, len(q), group):
            qs = q[g0:g0 + group]
            pos = [[len(qi) - 1] + [p for p in opt[g0 + i] if p >= 0] for i, qi in enumerate(qs)]
            h = self.torso.fork(s, qs, pos)
            r = 0
            for i, ps in enumerate(pos):
                pooled[g0 + i] = h[r]
                options[g0 + i, : len(ps) - 1] = h[r + 1: r + len(ps)]
                r += len(ps)
        with torch.inference_mode():
            p = torch.from_numpy(pooled)
            raw = self.model.head(p, torch.from_numpy(options)) if pointer else self.model.head(p)
            logits = apply_temperature(raw, self._temperatures(kinds))
            probs = masked_log_softmax(logits, torch.tensor(n_slots)).exp()
        return probs, len(s) + sum(len(qi) for qi in q)


def load_llamacpp_engine(checkpoint: str, gguf: str, lib: str, threads: int, n_ctx: int | None = None,
                         mmap: bool = True, n_seq: int = 1) -> LlamaCppEngine:
    """n_seq > 1 turns on the shared-state fork for requests with several questions (n_seq - 1 per fork); the
    context then holds 16,384 tokens so a 4k state plus its questions fit in one fork."""
    path = checkpoint_dir(checkpoint)
    dc = StrandsDeciderConfig.from_json(config_path(path))
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_ctx = n_ctx or (4224 if n_seq == 1 else 16384)
    torso = Torso(lib, gguf, threads, n_ctx, mmap, n_seq=n_seq)
    head = build_head(dc, torso.dim)
    head.load_state_dict(load_head_state(path))
    head.to(torch.float32).eval()
    return LlamaCppEngine(torso, _Readout(dc, tok, head), EngineConfig(device="cpu", use_prefix_cache=n_seq > 1))
