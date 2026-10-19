"""Our llama.cpp torso as the backend of PR #77's strands_decider.hidden_engine.HiddenStateEngine (shipped as engine.py).

v2b: v2/engine.py (version 3 of strands-decider-2b-v2) with one change, the rows the torso is asked for. PR #77's
ForwardShared contract returns every question token's hidden state, and HiddenStateEngine then reads only hidden[-1]
(the `<answer>` token) and the option rows; v2 met the contract by marking every question token as a llama.cpp output,
and with the prefill patch each marked row costs the last layer and the output norm. Here HeadRowsEngine (a subclass;
its evaluate is PR #77's, line for line, except the forward call) passes the head's rows to Shim.forward_shared, which
marks only those: each question's option ends, then its last token, as the deployed engine's pointer gather
(lcpp_engine._slot_probs_batched / _slot_probs_shared_prefix) marks them. _probs gets the compact rows with the option
positions renumbered 0..K-1 and the decide row last, so the head, the temperature and the softmax run on the same
values as before. Called without rows, forward_shared keeps PR #77's contract (every question token).

v2 was a copy of h2h/snap_engine.py. Drop-in for PR #77's examples/lambda/app/engine.py: same load_engine(lib_path, gguf_path,
assets_dir, threads, **overrides) -> (engine, shim), so its handler.py and loader.py run unchanged. The backend is ours:
llama.cpp f74e2fa6a with trade/lambda/engine/patch_llama.py, the portable variant set with KleidiAI, and decshim.c
(lib/libdecshim.so beside the handler's lib/libdecider_hidden.so path, which is not shipped). The shim is built from
rotabl/decshim.c, which adds one hook to the deployed engine/decshim.c: DEC_UBATCH=n sets the micro-batch (unset, it is
the deployed min(n_ctx, 4224)).

Two weight sets, one code path:
  v2   H512 Q8_0 (per-matmul runtime Hadamard, prism.hadamard), as deployed.
  v2q  A6 Q8_0 (rotabl/fold_quarot.py: a QuaRot-style rotation Q folded into the residual stream, the final norm set to
       identity, runtime Hadamard on the down_proj, o_proj and GDN out_proj inputs). The torso then returns x_hat Q, and
       DEC_UNROTATE / DEC_UNROTATE_S3 (q_and_scale.npz: q = Q, scale = g, the final RMSNorm's effective scale 1 + w)
       maps it back to the released model's post-norm state, h = (x_hat Q) Q^T g = FWHT(x_hat Q) * d * g in float64
       with Q = diag(d) H, before the head. This is rotabl/lcpp_engine.py's numpy butterfly, which was bit-identical
       there to the BLAS product. Applied to every row the shim returns (dec_eval and dec_eval_fork).

forward_shared follows lcpp_engine.LlamaCppEngine, so the hidden states are the deployed engine's:
  one question       one forward over state + question (dec_eval), as SystemOneEngine._slot_probs_batched;
  several questions  the shared-state fork (dec_eval_fork): the state once, then up to N_SEQ - 1 questions per fork.
HiddenStateEngine reads hidden[-1] and the option rows of each question; HeadRowsEngine asks for exactly those (with
the prefill patch, outputs decide which rows the last layer finishes). A fork group is also cut where the
state plus its questions would overflow the context (lcpp_engine would refuse such a group; it changes no answer).
DEC_KV=f32 and flash attention off are decshim's environment settings, as deployed. Shim.hidden(ids, positions) is
lcpp_engine.Torso.hidden, for the eval mode (eval_np.py).
"""

from __future__ import annotations

import ctypes
import io
import logging
import os
from typing import Any

import numpy as np

from strands_decider.hidden_engine import HiddenStateEngine
from strands_decider.prompting import option_token_index, render_question, render_state, to_answer
from strands_decider.schema import SystemOneRequest, SystemOneResponse, Usage

log = logging.getLogger("decider.engine")
_I32 = ctypes.POINTER(ctypes.c_int32)
_F32 = ctypes.POINTER(ctypes.c_float)
SHIM: "Shim | None" = None  # the loaded shim, for the eval mode


def _cpu_part() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("CPU part"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def _fwht(x: np.ndarray) -> np.ndarray:
    """Normalized Sylvester-Walsh-Hadamard transform along the last axis (row vectors times H). rotabl/lcpp_engine.py."""
    n = x.shape[-1]
    y, h = x.reshape(-1, n), 1
    while h < n:
        y = y.reshape(y.shape[0], n // (2 * h), 2, h)
        y = np.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), axis=2)
        h *= 2
    return y.reshape(x.shape) / np.sqrt(n)


def _load_unrotation(dim: int) -> np.ndarray | None:
    """d * g from q_and_scale.npz (a local path in DEC_UNROTATE or an s3:// URI in DEC_UNROTATE_S3), or None."""
    path, uri = os.environ.get("DEC_UNROTATE"), os.environ.get("DEC_UNROTATE_S3")
    if not path and not uri:
        return None
    if uri:
        import boto3
        b, k = uri[5:].split("/", 1)
        z = np.load(io.BytesIO(boto3.client("s3").get_object(Bucket=b, Key=k)["Body"].read()))
    else:
        z = np.load(path)
    q, g = z["q"].astype(np.float64), z["scale"].astype(np.float64)
    assert q.shape == (dim, dim) and g.shape == (dim,), (q.shape, g.shape, dim)
    d = np.sign(q[:, 0])
    if not np.allclose(_fwht(np.diag(d)), q, atol=1e-12):  # the butterfly is exact only for Q = diag(d) H
        raise RuntimeError("q_and_scale.npz: Q is not diag(d) H; the FWHT un-rotation does not apply")
    log.info("un-rotation from %s: Q = diag(d) H verified, scale range %.4f..%.4f", uri or path, g.min(), g.max())
    return d * g


class Shim:
    """ctypes binding for decshim.c: dec_load_seq, dec_eval, dec_eval_fork."""

    def __init__(self, lib_path: str, model_path: str, threads: int, n_ctx: int, n_seq: int):
        os.environ.setdefault("DEC_GGML_DIR", os.path.dirname(lib_path))  # the libggml-cpu-*.so variants
        lib = ctypes.CDLL(lib_path)
        lib.dec_load_seq.restype = ctypes.c_void_p
        lib.dec_load_seq.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int]
        lib.dec_eval.restype = ctypes.c_int
        lib.dec_eval.argtypes = [ctypes.c_void_p, _I32, ctypes.c_int, _I32, ctypes.c_int, _F32]
        lib.dec_eval_fork.restype = ctypes.c_int
        lib.dec_eval_fork.argtypes = [ctypes.c_void_p, _I32, ctypes.c_int, _I32, _I32, ctypes.c_int, _I32, _I32, _F32]
        lib.dec_n_embd.argtypes = [ctypes.c_void_p]
        self.h = lib.dec_load_seq(model_path.encode(), threads, n_ctx, 1, 0, n_seq)
        if not self.h:
            raise RuntimeError(f"dec_load_seq failed for {model_path}")
        self.lib, self.n_ctx, self.n_seq = lib, n_ctx, n_seq
        self.n_embd = lib.dec_n_embd(self.h)
        self.unrot = _load_unrotation(self.n_embd)
        variant = sorted({ln.split()[-1].rsplit("/", 1)[-1] for ln in open("/proc/self/maps") if "libggml-cpu" in ln})
        self.init_info = {"cpu_part": _cpu_part(), "ggml_cpu": variant, "threads": threads,
                          "ubatch": os.environ.get("DEC_UBATCH", "default"), "kv": os.environ.get("DEC_KV", "f16"),
                          "unrotate": self.unrot is not None, "n_ctx": n_ctx, "n_seq": n_seq}
        log.info("init cpu_part=%s ggml_cpu=%s threads=%d ubatch=%s kv=%s unrotate=%s n_ctx=%d n_seq=%d",
                 self.init_info["cpu_part"], ",".join(variant), threads, self.init_info["ubatch"],
                 self.init_info["kv"], self.init_info["unrotate"], n_ctx, n_seq)

    def _unrotate(self, out: np.ndarray) -> np.ndarray:
        if self.unrot is None:
            return out
        return (_fwht(out.astype(np.float64)) * self.unrot).astype(np.float32)

    def _eval(self, ids: list[int], pos: list[int]) -> np.ndarray:
        t = np.ascontiguousarray(ids, dtype=np.int32)
        p = np.ascontiguousarray(pos, dtype=np.int32)
        out = np.empty((len(p), self.n_embd), dtype=np.float32)
        rc = self.lib.dec_eval(self.h, t.ctypes.data_as(_I32), len(t), p.ctypes.data_as(_I32), len(p),
                               out.ctypes.data_as(_F32))
        if rc != 0:
            raise RuntimeError(f"dec_eval rc={rc} on {len(t)} tokens")
        return self._unrotate(out)

    hidden = _eval  # lcpp_engine.Torso.hidden(ids, positions)

    def _fork(self, state: list[int], qs: list[list[int]], rows: list[list[int]]) -> np.ndarray:
        """State once, then each question from a copy of it: hidden states at each question's (question-relative)
        rows, concatenated in order."""
        st = np.ascontiguousarray(state, dtype=np.int32)
        qt = np.ascontiguousarray([t for q in qs for t in q], dtype=np.int32)
        ql = np.ascontiguousarray([len(q) for q in qs], dtype=np.int32)
        op = np.ascontiguousarray([p for r in rows for p in r], dtype=np.int32)
        no = np.ascontiguousarray([len(r) for r in rows], dtype=np.int32)
        out = np.empty((len(op), self.n_embd), dtype=np.float32)
        rc = self.lib.dec_eval_fork(self.h, st.ctypes.data_as(_I32), len(st), qt.ctypes.data_as(_I32),
                                    ql.ctypes.data_as(_I32), len(ql), op.ctypes.data_as(_I32),
                                    no.ctypes.data_as(_I32), out.ctypes.data_as(_F32))
        if rc != 0:
            raise RuntimeError(f"dec_eval_fork rc={rc}: state {len(st)} tokens, {len(ql)} questions")
        return self._unrotate(out)

    def forward_shared(self, state_ids: list[int], q_ids: list[list[int]],
                       rows: list[list[int]] | None = None) -> list[np.ndarray]:
        """Each question's hidden states continuing the state: at its question-relative `rows` ([len(rows[i]),
        n_embd]), or at every token without rows ([len(q), n_embd], PR #77's ForwardShared)."""
        if rows is None:
            rows = [list(range(len(q))) for q in q_ids]
        n_s = len(state_ids)
        if len(q_ids) == 1:
            return [self._eval(list(state_ids) + list(q_ids[0]), [n_s + p for p in rows[0]])]
        out: list[np.ndarray] = []
        g, i = [], 0
        while i <= len(q_ids):
            fits = i < len(q_ids) and len(g) < self.n_seq - 1 and n_s + sum(len(q_ids[j]) for j in g) + len(
                q_ids[i]) <= self.n_ctx
            if fits:
                g.append(i)
                i += 1
                continue
            if not g:  # one question over the context on its own: the engine's window fitting prevents this
                raise RuntimeError(f"question {i} does not fit the context with the state")
            h = self._fork(state_ids, [q_ids[j] for j in g], [rows[j] for j in g])
            out.extend(np.split(h, np.cumsum([len(rows[j]) for j in g])[:-1]))
            g = []
            if i == len(q_ids):
                break
        return out


class HeadRowsEngine(HiddenStateEngine):
    """HiddenStateEngine whose forward computes only the rows the pointer head reads.

    evaluate is PR #77's HiddenStateEngine.evaluate except for the forward: the torso is asked for each question's
    option rows and then its last row (`<answer>`), and _probs reads them as rows 0..K-1 and -1 of that compact array,
    the same values it read from the full one at opos and -1."""

    def evaluate(self, request: SystemOneRequest | Any) -> SystemOneResponse:
        req = (
            request
            if isinstance(request, SystemOneRequest)
            else SystemOneRequest.model_validate(request)
        )
        if req.images:
            raise ValueError("this engine serves the text model; images are not supported")
        names = list(req.questions)
        rendered = [render_question(req.questions[n]) for n in names]
        s, q, offs = self.fit(render_state(req.state), [rq.text for rq in rendered])
        n_input = len(s) + sum(len(x) for x in q)
        if self.cfg.max_request_tokens and n_input > self.cfg.max_request_tokens:
            raise ValueError(
                f"request is {n_input} tokens (state {len(s)} + {len(q)} questions "
                f"{n_input - len(s)}); this deployment takes at most "
                f"{self.cfg.max_request_tokens} per request. Split the questions across "
                f"requests: each pays the state's {len(s)} tokens again."
            )
        # Before the forward: a question trimmed through its options is refused at once.
        opos = [option_token_index(o, rq.option_spans, 0) for rq, o in zip(rendered, offs, strict=True)]
        hidden = self.forward_shared(s, q, [list(p) + [len(x) - 1] for p, x in zip(opos, q, strict=True)])
        answers = {}
        for name, rq, h, p in zip(names, rendered, hidden, opos, strict=True):
            answers[name] = to_answer(
                rq, self._probs(rq, np.asarray(h, dtype=np.float64), list(range(len(p)))),
                ordinal_smoothing=self.cfg.ordinal_smoothing,
            )
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=n_input, output_tokens=len(names)),
        )


def load_engine(lib_path: str, gguf_path: str, assets_dir: str, threads: int,
                **overrides: Any) -> tuple[HeadRowsEngine, Shim]:
    """The engine over a loaded GGUF. `overrides` set HiddenEngineConfig fields. lib_path names the handler's shim
    path; ours is libdecshim.so in the same directory."""
    global SHIM
    n_seq = int(os.environ.get("N_SEQ", "17"))
    shim = Shim(os.path.join(os.path.dirname(lib_path), "libdecshim.so"), gguf_path, threads,
                n_ctx=4224 if n_seq == 1 else 16384, n_seq=n_seq)
    SHIM = shim
    return HeadRowsEngine.from_assets(assets_dir, shim.forward_shared, **overrides), shim
