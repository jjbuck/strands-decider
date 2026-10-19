"""Accuracy evaluation inside the v2 zip functions by the scaling study's native method, torch-free.

The same procedure, row for row and key for key, as image/eval_native.py (the deployed function's eval mode), which is
jevscale.native.strands.StrandsEngine's: every question is scored separately (no shared prefix); render_state +
render_question; a strict 4,096-token window (a prompt over it is refused, not truncated, with SystemOneEngine._fit's
own message); pointer positions from the last token of each option's line; the head with the calibrated per-type
temperature and the log-softmax over the question's slots; unrounded log-probs (5 decimals) in canonical key order.
Output rows are jevscale's results format, so jevscale.aggregate scores them unchanged:
  {"rid", "q": {qkey: {"keys", "n", "r": {"0": [logp ...]}}}, "unsupported": {qkey: reason}, "t_s": seconds, "cpu"}
plus <out>.stats.json {"rows", "questions", "unsupported", "seconds", "cpu"}.

What differs from eval_native.py is only the code that runs it, none of the method:
  rendering, option positions  strands_decider.prompting from PR #77 (render_state / render_question unchanged from
                               6039cf8; option_token_index is infer._option_token_index moved there)
  tokenisation                 tokenizers.Tokenizer on the zip's tokenizer.json (PR #77's prepare_model.py export),
                               not transformers' AutoTokenizer on the checkpoint; the strict fit is written out here
                               because PR #77's HiddenStateEngine.fit words its refusal differently
  head                         PR #77's NumpyPointerHead in float64 (LayerNorm, q/k projections, scaled dot), on the
                               fp32 hidden states as eval_native's torch fp32 head; temperatures from head_config.json
  torso                        Shim.hidden (dec_eval), as lcpp_engine.Torso.hidden; v2q un-rotates inside the shim
"""
import gzip
import io
import json
import math
import time

import numpy as np

from strands_decider.prompting import option_token_index, render_question, render_state
from strands_decider.schema import SystemOneRequest

FLOOR = math.log(1e-9)  # jevscale PROB_FLOOR


def system_one_question(q):  # jevscale.native.base.system_one_question (noul: null descriptions dropped)
    out = {"type": q["type"], "instructions": q.get("instructions")}
    if q["type"] == "noul":
        c = {k: v for k, v in (q.get("criteria") or {}).items() if k in ("true", "false") and v not in (None, "")}
        if c:
            out["criteria"] = c
    else:
        out["criteria"] = q["criteria"]
    return out


def answer_keys(q):  # jevscale.prompt.answer_keys
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return [True, False]
    return list(range(len(q["criteria"])))


def to_canonical(q, by_native_key):  # jevscale.native.base.to_canonical
    names = list(q["criteria"]) if q["type"] == "choice" else (
        ["true", "false"] if q["type"] == "noul" else [str(i) for i in range(len(q["criteria"]))])
    if any(n not in by_native_key for n in names) or len(by_native_key) != len(names):
        raise ValueError("invalid_model_output: keys do not match the question")
    return [by_native_key[n] for n in names]


def fit_strict(eng, state_text, question_texts):
    """SystemOneEngine._fit with strict_window=True: question ids and offsets without special tokens, the state with
    them, and a refusal (the released message) when state + longest question passes the window."""
    max_len = eng.cfg.max_length
    encs = eng.tok.encode_batch(question_texts, add_special_tokens=False)
    q = [e.ids for e in encs]
    offs = [list(e.offsets) for e in encs]
    longest = max(len(x) for x in q)
    s = eng.tok.encode(state_text, add_special_tokens=True).ids
    if len(s) + longest > max_len:
        raise ValueError(f"prompt of {len(s) + longest} tokens exceeds the context window of {max_len} tokens")
    return s, q, offs


def log_probs(eng, h, kind, n_slots):
    """Head on [decide; options] hidden rows, / per-type temperature, log-softmax over the first n_slots
    (modeling.apply_temperature + masked_log_softmax)."""
    h = np.asarray(h, dtype=np.float64)
    logits = eng.head.logits(h[0], h[1:])
    t = float(eng.cfg.temperature_by_kind.get(kind, eng.cfg.temperature))
    x = logits[:n_slots] / max(t, 1e-6)
    m = np.max(x)
    return (x - m - np.log(np.sum(np.exp(x - m)))).tolist()


def evaluate_rows(eng, shim, rows):
    out = []
    for row in rows:
        res = {"rid": row["rid"], "q": {}, "unsupported": {}}
        t0 = time.perf_counter()
        for k, q in row["questions"].items():
            try:
                req = SystemOneRequest(state=row["state"], questions={k: system_one_question(q)})
            except Exception as e:  # pydantic ValidationError
                res["unsupported"][k] = f"invalid_question: {str(e)[:150]}"
                continue
            rq = render_question(req.questions[k])
            try:
                s, qs, offs = fit_strict(eng, render_state(req.state), [rq.text])
            except ValueError as e:
                res["unsupported"][k] = f"max_length_exceeded: {e}"
                continue
            ids = s + qs[0]
            try:
                opt = [len(s) + i for i in option_token_index(offs[0], rq.option_spans, 0)]
            except ValueError as e:
                res["unsupported"][k] = f"invalid_question: {e}"
                continue
            h = shim.hidden(ids, [len(ids) - 1] + opt)
            lp = log_probs(eng, h, rq.kind, rq.n_slots)
            try:
                vals = to_canonical(q, dict(zip(rq.slot_labels, lp)))
            except ValueError as e:
                res["unsupported"][k] = str(e)
                continue
            res["q"][k] = {"keys": answer_keys(q), "n": len(ids),
                           "r": {"0": [round(v if math.isfinite(v) else FLOOR, 5) for v in vals]}}
        res["t_s"] = round(time.perf_counter() - t0, 4)
        out.append(res)
    return out


def run_shard(eng, shim, src, dst, cpu):
    import boto3
    s3 = boto3.client("s3")
    b, k = src[5:].split("/", 1)
    rows = [json.loads(l) for l in gzip.open(io.BytesIO(s3.get_object(Bucket=b, Key=k)["Body"].read()), "rt")]
    t0 = time.perf_counter()
    res = evaluate_rows(eng, shim, rows)
    wall = time.perf_counter() - t0
    for r in res:
        r["cpu"] = cpu  # the sandbox's CPU, for latency by host
    body = "".join(json.dumps(r) + "\n" for r in res).encode()
    b2, k2 = dst[5:].split("/", 1)
    s3.put_object(Bucket=b2, Key=k2, Body=gzip.compress(body))
    nq = sum(len(r["q"]) + len(r["unsupported"]) for r in res)
    stats = {"rows": len(rows), "questions": nq, "unsupported": sum(len(r["unsupported"]) for r in res),
             "seconds": round(wall, 2), "cpu": cpu}
    s3.put_object(Bucket=b2, Key=k2 + ".stats.json", Body=json.dumps(stats).encode())
    print("native_eval", json.dumps({"in": src, "out": dst, **stats}), flush=True)
    return stats
