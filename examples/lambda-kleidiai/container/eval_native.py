"""Accuracy evaluation inside the Lambda function by the scaling study's native method, with our torso.

This is jevscale.native.strands.StrandsEngine's procedure (the one behind the study's JEVal / BEV / Decision Index
numbers), step for step: every question is scored separately (no shared prefix); render_state + render_question; the
engine's _fit with strict_window=True (a prompt over the 4,096-token window is refused, not truncated); pointer
positions from _option_token_index; the fp32 head with the calibrated per-type temperature and the masked
log-softmax; unrounded log-probs in canonical key order. Only the torso forward differs: llama.cpp (Torso.hidden)
instead of transformers. Output rows are jevscale's results format, so jevscale.aggregate scores them unchanged:
  {"rid", "q": {qkey: {"keys", "n", "r": {"0": [logp ...]}}}, "unsupported": {qkey: reason}, "t_s": seconds}
"""
import gzip
import io
import json
import math
import time
from dataclasses import replace

import torch
from strands_decider.infer import _option_token_index
from strands_decider.modeling import apply_temperature, masked_log_softmax
from strands_decider.prompting import render_question, render_state
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


def evaluate_rows(eng, rows):
    eng.cfg = replace(eng.cfg, strict_window=True)
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
                s, qs, offs = eng._fit(render_state(req.state), [rq.text])
            except ValueError as e:
                res["unsupported"][k] = f"max_length_exceeded: {e}"
                continue
            ids = s + qs[0]
            try:
                opt = [len(s) + i for i in _option_token_index(offs[0], rq.option_spans, 0)]
            except ValueError as e:
                res["unsupported"][k] = f"invalid_question: {e}"
                continue
            h = eng.torso.hidden(ids, [len(ids) - 1] + opt)
            with torch.inference_mode():
                raw = eng.model.head(torch.from_numpy(h[:1]), torch.from_numpy(h[1:][None]))
                logits = apply_temperature(raw, eng._temperatures([rq.kind]))
                lp = masked_log_softmax(logits, torch.tensor([rq.n_slots]))[0, : rq.n_slots].tolist()
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


def run_shard(eng, src, dst, cpu):
    import boto3
    s3 = boto3.client("s3")
    b, k = src[5:].split("/", 1)
    rows = [json.loads(l) for l in gzip.open(io.BytesIO(s3.get_object(Bucket=b, Key=k)["Body"].read()), "rt")]
    t0 = time.perf_counter()
    res = evaluate_rows(eng, rows)
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
    return stats
