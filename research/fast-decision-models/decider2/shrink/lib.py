"""Shared helpers: load hobson-v19 (LoRA merged) on MPS, run a custom layer loop with
skip/truncate/record, build JevBench + training prompts the way the serving engine does."""
import os, sys, glob, json, time, math, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F
from strands_decider.modeling import StrandsDeciderModel, PointerHead, gather_options, masked_log_softmax, MASK_VALUE
from strands_decider.infer import SystemOneEngine, EngineConfig
from strands_decider.prompting import render_question, render_state, read_noul
from strands_decider.schema import NoulQuestion, ChoiceQuestion, ScoreQuestion
from strands_decider.data.format import Example
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

CK = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*"))[0]
JB = os.path.expanduser("~/work/shrink/jb_public")
SD = os.path.expanduser("~/work/sd")


def load(device="cuda", merge=True, max_length=3072):
    model = StrandsDeciderModel.load(CK)
    if merge:
        model.torso = model.torso.merge_and_unload()
    model.config.max_length = max_length
    eng = SystemOneEngine(model, EngineConfig(device=device, use_prefix_cache=False))
    return eng


def jevbench_tasks():
    rows = []
    for f in ["easy", "original", "hard"]:
        for l in open(f"{JB}/{f}.jsonl"):
            rows.append(json.loads(l))
    return rows


def jb_question(t):
    q = t["question"]
    if q["type"] == "noul":
        return NoulQuestion(instructions=q["instructions"], criteria=q.get("criteria"))
    if q["type"] == "choice":
        return ChoiceQuestion(instructions=q["instructions"], criteria=q["criteria"])
    return ScoreQuestion(instructions=q["instructions"], criteria=q["criteria"])


def jb_label(t, rq):
    """Index in rendered slot order of the gold answer (rendered order == criteria order)."""
    typ = t["question"]["type"]
    if typ == "noul":
        want = "true" if t["expected"] == "yes" else "false"
        return list(rq.slot_labels).index(want)
    if typ == "choice":
        return list(rq.slot_labels).index(t["expected"])
    return int(t["expected"])


def encode(eng, state, question, order=None):
    """-> dict(ids [1,T], opt_idx [K], rq, n_tokens, ans_pos)"""
    rq = render_question(question, option_order=order)
    s, q = eng._fit(render_state(state), [rq.text])
    ids = torch.tensor([s + q[0]], device=eng.device)
    opt = eng._option_idx([rq], len(s))[0]
    return dict(ids=ids, opt=opt, rq=rq, T=ids.size(1))


class Runner:
    """Custom layer loop over the merged Qwen3.5 text torso."""

    def __init__(self, eng):
        self.eng = eng
        self.model = eng.model
        self.torso = eng.model.torso
        self.cfg = self.torso.config
        self.dev = eng.device
        self.n = self.cfg.num_hidden_layers
        self.types = self.cfg.layer_types

    @torch.inference_mode()
    def run(self, ids, skip=(), upto=None, record_pos=None, want_bi=False, rec_layers=None):
        """Returns dict: final (normed hidden at last kept layer), recs (list per layer index of [P,d]
        residual at record_pos AFTER that layer; None if skipped), bi (list of mean 1-cos per layer)."""
        tm = self.torso
        upto = self.n if upto is None else upto
        emb = tm.embed_tokens(ids)
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).view(1, 1, -1).expand(4, B, -1)
        text_pos = pos[0]
        pos3 = pos[1:]
        am = torch.ones(B, T, dtype=torch.long, device=ids.device)
        mk = dict(config=self.cfg, inputs_embeds=emb, attention_mask=am, past_key_values=None, position_ids=text_pos)
        masks = {"full_attention": create_causal_mask(**mk),
                 "linear_attention": create_recurrent_attention_mask(**mk)}
        h = emb
        pe = tm.rotary_emb(h, pos3)
        recs, bi = [], []
        for i in range(upto):
            if i in skip:
                recs.append(None if record_pos is None else (recs[-1] if recs else None))
                bi.append(0.0)
                continue
            hn = tm.layers[i](h, position_embeddings=pe, attention_mask=masks[self.types[i]],
                              position_ids=text_pos, past_key_values=None, use_cache=False)
            if isinstance(hn, tuple):
                hn = hn[0]
            if want_bi:
                bi.append(float((1 - F.cosine_similarity(h.float(), hn.float(), dim=-1)).mean()))
            h = hn
            if record_pos is not None and (rec_layers is None or (i + 1) in rec_layers):
                recs.append(h[0, record_pos].to(torch.bfloat16).cpu())
        return dict(h=h, final=tm.norm(h), recs=recs, bi=bi)

    def head_logits(self, hid_last, opt, ans_pos=-1):
        pooled = hid_last[0, ans_pos].float().unsqueeze(0)
        options = hid_last[0, opt].float().unsqueeze(0)
        return self.model.head(pooled, options)[0]  # [K]


def temp_of(model, kind):
    bk = model.config.temperature_by_kind or {}
    return float(bk.get(kind, model.config.temperature))


def ece(conf, correct, nb=10):
    conf = torch.as_tensor(conf, dtype=torch.float32)
    correct = torch.as_tensor(correct, dtype=torch.float32)
    e = 0.0
    for b in range(nb):
        m = (conf >= b / nb) & (conf < (b + 1) / nb if b < nb - 1 else conf <= 1.0)
        if m.any():
            e += float(m.float().mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return e
