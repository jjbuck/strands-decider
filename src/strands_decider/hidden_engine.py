"""System One evaluation over any backend that returns the torso's final hidden states.

`SystemOneEngine` (infer.py) runs the torso and the pointer head in torch. This engine
takes the torso's output from a callable instead and runs the pointer head in numpy, so
it imports neither torch nor transformers. The AWS Lambda recipe (examples/lambda) uses
it with llama.cpp computing the hidden states on CPU.

Everything around the torso follows `SystemOneEngine`, and tests/test_hidden_engine.py
holds the two to the same answers: the prompts and readouts come from prompting.py, the
state is fitted to the window the same way (trimmed by default, refused with
`strict_window`), and the pointer head is the same function of the same weights.

Two differences, both in what is counted rather than what is answered. The backend
decodes the state once for any number of questions, so `usage.input_tokens` counts it
once; `SystemOneEngine` counts it once per batch of `max_batch` questions. And the state
is fitted once for all the questions, where `SystemOneEngine` fits it per batch: with more
than `max_batch` questions and a state that has to be trimmed, this engine trims it for
the longest question of all.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from .prompting import (
    RenderedQuestion,
    option_token_index,
    render_question,
    render_state,
    to_answer,
)
from .schema import Answer, SystemOneRequest, SystemOneResponse, Usage

# (state token ids, question token ids) -> each question's hidden states, [len(q), hidden].
# The questions continue the state independently: none sees another's tokens.
ForwardShared = Callable[[list[int], list[list[int]]], list[np.ndarray]]


class NumpyPointerHead:
    """modeling.PointerHead in numpy, from its state dict.

    LayerNorm, then a query from the decide position (the last token, `<answer>`) and a
    key from the last token of each option's line; the logit is their scaled dot product.
    """

    def __init__(self, weights: Mapping[str, np.ndarray], pointer_dim: int, eps: float = 1e-5):
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.scale = pointer_dim**-0.5
        self.eps = eps

    def _norm(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + self.eps) * self.w["norm.weight"] + self.w["norm.bias"]

    def logits(self, decide: np.ndarray, options: np.ndarray) -> np.ndarray:
        """decide [hidden], options [K, hidden] -> logits [K]."""
        q = self._norm(decide) @ self.w["q.weight"].T + self.w["q.bias"]
        k = self._norm(options) @ self.w["k.weight"].T + self.w["k.bias"]
        out: np.ndarray = (k @ q) * self.scale
        return out


@dataclass
class HiddenEngineConfig:
    max_length: int
    temperature: float = 1.0
    temperature_by_kind: dict[str, float] = field(default_factory=dict)
    ordinal_smoothing: float = 0.0
    model_name: str = "strands-decider"
    # As EngineConfig: refuse a prompt that does not fit the window instead of trimming the
    # state, and the largest share of the window the question may claim.
    strict_window: bool = False
    max_question_fraction: float = 0.75
    # Most tokens one request may forward, the state once plus every question. None: no limit.
    max_request_tokens: int | None = None


class HiddenStateEngine:
    def __init__(
        self,
        tokenizer: Tokenizer,
        head: NumpyPointerHead,
        config: HiddenEngineConfig,
        forward_shared: ForwardShared,
    ):
        self.tok = tokenizer
        # Count every token and trim only where fit() decides to: the tokenizer's own
        # truncation, if its file set one, would cut the question and its options. And no
        # padding: a tokenizer.json saved with it pads every question in a batch to the
        # longest, which the prompt never had.
        self.tok.no_truncation()
        self.tok.no_padding()
        self.head = head
        self.cfg = config
        self.forward_shared = forward_shared

    @classmethod
    def from_assets(cls, assets_dir: str, forward_shared: ForwardShared, **overrides: Any) -> HiddenStateEngine:
        """Load what examples/lambda/prepare_model.py writes: tokenizer.json, head.safetensors
        and head_config.json. `overrides` set HiddenEngineConfig fields."""
        from safetensors.numpy import load_file

        with open(os.path.join(assets_dir, "head_config.json"), encoding="utf-8") as f:
            hc = json.load(f)
        config = HiddenEngineConfig(
            max_length=int(hc["max_length"]),
            temperature=float(hc["temperature"]),
            temperature_by_kind=hc.get("temperature_by_kind") or {},
            ordinal_smoothing=float(hc.get("ordinal_smoothing", 0.0)),
            model_name=os.path.basename(str(hc.get("checkpoint", "strands-decider")).rstrip("/")),
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        head = NumpyPointerHead(
            load_file(os.path.join(assets_dir, "head.safetensors")), int(hc["pointer_dim"])
        )
        tok = Tokenizer.from_file(os.path.join(assets_dir, "tokenizer.json"))
        return cls(tok, head, config, forward_shared)

    # ------------------------------------------------------------------ window
    def fit(
        self, state_text: str, question_texts: list[str]
    ) -> tuple[list[int], list[list[int]], list[list[tuple[int, int]]]]:
        """SystemOneEngine._fit: the question gets first claim on the window.

        Returns the state's ids, each question's ids and each question's character offsets.
        Special tokens go on the state only (Gemma's BOS; nothing for Qwen). A question
        longer than its reserve loses tokens from the front, keeping its options and the
        `<answer>` marker; the state then gets the rest of the window, trimmed from the end.
        """
        max_len = self.cfg.max_length
        encs = self.tok.encode_batch(question_texts, add_special_tokens=False)
        q = [e.ids for e in encs]
        offs = [list(e.offsets) for e in encs]
        longest = max(len(x) for x in q)
        if self.cfg.strict_window:
            s = self.tok.encode(state_text, add_special_tokens=True).ids
            if len(s) + longest > max_len:
                raise ValueError(
                    f"prompt is {len(s) + longest} tokens (state + question); the model's window "
                    f"is {max_len}. Shorten the state by at least {len(s) + longest - max_len} "
                    "tokens."
                )
            return s, q, offs
        reserve = min(longest, max(1, int(max_len * self.cfg.max_question_fraction)))
        cut = [max(0, len(x) - reserve) for x in q]
        q = [x[c:] for x, c in zip(q, cut, strict=True)]
        offs = [o[c:] for o, c in zip(offs, cut, strict=True)]
        self.tok.enable_truncation(max_length=max(1, max_len - reserve))
        try:
            s = self.tok.encode(state_text, add_special_tokens=True).ids
        finally:
            self.tok.no_truncation()
        return s, q, offs

    # ---------------------------------------------------------------- evaluate
    def _probs(self, rq: RenderedQuestion, hidden: np.ndarray, opos: list[int]) -> list[float]:
        logits = self.head.logits(hidden[-1], hidden[opos])
        t = float(self.cfg.temperature_by_kind.get(rq.kind, self.cfg.temperature))
        logits = logits / max(t, 1e-6)
        e = np.exp(logits - np.max(logits))
        return list((e / e.sum()).tolist())

    def evaluate(self, request: SystemOneRequest | Mapping[str, Any]) -> SystemOneResponse:
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
        hidden = self.forward_shared(s, q)
        answers: dict[str, Answer] = {}
        for name, rq, h, p in zip(names, rendered, hidden, opos, strict=True):
            answers[name] = to_answer(
                rq, self._probs(rq, np.asarray(h, dtype=np.float64), p),
                ordinal_smoothing=self.cfg.ordinal_smoothing,
            )
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=n_input, output_tokens=len(names)),
        )
