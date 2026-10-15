"""HiddenStateEngine must answer as SystemOneEngine does, given the same hidden states.

The hidden-state engine runs everything but the torso without torch (the AWS Lambda recipe
feeds it llama.cpp's hidden states). Here the torso is a tiny random-weight Qwen3.5 run in
torch, so any difference comes from the engine itself: the window fitting, the offsets, the
numpy pointer head or the readouts. CPU only; nothing is downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from tokenizers import Tokenizer  # noqa: E402

from strands_decider.hidden_engine import (  # noqa: E402
    HiddenEngineConfig,
    HiddenStateEngine,
    NumpyPointerHead,
)
from strands_decider.infer import EngineConfig, SystemOneEngine, load_engine  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    PointerHead,
    StrandsDeciderConfig,
    StrandsDeciderModel,
    ensure_bos,
)
from strands_decider.prompting import render_question, render_state  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

POLICY = ("Loans run 21 days and renew twice online unless another patron holds the item. "
          "Accounts with fines above ten dollars cannot renew until the fine is paid. ")
QUESTIONS = {
    "team": ChoiceQuestion(instructions="Which team should handle this?",
                           criteria={"circulation": "renewals and holds", "billing": "fines",
                                     "reference": "research help"}),
    "urgent": NoulQuestion(instructions="Does the patron convey urgency?"),
    "mood": ScoreQuestion(instructions="How frustrated is the patron?",
                          criteria=["calm", "annoyed", "angry"]),
}
# Over 384 tokens, the most of a 512-token window a question may claim: trimmed from the
# front, it loses its first options, and both engines refuse it.
LONG_CHOICE = ChoiceQuestion(instructions="Which shelf holds the book?",
                             criteria={f"shelf {i}": "fiction and poetry, rows A to F" for i in range(60)})
REQUESTS = {
    "one question": SystemOneRequest(state=POLICY + "Can I renew the cookbook?",
                                     questions={"team": QUESTIONS["team"]}),
    "three questions": SystemOneRequest(state=POLICY * 6 + "My renewal failed and I leave Friday!",
                                        questions=QUESTIONS),
    # About 1,100 tokens against a 512-token window: both engines trim the state.
    "trimmed state": SystemOneRequest(state=POLICY * 30 + "Help!", questions=QUESTIONS),
}


def _tokenizer(bos: bool):
    """A word-level tokenizer covering every rendered prompt, optionally adding a BOS."""
    from tokenizers import models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    texts = [render_state(r.state) for r in REQUESTS.values()]
    texts += [render_question(q).text for q in [*QUESTIONS.values(), LONG_CHOICE]]
    words = {w for text in texts for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(text)}
    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", "<bos>", *sorted(words)])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                 unk_token="<unk>", bos_token="<bos>" if bos else None)
    return ensure_bos(hf) if bos else hf


def _checkpoint(tmp_path, bos: bool):
    tok = _tokenizer(bos)
    torch.manual_seed(0)
    base_cfg = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"])
    base = tmp_path / "base"
    transformers.Qwen3_5ForCausalLM(base_cfg).save_pretrained(base)
    cfg = StrandsDeciderConfig(base_model=str(base), head_type="pointer", pointer_dim=16,
                               max_length=512, torch_dtype="float32", lora_r=4, lora_alpha=8,
                               temperature=0.8, temperature_by_kind={"choice": 0.6, "score": 1.5},
                               ordinal_smoothing=0.1, force_bos=bos)
    torso = transformers.Qwen3_5ForCausalLM.from_pretrained(base).model
    model = StrandsDeciderModel(cfg, torso, tok)
    model.attach_lora()
    with torch.no_grad():  # lora_B starts at zero: make the adapter change the torso
        for name, p in model.torso.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.1)
    path = tmp_path / "ckpt"
    model.save_pretrained(str(path))
    return path


def _hidden_engine(torch_engine: SystemOneEngine, **overrides) -> HiddenStateEngine:
    """A HiddenStateEngine on the torch engine's own torso, head and tokenizer."""
    model = torch_engine.model
    cfg = model.config

    @torch.inference_mode()
    def forward_shared(state_ids, question_ids):
        out = []
        for q in question_ids:  # each question continues the state on its own
            ids = torch.tensor([state_ids + q])
            hidden = model.encode(ids, torch.ones_like(ids))[0, len(state_ids):]
            out.append(hidden.float().numpy())
        return out

    head = NumpyPointerHead({k: v.numpy() for k, v in model.head.state_dict().items()},
                            cfg.pointer_dim)
    config = HiddenEngineConfig(
        max_length=cfg.max_length, temperature=cfg.temperature,
        temperature_by_kind=cfg.temperature_by_kind, ordinal_smoothing=cfg.ordinal_smoothing,
        model_name=torch_engine.cfg.model_name, **overrides)
    tok = Tokenizer.from_str(torch_engine.tok.backend_tokenizer.to_str())
    # Saved tokenizers can carry padding and truncation (Qwen3.5's pads); the engine must
    # count every token of every question as SystemOneEngine does, padded or not.
    tok.enable_padding(pad_id=0, pad_token="<pad>")
    tok.enable_truncation(max_length=8)
    return HiddenStateEngine(tok, head, config, forward_shared)


def _probabilities(response):
    return {name: [a.noul] if a.type == "noul" else list(a.probabilities.values())
            for name, a in response.answers.items()}


@pytest.fixture(scope="module", params=[False, True], ids=["no-bos", "bos"])
def engines(request, tmp_path_factory):
    path = _checkpoint(tmp_path_factory.mktemp("bos" if request.param else "nobos"), request.param)
    torch_engine = load_engine(str(path), device="cpu")
    return torch_engine, _hidden_engine(torch_engine)


def test_numpy_head_matches_pointer_head():
    torch.manual_seed(1)
    head = PointerHead(hidden_size=32, dim=8).eval()
    with torch.no_grad():
        head.norm.weight.normal_()
        head.norm.bias.normal_()
    decide, options = torch.randn(32) * 3, torch.randn(5, 32) * 3
    with torch.no_grad():
        want = head(decide[None], options[None])[0].numpy()
    got = NumpyPointerHead({k: v.numpy() for k, v in head.state_dict().items()}, 8)
    np.testing.assert_allclose(got.logits(decide.numpy(), options.numpy()), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", list(REQUESTS))
def test_hidden_engine_answers_as_torch_does(engines, name):
    torch_engine, hidden_engine = engines
    expected = torch_engine.evaluate(REQUESTS[name])
    got = hidden_engine.evaluate(REQUESTS[name].model_dump())
    assert got.model == expected.model
    assert got.usage == expected.usage
    want, have = _probabilities(expected), _probabilities(got)
    assert have.keys() == want.keys()
    for q in want:
        assert have[q] == pytest.approx(want[q], abs=1.5e-4), q  # both round to 4 places
    for q, answer in expected.answers.items():
        if answer.type == "choice":
            assert got.answers[q].choice == answer.choice


def test_fit_matches_system_one_engine(engines):
    torch_engine, hidden_engine = engines
    state = render_state(REQUESTS["trimmed state"].state)
    texts = [render_question(q).text for q in QUESTIONS.values()]
    s, q, offs = hidden_engine.fit(state, texts)
    want_s, want_q = torch_engine._fit(state, texts)
    assert (s, q) == (want_s, want_q)
    assert [[tuple(o) for o in x] for x in torch_engine._last_offsets] == offs
    assert len(s) + max(len(x) for x in q) <= torch_engine.model.config.max_length
    if torch_engine.tok.bos_token_id is not None:
        assert s[0] == torch_engine.tok.bos_token_id


def test_strict_window_refuses_as_torch_does(engines):
    torch_engine, _ = engines
    strict_torch = SystemOneEngine(torch_engine.model, EngineConfig(device="cpu", strict_window=True))
    strict_hidden = _hidden_engine(torch_engine, strict_window=True)
    with pytest.raises(ValueError, match="window"):
        strict_torch.evaluate(REQUESTS["trimmed state"])
    with pytest.raises(ValueError, match="Shorten the state by at least"):
        strict_hidden.evaluate(REQUESTS["trimmed state"])
    assert strict_hidden.evaluate(REQUESTS["one question"]).answers.keys() == {"team"}


def test_question_trimmed_through_its_options_is_refused_before_the_forward(engines):
    torch_engine, _ = engines
    request = SystemOneRequest(state="Where is it?", questions={"shelf": LONG_CHOICE})
    with pytest.raises(ValueError, match="truncated through its option list"):
        torch_engine.evaluate(request)
    hidden = _hidden_engine(torch_engine)
    calls = []
    hidden.forward_shared = lambda s, q: calls.append(1) or []
    with pytest.raises(ValueError, match="truncated through its option list"):
        hidden.evaluate(request)
    assert not calls


def test_request_budget_and_images_are_refused(engines):
    torch_engine, _ = engines
    budget = _hidden_engine(torch_engine, max_request_tokens=100)
    with pytest.raises(ValueError, match="at most 100 per request"):
        budget.evaluate(REQUESTS["three questions"])
    with_image = REQUESTS["one question"].model_copy(update={"images": ["aGk="]})
    with pytest.raises(ValueError, match="images"):
        budget.evaluate(with_image)
