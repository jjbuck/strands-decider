"""The teacher must see exactly the prompt its benchmark score was measured on."""
from __future__ import annotations

import pytest

from strands_decider.data.format import Example
from strands_decider.data.teacher import LETTERS, render, to_row

NOUL = Example(kind="noul", state="WIN a FREE cruise", instructions="Is this spam?",
               options=[["false", "genuine"], ["true", "promotional"]], label=1, task="spam")
CHOICE = Example(kind="choice", state="Arsenal won 3-1.", instructions="Which section?",
                 options=[["World", "international"], ["Sports", ""]], label=1, task="ag_news")
SCORE = Example(kind="score", state="Great food.", instructions="How positive?",
                options=[["0", "bad"], ["1", "ok"], ["2", "good"]], label=2, task="yelp")


def test_rows_match_the_jevbench_semif_adapter_mapping():
    row, order = to_row(NOUL)
    # true first, as the adapter lists it; order maps teacher position -> canonical index
    assert [o["id"] for o in row["options"]] == ["true", "false"]
    assert [o["description"] for o in row["options"]] == ["true: promotional", "false: genuine"]
    assert order == [1, 0]
    row, order = to_row(CHOICE)
    # an empty description falls back to the option name, as the adapter does
    assert [o["description"] for o in row["options"]] == ["World: international", "Sports: Sports"]
    assert order == [0, 1]
    row, _ = to_row(SCORE)
    assert [o["description"] for o in row["options"]] == ["0: bad", "1: ok", "2: good"]


def test_too_many_options_get_no_teacher():
    many = Example(kind="choice", state="s", instructions="q",
                   options=[[f"o{i}", ""] for i in range(len(LETTERS) + 1)], label=0)
    assert to_row(many) is None


@pytest.mark.parametrize("ex", [NOUL, CHOICE, SCORE])
def test_prompt_is_byte_identical_to_semif(ex):
    semif = pytest.importorskip("semif_phase1.direct")
    transformers = pytest.importorskip("transformers")
    try:  # same chat template family as the teacher; skip if not cached offline
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    row, _ = to_row(ex)
    ids, _, _ = semif.encode_prompt(tok, row, 4096)
    assert tok.encode(render(tok, row), add_special_tokens=False) == ids

