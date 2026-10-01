"""data/SHA256SUMS: the exact bytes that recipe.sh verify accepts.

Teacher and replay targets attach to corpus rows by position, so recipe.sh checks its
inputs against this file before every stage that uses them.
"""

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Entries for files that are not committed, and the recipe.sh stage that writes each one.
PRODUCED = {
    "data/train_v5.jsonl": "build",
    "data/train_v5.holdout.jsonl": "build",
    "data/holdout_v5_norule.jsonl": "build",
    "data/multistep_v14.jsonl": "multistep",
    "data/multistep_v14_eval.jsonl": "multistep",
    "data/raw/contract-nli.zip": "fetch",
    "data/raw/musique_data_v1.0.zip": "fetch",
    "data/raw/helpsteer2/train.jsonl.gz": "fetch",
    "data/raw/helpsteer2/validation.jsonl.gz": "fetch",
}


def _manifest():
    """{path: sha256}, from lines in sha256sum's text format."""
    lines = (ROOT / "data" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        assert re.fullmatch(r"[0-9a-f]{64}  \S+", line), line
    return {line[66:]: line[:64] for line in lines}


def test_synthetic_entries_match_the_committed_files():
    want = {p: h for p, h in _manifest().items() if p.startswith("data/synthetic/")}
    got = {
        f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in ROOT.glob("data/synthetic/*.jsonl")
    }
    assert got == want


def test_every_entry_is_committed_or_written_by_a_recipe_stage():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "data"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
    )
    paths = set(_manifest())
    assert paths - tracked == set(PRODUCED)
    recipe = (ROOT / "training" / "recipe.sh").read_text(encoding="utf-8")
    for stage in set(PRODUCED.values()):
        assert re.search(rf"^{stage}\(\) \{{", recipe, re.M), stage
