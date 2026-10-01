"""`strands-decider data build`: one failed recipe stops the build, and no corpus is written."""

from typer.testing import CliRunner

import strands_decider.data.recipes as recipes
from strands_decider.cli import app
from strands_decider.data.format import Example, read_jsonl


def _build(tmp_path, *names):
    out = tmp_path / "train.jsonl"
    args = ["data", "build", "--out", str(out)]
    for name in names:
        args += ["-r", name]
    return CliRunner().invoke(app, args), out


def _rows(name, **_):
    return [Example("noul", f"{name} state", "Is it?", [["false", ""], ["true", ""]], 1, task=name)]


def test_a_failed_recipe_stops_the_build(monkeypatch, tmp_path):
    def build_recipe(name, **kw):
        if name == "b":
            raise ConnectionError("offline")
        return _rows(name)

    monkeypatch.setattr(recipes, "build_recipe", build_recipe)
    res, out = _build(tmp_path, "a", "b", "c")
    assert res.exit_code != 0
    assert "recipe b failed" in res.output
    assert not out.exists()


def test_a_successful_build_writes_the_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(recipes, "build_recipe", _rows)
    res, out = _build(tmp_path, "a", "b")
    assert res.exit_code == 0, res.output
    assert [ex.task for ex in read_jsonl(str(out))] == ["a", "b"]


def test_a_missing_datasets_package_names_the_train_extra(monkeypatch, tmp_path):
    def build_recipe(name, **kw):
        raise ModuleNotFoundError("No module named 'datasets'", name="datasets")

    monkeypatch.setattr(recipes, "build_recipe", build_recipe)
    res, out = _build(tmp_path, "a")
    assert res.exit_code != 0
    assert '".[train]"' in res.output
    assert not out.exists()
