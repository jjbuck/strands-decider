"""Device selection: what `--device` resolves to, and how a missing MLX install is reported.

Needs neither weights nor a GPU: the availability checks are replaced, so every ordering is
exercised on any machine, CI's Linux runners included.
"""

from __future__ import annotations

import pytest
import torch

from strands_decider import cli, infer


@pytest.mark.parametrize(
    ("cuda", "mlx", "mps", "expected"),
    [
        (True, True, True, "cuda"),
        (False, True, True, "mlx"),
        (False, False, True, "mps"),
        (False, False, False, "cpu"),
    ],
)
def test_auto_device_prefers_cuda_then_mlx_then_mps(monkeypatch, cuda, mlx, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(infer, "mlx_available", lambda: mlx)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert cli._auto_device() == expected


def test_mlx_is_unavailable_off_apple_silicon(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert not infer.mlx_available()


def test_device_mlx_without_the_extra_names_it_before_loading_anything(monkeypatch):
    monkeypatch.setattr(infer, "mlx_available", lambda: False)

    def refuse(*args, **kwargs):
        raise AssertionError("nothing should load")

    monkeypatch.setattr(infer.StrandsDeciderModel, "load", refuse)
    with pytest.raises(RuntimeError, match=r"strands-decider\[mlx\]"):
        infer.load_engine("any/checkpoint", device="mlx")


def test_the_server_builds_the_mlx_engine_for_device_mlx(monkeypatch):
    from fastapi.testclient import TestClient

    from strands_decider import server

    class _Engine:
        cfg = type("Cfg", (), {"model_name": "hobson", "device": "mlx", "use_prefix_cache": True})()
        model = type("Model", (), {"config": type("Config", (), {
            "base_model": "stub", "num_slots": 24, "temperature": 1.0, "max_length": 512})()})()

    calls = []
    monkeypatch.setattr(server, "load_mlx", lambda checkpoint, **kw: calls.append((checkpoint, kw)) or _Engine())
    monkeypatch.setattr(server.StrandsDeciderModel, "load", lambda *a, **k: pytest.fail("torch load"))
    health = TestClient(server.create_app("org/hobson", device="mlx")).get("/health").json()
    assert health["device"] == "mlx"
    assert calls == [("org/hobson", {"use_prefix_cache": True, "model_name": "hobson"})]
