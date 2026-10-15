"""llama.cpp as the torso of strands_decider.hidden_engine.HiddenStateEngine.

The engine (prompts, window fitting, the pointer head in numpy, the readouts) is the
package's own, torch-free and tested against SystemOneEngine. This module only binds the
torso forward to shim/decider_hidden.c, which runs a Q8_0 GGUF through libllama.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from strands_decider.hidden_engine import HiddenStateEngine

_F = ctypes.POINTER(ctypes.c_float)
_I = ctypes.POINTER(ctypes.c_int)


class Shim:
    """ctypes binding for shim/decider_hidden.c."""

    def __init__(self, lib_path: str, model_path: str, threads: int):
        lib = ctypes.CDLL(lib_path)
        lib.dh_load.argtypes = [ctypes.c_char_p, ctypes.c_int, _I]
        lib.dh_load.restype = ctypes.c_void_p
        lib.dh_forward_shared.argtypes = [
            ctypes.c_void_p,
            _I,
            ctypes.c_int,
            _I,
            _I,
            ctypes.c_int,
            _F,
        ]
        lib.dh_forward_shared.restype = ctypes.c_int
        n_embd = ctypes.c_int(0)
        self.handle = lib.dh_load(model_path.encode(), threads, ctypes.byref(n_embd))
        if not self.handle:
            raise RuntimeError(f"dh_load failed for {model_path}")
        self.lib = lib
        self.n_embd = n_embd.value

    def forward_shared(self, state_ids: list[int], q_ids: list[list[int]]) -> list[np.ndarray]:
        """Decode the state once; return each question's hidden states, [len(q), n_embd]."""
        flat = [t for q in q_ids for t in q]
        lens = [len(q) for q in q_ids]
        out = np.empty((len(flat), self.n_embd), dtype=np.float32)
        rc = self.lib.dh_forward_shared(
            self.handle,
            (ctypes.c_int * len(state_ids))(*state_ids),
            len(state_ids),
            (ctypes.c_int * len(flat))(*flat),
            (ctypes.c_int * len(lens))(*lens),
            len(lens),
            out.ctypes.data_as(_F),
        )
        if rc != 0:
            raise RuntimeError(f"dh_forward_shared rc={rc}")
        return np.split(out, np.cumsum(lens)[:-1])


def load_engine(
    lib_path: str, gguf_path: str, assets_dir: str, threads: int, **overrides: Any
) -> tuple[HiddenStateEngine, Shim]:
    """The engine over a loaded GGUF. `overrides` set HiddenEngineConfig fields."""
    shim = Shim(lib_path, gguf_path, threads)
    return HiddenStateEngine.from_assets(assets_dir, shim.forward_shared, **overrides), shim
