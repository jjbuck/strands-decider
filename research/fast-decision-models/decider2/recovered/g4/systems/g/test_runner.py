import time, torch
from common import *
from runner import *
from strands_decider.infer import load_engine
from lean import Lean
eng = load_engine(CKPT, device="cuda"); tok = eng.model.tokenizer; mk = make_state_fn(tok)
st = mk(300, "a")
for nq in (1, 4):
    ref = eng.ask(st, qs(nq)).answers
    torso = eng.model.torso.merge_and_unload().eval() if nq == 1 else torso
    eng.model.torso = torso
    for name, fe in [("hfgraph", FastEngine(eng, lambda ids: torso(input_ids=ids, use_cache=False).last_hidden_state, graph=True)),
                     ("lean_graph", FastEngine(eng, Lean(torso).forward, graph=True))]:
        out = fe.ask(st, qs(nq))
        for k in ref:
            print(nq, name, k, ref[k].model_dump() if hasattr(ref[k], "model_dump") else ref[k], "||", out[k].model_dump() if hasattr(out[k], "model_dump") else out[k])
