import time, torch, sys
from common import *
from runner import *
from packed import *
from strands_decider.infer import load_engine
from lean import Lean
eng = load_engine(CKPT, device="cuda"); tok = eng.model.tokenizer; mk = make_state_fn(tok)
torso = eng.model.torso.merge_and_unload().eval(); eng.model.torso = torso
ln = Lean(torso)
for n in (300, 1000):
    st = mk(n, "a")
    ref = eng.ask(st, qs(4)).answers
    pe = PackedEngine(eng, ln, graph=True).ask(st, qs(4))
    pe2 = PackedEngine(eng, ln, graph=False).ask(st, qs(4))
    for k in ref:
        a, b, c = ref[k].model_dump(), pe[k].model_dump(), pe2[k].model_dump()
        f = lambda x: x.get("noul", x.get("probabilities", x.get("score")))
        print(n, k, f(a), f(b), f(c))
