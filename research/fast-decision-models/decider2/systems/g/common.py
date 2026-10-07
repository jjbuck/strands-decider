import glob, json, statistics as st, sys, time, torch
CKPT = glob.glob("/home/ubuntu/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*")[0]
base = ("user: Hi, I'd like to return the blue jacket from order #W1234567 and exchange the boots for a size 10. "
        "assistant: Sure, I can help with that. Let me look up your order. tool: {\"order_id\": \"#W1234567\", \"items\": "
        "[{\"name\": \"jacket\", \"color\": \"blue\", \"price\": 129.99}, {\"name\": \"boots\", \"size\": 9, \"price\": 89.5}], \"status\": \"delivered\"} ")
def make_state_fn(tok):
    def make_state(n_tokens, salt):
        ids = tok(base * (n_tokens // 60 + 2) + f" ref {salt}", add_special_tokens=False)["input_ids"][:n_tokens]
        return tok.decode(ids)
    return make_state
def qs(k):
    from strands_decider.schema import NoulQuestion, ChoiceQuestion, ScoreQuestion
    out = {"asked_human": NoulQuestion(instructions="Did the user explicitly ask to speak to a human agent?")}
    if k > 1: out["intent"] = ChoiceQuestion(instructions="What is the user's main intent?", criteria={"return": "return an item", "exchange": "exchange an item", "cancel": "cancel an order", "other": "something else"})
    if k > 2: out["urgency"] = ScoreQuestion(instructions="How urgent is the request?", criteria=["not urgent", "somewhat urgent", "very urgent"])
    if k > 3: out["refund"] = NoulQuestion(instructions="Is a refund requested?")
    return out
def timeit(fn, reps=15, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return dict(median=round(st.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1))], 2), min=round(ts[0], 2))
