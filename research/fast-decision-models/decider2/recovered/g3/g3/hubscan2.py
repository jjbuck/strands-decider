import json, urllib.request, urllib.parse, sys, concurrent.futures as cf
ids = json.load(open("hub_ids.json"))
bad = ("gguf", "GGUF", "awq", "AWQ", "gptq", "GPTQ", "mlx", "MLX", "exl2", "-4bit", "-8bit", "bnb", "lora", "LoRA", "onnx", "ONNX", "fp8", "FP8", "int4", "int8", "nf4")
ids = [i for i in ids if not any(b in i for b in bad)]
print(len(ids), "after name filter", file=sys.stderr)
def get(i):
    url = f"https://huggingface.co/api/models/{i}?" + urllib.parse.urlencode([("expand[]", "config"), ("expand[]", "safetensors"), ("expand[]", "downloads"), ("expand[]", "cardData"), ("expand[]", "lastModified")])
    try:
        return i, json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as e:
        return i, None
out = {}
with cf.ThreadPoolExecutor(16) as ex:
    for i, d in ex.map(get, ids):
        if d: out[i] = d
json.dump(out, open("hub_meta.json", "w"))
print(len(out), file=sys.stderr)
