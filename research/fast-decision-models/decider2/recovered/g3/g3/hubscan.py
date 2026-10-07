import json, urllib.request, urllib.parse, sys
seen = {}
qs = ["moe", "A1B", "A0.6B", "A0.3B", "A0.5B", "A0.8B", "a400m", "a800m", "A1.5B", "A2B", "-A", "MoE-", "experts", "olmoe", "granitemoe", "smallthinker", "tiny-moe", "mini-moe", "nano"]
for q in qs:
    for sort in ("downloads", "lastModified"):
        url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(dict(search=q, sort=sort, direction=-1, limit=100))
        try:
            d = json.load(urllib.request.urlopen(url, timeout=30))
        except Exception as e:
            print("ERR", q, e, file=sys.stderr); continue
        for m in d:
            seen.setdefault(m["id"], m)
print(len(seen), "candidates", file=sys.stderr)
json.dump(list(seen.keys()), open("hub_ids.json", "w"))
