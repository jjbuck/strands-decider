"""Token classes for B8 (value-bearing tokens). Decode token by token, build the text with char offsets, label char spans by regex,
give each token the class of its first non-space character (priority order below)."""
import re

CLASSES = ['email', 'date', 'amount', 'phone', 'id', 'digit', 'json_key', 'json_punct', 'newline', 'prose', 'other']
PATS = [
    ('email', re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')),
    ('date', re.compile(r'\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}(?:, \d{4})?')),
    ('amount', re.compile(r'\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\.\d{2}\b|\b\d+(?:\.\d+)?\s?(?:USD|dollars|%)')),
    ('phone', re.compile(r'\+?\d[\d\-\s().]{8,}\d')),
    ('id', re.compile(r'\b(?=[A-Za-z_\-]*\d)(?=\w*[A-Za-z_])[A-Za-z0-9_\-#]{4,}\b|#\d+')),
    ('digit', re.compile(r'\d+')),
    ('json_key', re.compile(r'"[A-Za-z_][\w ]*"\s*:')),
    ('json_punct', re.compile(r'[{}\[\]":,]')),
    ('newline', re.compile(r'\n+')),
    ('prose', re.compile(r'[A-Za-z]+')),
]


def classify(tok, ids):
    pieces = [tok.decode([t]) for t in ids]
    text = ''.join(pieces)
    lab = ['other'] * len(text)
    for name, pat in reversed(PATS):          # lowest priority first; higher priority overwrites
        for m in pat.finditer(text):
            for c in range(m.start(), m.end()): lab[c] = name
    out = []; pos = 0
    for p in pieces:
        cls = 'other'
        for j, ch in enumerate(p):
            if not ch.isspace() or ch == '\n':
                cls = lab[pos + j] if pos + j < len(lab) else 'other'; break
        out.append(cls); pos += len(p)
    return out
