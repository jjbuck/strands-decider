"""ideas_diagrams.py: the hand-drawn concept diagrams in docs/figures/ideas/ (python3 ideas_diagrams.py)."""
import random
import math as m
from sketch import *


def d01_decision_pass():
    s = Sketch(960, 470)
    s.text(30, 38, 'One decision = one forward pass (no generation)', 22)
    x0, y0 = 40, 380
    cx = x0
    for n, c, f in ((14, BLUE, BLUE_F), (6, ORANGE, ORANGE_F)):
        for i in range(n):
            s.rect(cx, y0, 34, 34, stroke=c, fill=f, gap=6); cx += 38
        cx += 12
    s.text(x0 + 14 * 38 / 2, y0 + 62, 'state rows (~90%): conversation, tool outputs, documents', 15, BLUE, 'middle')
    qx = x0 + 14 * 38 + 12
    s.text(qx + 3 * 38, y0 + 62, 'question rows: rubric, options, <answer>', 15, ORANGE, 'middle')
    for k in (2, 4, 5):
        s.rect(qx + k * 38 - 3, y0 - 3, 40, 40, stroke=RED, sw=2.2)
    s.rect(40, 150, 790, 190, stroke=INK, fill=GRAY_F, gap=11)
    for i in range(1, 6):
        s.line(40, 150 + i * 32, 830, 150 + i * 32, GRAY, 1.0, double=False)
    s.text(435, 250, '24 layers: 18 GDN + 6 attention, each with an MLP', 18, INK, 'middle', bg=True)
    s.text(435, 280, 'every row goes through every matrix multiply', 15, GRAY, 'middle', bg=True)
    for k in range(0, 20, 3):
        s.arrow(x0 + 17 + k * 38, y0 - 6, x0 + 17 + k * 38, 345, GRAY, 1.3, 7)
    s.rect(560, 60, 270, 56, stroke=BLUE, fill=BLUE_F, gap=6, sw=2.2)
    s.text(695, 94, 'pointer head', 18, INK, 'middle', bg=True)
    for k in (2, 4, 5):
        s.arrow(qx + k * 38 + 17, 148, 640 + (k - 2) * 50, 120, RED, 1.6, 8)
    s.arrow(830, 88, 880, 88, INK, 1.8)
    s.lines(888, 72, ['p(A)', 'p(B)', 'p(C)'], 15)
    s.text(40, 132, 'read out: K option-end rows + the <answer> row (red)', 15, RED)
    s.save('d01_decision_pass')


def d02_time_split():
    s = Sketch(960, 330)
    s.text(30, 38, 'Where 52.7 ms goes at a 1,000-token state (A10G, bf16, fused)', 21)
    parts = [(46.0, BLUE, BLUE_F, ''), (3.4, GREEN, GREEN_F, 'GDN scan 3.4'), (1.2, TEAL, TEAL_F, 'conv 1.2'),
             (0.9, RED, RED_F, 'attention 0.9'), (1.3, GRAY, GRAY_F, 'other 1.3')]
    x, y, sc = 40, 90, 860 / 52.8
    xs = []
    for ms, c, f, lab in parts:
        w = ms * sc
        s.rect(x, y, w, 70, stroke=c, fill=f, gap=6 if w > 30 else 4)
        xs.append((x, w, lab, c)); x += w
    s.text(40 + 46.0 * sc / 2, y + 44, 'matrix multiplies: 46.0 ms (87%)', 20, BLUE, 'middle', bg=True)
    for i, (x0, w, lab, c) in enumerate(xs[1:]):
        lx = 600 + i * 95
        s.line(x0 + w / 2, y + 72, lx, 205, c, 1.3)
        s.text(lx, 225, lab, 15, c, 'middle')
    s.text(40, 270, 'Attention is 0.9 ms (1.7%). All token mixing together is about 10%.', 17)
    s.text(40, 300, 'latency ≈ 2.75 GFLOP per row × rows ÷ the matrix-multiply rate', 17, PURPLE)
    s.save('d02_time_split')


def d03_floor():
    s = Sketch(960, 420)
    s.text(30, 38, 'Two limits: loading the weights, or doing the arithmetic', 21)
    ox, oy, W, H = 90, 360, 760, 270
    s.arrow(ox, oy, ox + W + 20, oy, INK, 1.8); s.arrow(ox, oy, ox, oy - H - 20, INK, 1.8)
    s.text(ox + W + 20, oy + 30, 'rows in the request (N)', 15, INK, 'end')
    s.text(ox - 18, oy - H / 2, 'time', 15, INK, 'middle', rotate=-90)
    ridge = ox + 230
    s.line(ox, oy - 90, ox + W, oy - 90, ORANGE, 2.2)
    s.text(ox + W - 10, oy - 100, 'weight loading  b·P / B', 15, ORANGE, 'end')
    s.line(ox, oy - 20, ox + W, oy - 250, BLUE, 2.2)
    s.text(ox + 330, oy - 170, 'arithmetic  2·P·N / (η·R)', 15, BLUE)
    s.line(ridge, oy, ridge, oy - 110, GRAY, 1.4, dash=6)
    s.text(ridge, oy + 26, 'N* ≈ 150 rows on a 3090', 15, GRAY, 'middle')
    s.text(ridge, oy + 46, '(~330 on a 4090)', 13, GRAY, 'middle')
    s.ellipse(ridge - 8, oy - 90, 7, 7, PURPLE, 2.0)
    s.text(ridge + 14, oy - 62, 'median JevBench request (143 rows)', 14, PURPLE)
    xx = ox + 600; yy = oy - 20 - 600 * 230 / W
    s.ellipse(xx, yy, 7, 7, PURPLE, 2.0)
    s.text(xx + 14, yy + 5, '1,000-token request', 14, PURPLE)
    s.lines(ox + 30, 100, ['Past N*, arithmetic sets the time:', 'fewer rows, fewer layers per row,', 'or cheaper multiplies are the levers.'], 15, INK)
    s.save('d03_floor')


def d04_rotation():
    s = Sketch(960, 360)
    s.text(30, 38, 'P1: rotate first, then round to int8', 21)
    r = random.Random(3)
    vals = [r.gauss(0, 1) for _ in range(24)]
    vals[5], vals[17] = 9.0, -7.5
    ox, oy = 50, 210
    s.line(ox, oy, ox + 24 * 13, oy, GRAY, 1.0)
    for i, v in enumerate(vals):
        s.line(ox + 6 + i * 13, oy, ox + 6 + i * 13, oy - v * 14, RED if abs(v) > 5 else BLUE, 3.0, 0.4, double=False)
    s.text(ox + 156, 315, 'one token, raw: a few huge values', 15, RED, 'middle')
    s.text(ox + 156, 337, '(max/rms 15–40) waste the int8 grid', 15, RED, 'middle')
    s.arrow(400, 210, 520, 210, INK, 2.0)
    s.text(460, 195, '× R', 18, PURPLE, 'middle')
    s.text(460, 240, '(Hadamard)', 13, PURPLE, 'middle')
    ox2 = 560
    rv = [r.gauss(0, 2.2) for _ in range(24)]
    for k in range(-3, 4):
        s.line(ox2 - 4, oy - k * 22, ox2 + 24 * 13 + 4, oy - k * 22, GRAY_F, 0.8, double=False)
    for i, v in enumerate(rv):
        s.line(ox2 + 6 + i * 13, oy, ox2 + 6 + i * 13, oy - v * 14, BLUE, 3.0, 0.4, double=False)
    s.text(ox2 + 156, 315, 'rotated: same information, flat', 15, BLUE, 'middle')
    s.text(ox2 + 156, 337, 'so int8 rounding loses ~1%', 15, BLUE, 'middle')
    s.text(480, 82, 'X·W = (X·R)·(Rᵀ·W)    and Rᵀ·W is folded into the stored weights', 17, PURPLE, 'middle')
    s.save('d04_rotation')


def d05_rowrole():
    s = Sketch(960, 380)
    s.text(30, 38, 'P10 / P11: spend bits by row role, then by layer', 21)
    ox, oy = 60, 80
    s.rect(ox, oy, 120, 190, stroke=ORANGE, fill=ORANGE_F)
    s.rect(ox, oy + 190, 120, 40, stroke=BLUE, fill=BLUE_F, gap=5)
    s.text(ox + 60, oy + 105, 'state rows', 15, ORANGE, 'middle', bg=True)
    s.text(ox + 60, oy + 128, 'int4', 18, ORANGE, 'middle', bg=True)
    s.text(ox + 60, oy + 216, 'question int8', 14, BLUE, 'middle', bg=True)
    s.text(ox + 165, oy + 120, '×', 26, INK, 'middle')
    s.rect(ox + 200, oy + 60, 150, 110, stroke=INK, fill=GRAY_F)
    s.text(ox + 275, oy + 122, 'W', 22, INK, 'middle', bg=True)
    s.text(ox + 175, 345, 'row role (P10): 3.9% of decisions change', 15, INK, 'middle')
    gx, gy = 520, 90
    s.text(gx, gy - 14, 'k64rr (P11): the 96 matrix multiplies', 16)
    r = random.Random(5); mixed = set(r.sample(range(40, 96), 32))
    for i in range(96):
        x, y = gx + (i % 16) * 24, gy + 10 + (i // 16) * 30
        if i in mixed:
            s.rect(x, y, 20, 13, stroke=ORANGE, fill=ORANGE_F, gap=4, sw=1.2); s.rect(x, y + 13, 20, 7, stroke=BLUE, sw=1.2)
        else:
            s.rect(x, y, 20, 20, stroke=BLUE, fill=BLUE_F, gap=4, sw=1.2)
    s.lines(gx, 300, ['blue: int8 on all rows (the 64 most sensitive)', 'split: row role (state int4, question int8)'], 14, INK)
    s.text(gx, 345, 'meets the bar: 0.65% flips, 0.877x of W8A8', 15, GREEN)
    s.save('d05_rowrole')


def d06_exit():
    s = Sketch(960, 420)
    s.text(30, 38, 'R1: answer at layer 16 when confident', 21)
    ox, oy = 150, 380
    for i in range(24):
        y = oy - (i + 1) * 13
        s.rect(ox, y, 200, 11, stroke=BLUE if i < 16 else GRAY, sw=1.2)
    s.text(ox - 14, oy - 8 * 13, 'layers 1–16', 15, BLUE, 'end')
    s.text(ox - 14, oy - 20 * 13, 'layers 17–24', 15, GRAY, 'end')
    yx = oy - 16 * 13 + 4
    s.arrow(ox + 205, yx, ox + 290, yx, INK, 1.8)
    s.rect(ox + 295, yx - 26, 150, 52, stroke=PURPLE, fill=PURPLE_F, gap=6)
    s.text(ox + 370, yx + 6, 'exit head', 17, INK, 'middle', bg=True)
    s.arrow(ox + 450, yx, ox + 530, yx, INK, 1.8)
    s.text(ox + 490, yx - 12, 'margin > τ ?', 14, PURPLE, 'middle')
    s.text(ox + 540, yx + 6, 'yes (94%): answer now', 16, GREEN)
    s.arrow(ox + 370, yx - 30, ox + 250, oy - 21 * 13, ORANGE, 1.6, curve=-25, dash=5)
    s.text(ox + 390, oy - 22 * 13 + 4, 'no (6%): run layers 17–24', 15, ORANGE)
    s.lines(ox + 450, 330, ['0 of 6,216 real decisions changed', '0.688x of W8A8 end to end'], 15, INK)
    s.save('d06_exit')


def d07_depthsplit():
    s = Sketch(960, 420)
    s.text(30, 38, 'R2 / M3: shallow state, deep question', 21)
    ox, oy = 120, 380
    for i in range(24):
        y = oy - (i + 1) * 13
        if i < 8:
            s.rect(ox, y, 260, 11, stroke=BLUE, fill=BLUE_F, gap=5, sw=1.2)
        s.rect(ox + 330, y, 100, 11, stroke=ORANGE, fill=ORANGE_F, gap=5, sw=1.2)
    s.text(ox + 130, oy + 26, 'state rows: 8 of 24 layers', 15, BLUE, 'middle')
    s.text(ox + 380, oy + 26, 'question rows: all 24', 15, ORANGE, 'middle')
    for k in (10, 14, 18, 22):
        s.arrow(ox + 262, oy - 8 * 13 + 6, ox + 328, oy - k * 13 + 6, GRAY, 1.3, 8, curve=10)
    s.text(ox + 200, oy - 15 * 13, 'later layers read the', 13, GRAY, 'middle', bg=True)
    s.text(ox + 200, oy - 15 * 13 + 16, "state's layer-8 keys/values", 13, GRAY, 'middle', bg=True)
    s.lines(620, 140, ['state FLOPs: 0.35x', '1.73x faster at 1,000 tokens', '2.36x at 4,000', 'meets the accuracy bar,',
                       'loses multi-step deduction', '(RuleTaker depth 3: −17 points)'], 16, INK)
    s.save('d07_depthsplit')


def d08_compiled_docs():
    s = Sketch(960, 330)
    s.text(30, 38, "A6: compile the deployment's constant blocks once", 21)
    segs = [('frame', 50, GRAY, GRAY_F, False), ('KB document', 190, GREEN, GREEN_F, True), ('user msg', 100, BLUE, BLUE_F, False),
            ('tool result', 130, BLUE, BLUE_F, False), ('hook note', 90, GREEN, GREEN_F, True), ('KB document', 150, GREEN, GREEN_F, True),
            ('question', 100, ORANGE, ORANGE_F, False)]
    x = 40
    for lab, w, c, f, comp in segs:
        s.rect(x, 110, w, 60, stroke=c, fill=f, gap=5 if comp else 9, dash=4 if comp else None)
        s.text(x + w / 2, 147, lab, 14, c, 'middle', bg=True)
        x += w + 6
    s.lines(40, 220, ['green, dashed: precomputed once per deployment (K/V re-rotated to their position)',
                      'blue / orange: computed live per request'], 15, INK)
    s.text(40, 280, '46.5% of gate-state tokens are deployment constants: 1.9x on banking traffic', 16, PURPLE)
    s.save('d08_compiled_docs')


def d09_vocab():
    s = Sketch(960, 300)
    s.text(30, 38, "A7: a reader can carry its deployment's own vocabulary", 21)
    toks = ['"acct', '_', '750', '650', '",', ' "', 'daily', '_', 'transfer', '_', 'limit', '":']
    x = 40; xs = []
    for t in toks:
        w = max(44, s.text_width(t, 15) + 16)
        s.rect(x, 90, w, 40, stroke=BLUE, fill=BLUE_F, gap=6)
        s.text(x + w / 2, 116, t, 15, INK, 'middle', bg=True); xs.append((x, w)); x += w + 4
    for a, b in ((0, 4), (5, 11)):
        x0 = xs[a][0]; x1 = xs[b][0] + xs[b][1]
        s.brace(x1, 140, x0, 140, PURPLE, 10)
        s.rect(x0, 190, x1 - x0, 44, stroke=PURPLE, fill=PURPLE_F, gap=7)
        s.text((x0 + x1) / 2, 218, '1 super-token', 15, INK, 'middle', bg=True)
    s.text(40, 275, 'no output layer, so a bigger vocabulary costs nothing: 1.6–2.7x fewer rows, 2.0x faster on real traffic', 15, INK)
    s.save('d09_vocab')


def d10_q_in_weights():
    s = Sketch(960, 300)
    s.text(30, 38, 'A4: compile each deployed question into weights', 21)
    for i in range(16):
        s.rect(40 + i * 22, 100, 20, 30, stroke=ORANGE, fill=ORANGE_F, gap=5, sw=1.2)
    s.text(40 + 8 * 22, 160, 'question text: 78–1,201 tokens, every request', 15, ORANGE, 'middle')
    s.arrow(410, 115, 500, 115, INK, 2.0)
    s.rect(530, 80, 150, 70, stroke=PURPLE, fill=PURPLE_F, gap=6)
    s.text(605, 112, 'ΔW for this', 15, INK, 'middle', bg=True)
    s.text(605, 134, 'question', 15, INK, 'middle', bg=True)
    s.text(705, 121, '+', 22, INK, 'middle')
    for i in range(4):
        s.rect(730 + i * 26, 100, 22, 30, stroke=ORANGE, fill=ORANGE_F, gap=5, sw=1.2)
    s.text(780, 160, 'K+1 slot rows', 15, ORANGE, 'middle')
    s.text(40, 230, '4 deployed questions: 2.3x at 1,000 tokens, 5.4x at 64. Agreement .86 (the 23-way', 15, INK)
    s.text(40, 252, 'procedure questions are near-ties). Questions not in the deployment stay in context.', 15, INK)
    s.save('d10_q_in_weights')


def d11_rows_vs_dirs():
    s = Sketch(960, 380)
    s.text(30, 38, 'B1: where the decision is sensitive (state rows, layers 0–8)', 21)
    r = random.Random(11)
    ox, oy = 60, 280
    vals = sorted([r.uniform(0.3, 1.0) for _ in range(40)], reverse=True)
    for i, v in enumerate(vals):
        s.line(ox + i * 9, oy, ox + i * 9, oy - v * 110, BLUE, 3.0, 0.3, double=False)
    s.line(ox - 5, oy, ox + 370, oy, GRAY, 1.0)
    s.lines(ox, oy + 30, ['by direction: spread out', '90% needs 349–1,257 of 2,048 directions'], 15, BLUE)
    ox2 = 520
    vals2 = sorted([r.paretovariate(1.3) for _ in range(40)], reverse=True)
    mx = vals2[0]
    for i, v in enumerate(vals2):
        s.line(ox2 + i * 9, oy, ox2 + i * 9, oy - v / mx * 190, RED if i < 3 else ORANGE, 3.0, 0.3, double=False)
    s.line(ox2 - 5, oy, ox2 + 370, oy, GRAY, 1.0)
    s.lines(ox2, oy + 30, ['by row: concentrated', 'top 5% of state rows hold 63%, top 25% hold 90%'], 15, RED)
    s.text(480, 75, 'a low-rank correction cannot help; precision should follow the rows the question reads', 15, PURPLE, 'middle')
    s.save('d11_rows_vs_dirs')


def d12_waterfill():
    s = Sketch(960, 360)
    s.text(30, 38, 'B5: decision-weighted transform coding (bits by reverse water-filling)', 21)
    ox, oy = 70, 290
    vals = [m.exp(-i / 8.0) for i in range(48)]
    theta = 0.12
    for i, v in enumerate(vals):
        c = BLUE if v > 0.55 else ORANGE if v > theta else GRAY
        s.line(ox + i * 15, oy, ox + i * 15, oy - v * 200, c, 5.0, 0.3, double=False)
    s.line(ox - 10, oy - theta * 200, ox + 48 * 15, oy - theta * 200, TEAL, 1.6, dash=6)
    s.text(ox + 48 * 15 + 6, oy - theta * 200 + 5, 'θ', 18, TEAL)
    s.text(ox + 30, 80, '8 bits', 16, BLUE)
    s.text(ox + 160, 175, '4 bits', 16, ORANGE)
    s.text(ox + 560, 245, '0 bits: replaced by the mean, so the inner dimension shrinks', 14, GRAY, 'middle')
    s.text(ox, oy + 30, 'directions sorted by variance × decision sensitivity', 15, INK)
    s.text(ox, oy + 52, 'best 4-bit format found: 1.29% flips at 4.5 bits, but its per-layer transforms are too costly to run', 15, RED)
    s.save('d12_waterfill')


def d13_sparsity():
    s = Sketch(960, 340)
    s.text(30, 38, 'B11: 2:4 sparsity, chosen by the decision', 21)
    r = random.Random(4)
    ox, oy = 60, 90
    for row in range(5):
        for g in range(4):
            keep = r.sample(range(4), 2)
            for k in range(4):
                x = ox + (g * 4 + k) * 26 + g * 10; y = oy + row * 30
                if k in keep:
                    s.rect(x, y, 22, 22, stroke=BLUE, fill=BLUE_F, gap=4, sw=1.3)
                else:
                    s.rect(x, y, 22, 22, stroke=GRAY, sw=1.0)
                    s.line(x + 5, y + 5, x + 17, y + 17, GRAY, 1.0, double=False); s.line(x + 17, y + 5, x + 5, y + 17, GRAY, 1.0, double=False)
    for g in range(4):
        x = ox + g * 4 * 26 + g * 10
        s.brace(x + 100, oy - 8, x, oy - 8, GRAY, 7)
    s.text(ox + 220, oy + 185, 'each group of 4 keeps 2: the sparse tensor cores skip half the multiplies', 15, INK, 'middle')
    s.lines(560, 110, ["pick the 2 by the decision's sensitivity", '(2.2–2.4x lower decision error', 'than per-layer criteria)', '',
                       'meets the bar on state rows of', 'layers 12–23: 0.83–0.92x GEMM time'], 15, PURPLE)
    s.save('d13_sparsity')


def d14_neurons():
    s = Sketch(960, 360)
    s.text(30, 38, "B12: remove MLP neurons the deployment's decisions don't need", 21)
    ox, oy = 60, 100
    s.rect(ox, oy + 40, 40, 120, stroke=INK, fill=GRAY_F)
    s.text(ox + 20, oy + 185, 'x', 16, INK, 'middle')
    r = random.Random(9)
    for i in range(36):
        x = ox + 90 + i * 14
        dead = r.random() < 0.6
        s.rect(x, oy, 10, 200, stroke=GRAY if dead else ORANGE, fill=None if dead else ORANGE_F, gap=4, sw=1.0)
        if dead:
            s.line(x - 2, oy + 90, x + 12, oy + 110, RED, 1.2, double=False)
    s.text(ox + 90 + 18 * 14, oy + 232, '6,144 SwiGLU neurons per layer (rows of W_gate, W_up; columns of W_down)', 14, INK, 'middle')
    s.lines(640, 130, ['ranked by decision saliency,', 'kept ones re-fit (decision-weighted),', 'removed means become a bias', '',
                       '60% of layers 12–23 removed:', 'meets the bar, 0.82x GEMM time'], 15, PURPLE)
    s.save('d14_neurons')


def d15_exact():
    s = Sketch(960, 330)
    s.text(30, 38, 'B13: two exact eliminations', 21)
    ox, oy = 50, 90
    s.text(ox, oy, 'layer 23, state rows:', 16)
    for i, (lab, keep) in enumerate((('q', False), ('k, v', True), ('attn out', False), ('MLP', False))):
        x = ox + i * 105
        s.rect(x, oy + 20, 90, 50, stroke=BLUE if keep else GRAY, fill=BLUE_F if keep else None, gap=6)
        s.text(x + 45, oy + 52, lab, 15, INK if keep else GRAY, 'middle', bg=True)
        if not keep:
            s.line(x + 5, oy + 25, x + 85, oy + 65, RED, 1.6)
    s.text(ox, oy + 105, 'only their keys/values are ever read', 15, INK)
    ox2 = 520
    s.text(ox2, oy, 'layer 0:', 16)
    s.rect(ox2, oy + 20, 90, 50, stroke=INK, fill=GRAY_F, gap=7); s.text(ox2 + 45, oy + 52, 'token id', 15, INK, 'middle', bg=True)
    s.arrow(ox2 + 95, oy + 45, ox2 + 160, oy + 45)
    s.rect(ox2 + 165, oy + 20, 200, 50, stroke=GREEN, fill=GREEN_F, gap=6); s.text(ox2 + 265, oy + 52, 'table: x·W_in', 15, INK, 'middle', bg=True)
    s.text(ox2, oy + 105, 'its input depends only on the token: precompute it', 15, INK)
    s.text(40, 280, 'bit-exact; 0.958x of W8A8 end to end (most of it from layer 23)', 16, PURPLE)
    s.save('d15_exact')


def d16_segments():
    s = Sketch(960, 460)
    s.text(30, 38, 'M2: state segments attend only to themselves; question rows read everything', 21)
    ox, oy, n, cell = 80, 80, 12, 26
    for (a, b, c, f) in ((0, 3, GREEN, GREEN_F), (3, 5, BLUE, BLUE_F), (5, 8, GREEN, GREEN_F), (8, 10, BLUE, BLUE_F)):
        s.rect(ox + a * cell, oy + a * cell, (b - a) * cell, (b - a) * cell, stroke=c, fill=f, gap=5)
    s.rect(ox, oy + 10 * cell, 12 * cell, 2 * cell, stroke=ORANGE, fill=ORANGE_F, gap=5)
    s.rect(ox, oy, n * cell, n * cell, stroke=GRAY, sw=1.0)
    s.text(ox + n * cell / 2, oy - 10, 'keys (attended to)', 14, GRAY, 'middle')
    s.text(ox - 14, oy + n * cell / 2, 'rows (queries)', 14, GRAY, 'middle', rotate=-90)
    s.lines(450, 110, ['green blocks: documents and hook notes,', 'now independent, so precomputed exactly',
                       '(228 of 228 decisions unchanged)', '', 'blue blocks: messages, tool calls, results',
                       '', 'orange: question rows read every segment', 'at every layer (GDN state composed exactly)',
                       '', 'state depth 12: meets an accuracy bar,', '1.7–1.8x faster on 84 real requests (bf16)'], 15, INK)
    s.save('d16_segments')


def d17_stack():
    s = Sketch(960, 300)
    s.text(30, 38, 'C1: k64rr + the layer-16 exit', 21)
    s.rect(40, 100, 380, 70, stroke=BLUE, fill=BLUE_F, gap=7)
    s.text(230, 132, 'layers 1–16', 17, INK, 'middle', bg=True)
    s.text(230, 156, 'int8 (a few mixed 4/8)', 14, INK, 'middle', bg=True)
    s.arrow(425, 135, 480, 135)
    s.rect(485, 105, 120, 60, stroke=PURPLE, fill=PURPLE_F, gap=6); s.text(545, 141, 'exit head', 15, INK, 'middle', bg=True)
    s.arrow(610, 120, 700, 90); s.text(710, 94, '94%: answer (23.2 ms at 1,000 tokens)', 15, GREEN)
    s.arrow(610, 150, 660, 190)
    s.rect(665, 175, 250, 50, stroke=ORANGE, fill=ORANGE_F, gap=6); s.text(790, 206, 'layers 17–24, mixed 4/8', 14, INK, 'middle', bg=True)
    s.text(40, 262, '120 real requests: 50.6 ms mean against 79.7 for W8A8 (0.635x), decisions at the fidelity bar', 15, INK)
    s.save('d17_stack')


def d18_nki():
    s = Sketch(960, 390)
    s.text(30, 38, "HW3: keep Inferentia's engines busy", 21)

    def lane(y0, title, blocks, busy):
        s.text(40, y0 - 12, title, 16)
        for j, eng in enumerate(('Tensor', 'Vector', 'Scalar')):
            y = y0 + j * 34
            s.text(110, y + 20, eng, 14, GRAY, 'end')
            s.line(120, y + 28, 900, y + 28, GRAY_F, 1.0, double=False)
            for (x, w, c, f) in blocks.get(j, []):
                s.rect(x, y + 4, w, 22, stroke=c, fill=f, gap=4, sw=1.2)
        s.text(900, y0 + 122, busy, 15, INK, 'end')
    b1 = {0: [(130 + i * 80, 26, BLUE, BLUE_F) for i in range(9)], 1: [(160 + i * 80, 22, BLUE, BLUE_F) for i in range(9)],
          2: [(186 + i * 80, 14, BLUE, BLUE_F) for i in range(9)]}
    lane(85, 'J8: one head per program, one dependent chain', b1, 'engines 19–24% busy')
    cols = [(BLUE, BLUE_F), (ORANGE, ORANGE_F), (GREEN, GREEN_F), (PURPLE, PURPLE_F)]
    b2 = {0: [(130 + i * 31, 28, *cols[i % 4]) for i in range(24)], 1: [(150 + i * 33, 24, *cols[(i + 1) % 4]) for i in range(22)],
          2: [(170 + i * 40, 18, *cols[(i + 2) % 4]) for i in range(18)]}
    lane(240, 'N1: 4 heads interleaved per tile, no transposes in the solve', b2, 'Tensor Engine 83% busy, 1.47x faster')
    s.save('d18_nki')


def d19_strassen():
    s = Sketch(960, 320)
    s.text(30, 38, 'S10: Strassen trades multiplies for additions', 21)
    s.lines(40, 90, ['ordinary 2×2 blocks: 8 block products', 'Strassen: 7 products + 18 block additions', '',
                     'but the additions widen the inputs:', 'int8 + int8 = a 9-bit number'], 16, INK)
    s.rect(560, 80, 70, 40, stroke=BLUE, fill=BLUE_F, gap=5); s.text(595, 107, 'int8', 15, INK, 'middle', bg=True)
    s.text(650, 107, '+', 20, INK, 'middle')
    s.rect(670, 80, 70, 40, stroke=BLUE, fill=BLUE_F, gap=5); s.text(705, 107, 'int8', 15, INK, 'middle', bg=True)
    s.arrow(750, 100, 800, 100)
    s.rect(805, 80, 80, 40, stroke=RED, fill=RED_F, gap=5); s.text(845, 107, '9 bits', 15, INK, 'middle', bg=True)
    s.lines(560, 160, ["doesn't fit the int8 tensor cores, so inputs", 'must drop to 7 bits: 1.48% of decisions change', '',
                       'and it was never faster than the best', 'dense kernel at the same precision'], 15, RED)
    s.save('d19_strassen')


def d20_next_rows():
    s = Sketch(960, 330)
    s.text(30, 38, 'Next: precision per row, chosen by the question (B1 + B5)', 21)
    ox = 40
    r = random.Random(2); hot = set(r.sample(range(30), 6)) | {28, 29}
    for i in range(30):
        c, f = (BLUE, BLUE_F) if i in hot else (ORANGE, ORANGE_F)
        s.rect(ox + i * 22, 120, 20, 40, stroke=c, fill=f, gap=5, sw=1.2)
    for i in range(4):
        s.rect(ox + 30 * 22 + 12 + i * 22, 120, 20, 40, stroke=BLUE, fill=BLUE_F, gap=5, sw=1.2)
    s.text(ox + 15 * 22, 185, 'state rows: int4, except the few the question will read (int8)', 15, INK, 'middle')
    s.text(ox + 30 * 22 + 56, 185, 'question: int8', 14, BLUE, 'middle')
    s.lines(ox, 230, ['a small per-deployment lookahead predicts which rows matter before layer 1',
                      '(oracle top-20% rows: 2.4x less decision error; with B5: close to W8A8)',
                      'cost ≈ 0.8 × int4 + 0.2 × int8 ≈ 0.6x of W8A8 GEMM time (arithmetic)'], 15, PURPLE)
    s.save('d20_next_rows')


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('d') and k[1:3].isdigit() and callable(f):
            f(); print('drew', k)
