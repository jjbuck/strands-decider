"""ideas_diagrams_s.py: extra hand-drawn diagrams for the S and HW sections (python3 ideas_diagrams_s.py).

Kept in its own file so parallel edits to ideas_diagrams.py don't collide.
"""
from sketch import *


def d21_cp_wavefront():
    s = Sketch(960, 420)
    s.text(30, 38, 'S5: split the tokens across 4 GPUs; pass the small GDN state along', 21)
    x0, bw, gap, skew = 150, 44, 6, 30
    nl = 12
    for r in range(4):
        y = 90 + r * 62
        s.text(40, y + 26, f'GPU {r}', 16)
        s.text(40, y + 44, f'tokens {r * 250}–{r * 250 + 249}', 12, GRAY)
        for i in range(nl):
            x = x0 + r * skew + i * (bw + gap)
            c, f = ((PURPLE, PURPLE_F) if i % 4 == 3 else (BLUE, BLUE_F))
            s.rect(x, y, bw, 34, stroke=c, fill=f, gap=5, sw=1.2)
            if r < 3 and i % 4 != 3:
                s.arrow(x + bw - 6, y + 36, x + skew + 6, y + 60, ORANGE, 1.2, 6)
    s.text(x0, 365, 'each box is one layer, 12 of 24 shown (blue GDN, purple attention); every GPU holds all the weights', 14, INK)
    s.text(x0, 388, 'orange: each GDN layer hands its 1 MB state to the next GPU. The skew is paid once (~0.5 ms), not per layer.', 14, ORANGE)
    s.text(x0 + 4 * skew + nl * (bw + gap) - 10, 75, 'predicted 8–10 ms on 4× 3090 (unmeasured)', 14, RED, 'end')
    s.save('d21_cp_wavefront')


if __name__ == '__main__':
    d21_cp_wavefront(); print('drew d21_cp_wavefront')
