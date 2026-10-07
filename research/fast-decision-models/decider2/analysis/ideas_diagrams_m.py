"""ideas_diagrams_m.py: concept diagrams for the M section (python3 ideas_diagrams_m.py). Same style as ideas_diagrams.py;
kept in its own file so parallel edits to ideas_diagrams.py can't collide."""
from sketch import *


def d22_m1_mixers():
    s = Sketch(960, 450)
    s.text(30, 38, 'M1: two ways for rows to read earlier rows', 21)
    # left: GDN, a chain of state updates
    s.text(40, 82, 'GDN (hobson, 18 layers): a running state per head', 16, BLUE)
    x, y = 50, 112
    for i in range(5):
        s.rect(x + i * 82, y, 56, 56, stroke=BLUE, fill=BLUE_F, gap=6)
        s.text(x + i * 82 + 28, y + 34, f'S{i + 1}', 15, INK, 'middle', bg=True)
        if i < 4:
            s.arrow(x + i * 82 + 58, y + 28, x + i * 82 + 80, y + 28, INK, 1.5, 8)
        s.arrow(x + i * 82 + 28, y + 100, x + i * 82 + 28, y + 60, GRAY, 1.2, 7)
        s.text(x + i * 82 + 28, y + 120, f'row {i + 1}', 13, GRAY, 'middle')
    s.lines(40, 300, ['each 128 x 128 state is updated row by row (the delta rule):',
                      'each step waits for the one before it',
                      'cheap on a GPU (3.4 of 52.7 ms), but one long dependent',
                      'chain per head on Inferentia (72 of 152 ms in J8)'], 14, INK)
    # right: attention, one dense product
    ox, oy, n, c = 600, 100, 7, 22
    s.text(560, 82, 'attention (M1, all 24 layers)', 16, ORANGE)
    for i in range(n):
        for j in range(i + 1):
            s.rect(ox + j * c, oy + i * c, c, c, stroke=ORANGE, fill=ORANGE_F, gap=5, sw=1.0)
    s.text(ox + n * c / 2, oy + n * c + 22, 'every row scores every earlier row', 13, GRAY, 'middle')
    s.lines(560, 300, ['all rows at once: dense matrix multiplies,',
                       'which systolic arrays like Inferentia prefer',
                       "GDN's gates become additive attention biases",
                       'exact recall of any earlier row'], 14, INK)
    s.text(40, 420, 'A10G: 56.5 against 57.1 ms at 1,000 tokens.  inf2: the cheapest shape up to 256 tokens.', 15, PURPLE)
    s.save('d22_m1_mixers')


if __name__ == '__main__':
    d22_m1_mixers(); print('drew d22_m1_mixers')
