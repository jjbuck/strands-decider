"""sketch.py: hand-drawn-style SVG diagrams (Excalidraw look) for docs/IDEAS_EXPLORED.md.

Text is drawn as glyph outlines from the Virgil font (OFL, fonts/Virgil.ttf), so the SVGs render identically everywhere
without the font installed. Lines are slightly bowed double strokes; fills are hachure (diagonal hatching).

    from sketch import Sketch
    s = Sketch(900, 400); s.rect(20, 20, 200, 80, stroke=BLUE, fill=BLUE_F); s.text(120, 65, 'hello', anchor='middle'); s.save('x')  # writes docs/figures/ideas/x.png
"""
import math, os, random, subprocess
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

A = os.path.dirname(os.path.abspath(__file__))
FONT = TTFont(os.path.join(A, 'fonts', 'Virgil.ttf'))
GS = FONT.getGlyphSet(); CMAP = FONT.getBestCmap(); UPM = FONT['head'].unitsPerEm; HMTX = FONT['hmtx']
# fallback for symbols Virgil lacks (≈, ᵀ, Δ, →, ≤, ≥, √): Comic Sans MS if present, else a plain substitute
_FB = '/System/Library/Fonts/Supplemental/Comic Sans MS.ttf'
if os.path.exists(_FB):
    FB = TTFont(_FB); FB_GS = FB.getGlyphSet(); FB_CMAP = FB.getBestCmap(); FB_UPM = FB['head'].unitsPerEm; FB_HMTX = FB['hmtx']
else:
    FB = None
SUBST = {'ᵀ': 'T', '≈': '~', 'Δ': 'd', '→': '->', '≤': '<=', '≥': '>=', '√': 'sqrt'}


def _glyph(ch):
    """(glyphset, glyph name, advance scale-free, units per em) for one character"""
    if ch == 'ᵀ':
        ch = 'T'  # drawn smaller and raised by text()
    if ord(ch) in CMAP:
        g = CMAP[ord(ch)]; return GS, g, HMTX[g][0], UPM
    if FB is not None and ord(ch) in FB_CMAP:
        g = FB_CMAP[ord(ch)]; return FB_GS, g, FB_HMTX[g][0], FB_UPM
    g = CMAP.get(ord('?')); return GS, g, HMTX[g][0], UPM

INK = '#1e1e1e'; GRAY = '#868e96'; GRAY_F = '#dee2e6'
PURPLE = '#6741d9'; PURPLE_F = '#d0bfff'
ORANGE = '#f08c00'; ORANGE_F = '#ffd8a8'
GREEN = '#2f9e44'; GREEN_F = '#b2f2bb'
BLUE = '#1971c2'; BLUE_F = '#a5d8ff'
RED = '#e03131'; RED_F = '#ffc9c9'
TEAL = '#0c8599'; TEAL_F = '#99e9f2'
PAPER = '#fffefa'


class Sketch:
    def __init__(self, W, H, seed=7):
        self.W, self.H = W, H
        self.r = random.Random(seed)
        self.parts = [f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

    # ---------------------------------------------------------------- primitives
    def _j(self, a=1.2):
        return self.r.uniform(-a, a)

    def _stroke(self, x1, y1, x2, y2, color, w, rough):
        L = math.hypot(x2 - x1, y2 - y1) or 1
        bow = min(L / 60, 2.2) * rough
        nx, ny = -(y2 - y1) / L, (x2 - x1) / L
        mx, my = (x1 + x2) / 2 + nx * self.r.uniform(-bow, bow), (y1 + y2) / 2 + ny * self.r.uniform(-bow, bow)
        a = 0.9 * rough
        return (f'<path d="M{x1 + self._j(a):.1f},{y1 + self._j(a):.1f} Q{mx:.1f},{my:.1f} {x2 + self._j(a):.1f},{y2 + self._j(a):.1f}" '
                f'fill="none" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>')

    def line(self, x1, y1, x2, y2, color=INK, w=1.8, rough=1.0, dash=None, double=True):
        if dash:  # dashed: draw short rough segments
            L = math.hypot(x2 - x1, y2 - y1); n = max(1, int(L / (dash * 2)))
            for k in range(n):
                t0, t1 = (2 * k) * dash / L, min((2 * k + 1) * dash / L, 1)
                self.parts.append(self._stroke(x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0, x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1, color, w, rough * 0.5))
            return
        self.parts.append(self._stroke(x1, y1, x2, y2, color, w, rough))
        if double:
            self.parts.append(self._stroke(x1, y1, x2, y2, color, w * 0.6, rough))

    def hatch(self, x, y, w, h, color, gap=7, angle=-41, width=1.1):
        """hachure inside an axis-aligned rectangle"""
        t = math.tan(math.radians(angle))
        segs = []
        k0 = -h - w
        c = k0
        while c < w + h:
            # line: y' = (x' - c) * t  in local coords, clip to [0,w]x[0,h]
            pts = []
            for xx in (0, w):
                yy = (xx - c) * t
                if 0 <= yy <= h: pts.append((xx, yy))
            if t != 0:
                for yy in (0, h):
                    xx = yy / t + c
                    if 0 <= xx <= w: pts.append((xx, yy))
            if len(pts) >= 2:
                (ax, ay), (bx, by) = pts[0], pts[-1]
                if abs(ax - bx) + abs(ay - by) > 2:
                    segs.append((x + ax, y + ay, x + bx, y + by))
            c += gap / abs(math.sin(math.radians(angle)))
        for ax, ay, bx, by in segs:
            self.parts.append(self._stroke(ax, ay, bx, by, color, width, 0.5))

    def rect(self, x, y, w, h, stroke=INK, fill=None, sw=1.8, gap=7, solid=None, rough=1.0, dash=None):
        if solid:
            self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{solid}" rx="2"/>')
        if fill:
            self.hatch(x + 2, y + 2, w - 4, h - 4, fill, gap=gap)
        for (a, b, c, d) in ((x, y, x + w, y), (x + w, y, x + w, y + h), (x + w, y + h, x, y + h), (x, y + h, x, y)):
            self.line(a, b, c, d, stroke, sw, rough, dash=dash)

    def ellipse(self, cx, cy, rx, ry, stroke=INK, sw=1.8, fill=None):
        if fill:
            self.parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="0.5"/>')
        for k in range(2):
            pts = []
            for i in range(0, 37):
                a = 2 * math.pi * i / 36 + k * 0.2
                rr = 1 + self.r.uniform(-0.03, 0.03)
                pts.append((cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a)))
            d = 'M' + ' L'.join(f'{px:.1f},{py:.1f}' for px, py in pts)
            self.parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw * (1 if k == 0 else 0.6)}" stroke-linecap="round"/>')

    def arrow(self, x1, y1, x2, y2, color=INK, w=1.8, head=11, curve=0.0, dash=None):
        if curve:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            L = math.hypot(x2 - x1, y2 - y1) or 1
            cx, cy = mx - (y2 - y1) / L * curve, my + (x2 - x1) / L * curve
            for k in range(2):
                self.parts.append(f'<path d="M{x1:.1f},{y1:.1f} Q{cx + self._j(2):.1f},{cy + self._j(2):.1f} {x2:.1f},{y2:.1f}" fill="none" '
                                  f'stroke="{color}" stroke-width="{w * (1 if k == 0 else 0.6)}" stroke-linecap="round"'
                                  + (f' stroke-dasharray="{dash} {dash}"' if dash else '') + '/>')
            ang = math.atan2(y2 - cy, x2 - cx)
        else:
            self.line(x1, y1, x2, y2, color, w, dash=dash)
            ang = math.atan2(y2 - y1, x2 - x1)
        for s in (+1, -1):
            a = ang + math.pi - s * 0.45
            self.line(x2, y2, x2 + head * math.cos(a), y2 + head * math.sin(a), color, w, 0.5)

    def brace(self, x1, y1, x2, y2, color=INK, depth=10, w=1.6):
        """curly brace from (x1,y1) to (x2,y2), bulging to the left of the direction of travel"""
        L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L; nx, ny = uy, -ux
        P = lambda t, d: (x1 + ux * t * L + nx * d, y1 + uy * t * L + ny * d)
        pts = [P(0, 0), P(0.02, depth * 0.6), P(0.25, depth * 0.55), P(0.48, depth * 0.6), P(0.5, depth * 1.2),
               P(0.52, depth * 0.6), P(0.75, depth * 0.55), P(0.98, depth * 0.6), P(1, 0)]
        d = f'M{pts[0][0]:.1f},{pts[0][1]:.1f} ' + ' '.join(f'Q{pts[i][0]:.1f},{pts[i][1]:.1f} {(pts[i][0] + pts[i + 1][0]) / 2:.1f},{(pts[i][1] + pts[i + 1][1]) / 2:.1f}'
                                                             for i in range(1, len(pts) - 1)) + f' L{pts[-1][0]:.1f},{pts[-1][1]:.1f}'
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>')

    # ---------------------------------------------------------------- text as glyph outlines
    def text_width(self, s, size):
        return sum(_glyph(ch)[2] * size / _glyph(ch)[3] for ch in s)

    def text(self, x, y, s, size=16, color=INK, anchor='start', rotate=0, bg=False):
        w = self.text_width(s, size)
        x0 = x - (w if anchor == 'end' else w / 2 if anchor == 'middle' else 0)
        if bg:
            self.parts.append(f'<rect x="{x0 - 5:.1f}" y="{y - size * 0.85:.1f}" width="{w + 10:.1f}" height="{size * 1.2:.1f}" fill="{PAPER}" rx="4"/>')
        pen = SVGPathPen(GS); cx = 0.0
        for ch in s:
            gs, g, adv, upm = _glyph(ch)
            sc = size / upm
            dy = -size * 0.35 if ch == 'ᵀ' else 0
            k = 0.7 if ch == 'ᵀ' else 1.0
            tp = TransformPen(pen, (sc * k, 0, 0, -sc * k, cx, dy))
            gs[g].draw(tp)
            cx += adv * sc * k
        rot = f' rotate({rotate} {x} {y})' if rotate else ''
        tr = f'translate({x0:.1f} {y:.1f})' if not rotate else f'rotate({rotate} {x:.1f} {y:.1f}) translate({x0:.1f} {y:.1f})'
        self.parts.append(f'<path d="{pen.getCommands()}" fill="{color}" transform="{tr}"/>')
        return w

    def lines(self, x, y, rows, size=15, color=INK, anchor='start', lead=1.3, bg=False):
        for i, s in enumerate(rows):
            self.text(x, y + i * size * lead, s, size, color, anchor, bg=bg)

    # ---------------------------------------------------------------- output
    def save(self, name, out=None):
        out = out or os.path.expanduser('~/code/jit-eval/docs/figures/ideas')
        os.makedirs(out, exist_ok=True)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" width="{self.W}" height="{self.H}">'
               + ''.join(self.parts) + '</svg>')
        p = os.path.join(out, name + '.png')  # PNG only: the SVG is piped to rsvg-convert, never written
        subprocess.run(['rsvg-convert', '-z', '2', '-o', p], input=svg.encode('utf-8'), check=True)
        return p
