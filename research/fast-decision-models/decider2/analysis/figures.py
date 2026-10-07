"""figures.py: the post-processing figures, in the style of Brooker's hobson-bidi figures (research/scripts/bidi_figures.py).

    python3 figures.py            # reads results/metrics.json; writes figures/*.svg and a CSV beside each

Static SVG, no script: light and dark themes from CSS custom properties; at most three series colours per panel
(Okabe-Ito, colour and marker shape by category, hobson-v19 a black star; readable in greyscale); every point labelled with its
ID from docs/IDEAS_EXPLORED.md; a native <title> tooltip on every mark.

  fig1_accuracy_brier      accuracy against Brier score, JevBench and REAL-label
  fig2_frontier            accuracy against speedup over hobson at a 1000-token state (A10G, measured curves)
  fig3_latency_length      measured A10G latency curves against state length, four panels, log-log
  fig4_latency_box_model   per-request latency on JevBench and real traffic, modeled from the measured curves
  fig5_latency_box_meas    per-request latency measured end to end (J15: 120 requests; J9: 84 requests)
  fig6_length_scatter      the same measured requests: latency against state length
"""
import csv, json, math, os, statistics, sys
from html import escape

A = os.path.expanduser('~/decider2/analysis')
sys.path.insert(0, A)
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import evalkit as EK
from latency_grid import CURVES, curve_at, speedup
from registry import E, BY_ID

OUT = f'{A}/figures'
DISP = {'hobson': 'hobson-v19'}  # how the baseline is labelled in every figure


# Encoding: colour and shape both follow the ID's category (Okabe-Ito colours, validated for colour-vision deficiency
# with shapes and direct labels as the required secondary encoding); the baseline is a black star (white in dark mode).
# Hollow marks are smaller models (controls, or trained from scratch). Lines of one category differ by dash pattern.
def cat(i):
    # colours: P and B blue, R vermillion, A and M green, HW and E purple (4 validated Okabe-Ito slots; shapes separate the pairs)
    return ('base' if i == 'hobson' else 'H' if i.startswith(('HW', 'E')) or i[0] == 'C' else 'P' if i[0] in 'PB' else 'A' if i[0] in 'AM'
            else i[0] if i[0] == 'R' else 'P')


def shape(i):
    return ('star' if i == 'hobson' else 'diamond' if i.startswith('HW') else 'circle' if i[0] in 'ES' else
            {'P': 'square', 'R': 'up', 'A': 'down', 'B': 'pent', 'M': 'hex', 'C': 'cross'}[i[0]])


CATS = [('hobson', 'hobson-v19, the baseline (bf16)'), ('P1', 'P · precision'), ('B1', 'B · decision-specific numerics/structure'),
        ('R1', 'R · reducing work per token'), ('A1', 'A · architecture'), ('M1', 'M · decision-native layout'),
        ('C1', 'C · combinations'), ('HW1', 'HW · hardware'), ('E1', 'E · data and evaluation')]


def mk(i, cx, cy, r=6, hollow=False, title=None):
    cls = f'mk c{cat(i)}' + (' hollow' if hollow else '')
    sh = shape(i)
    if sh == 'circle':
        el = f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"'
    elif sh == 'square':
        a = 1.7 * r; el = f'<rect class="{cls}" x="{cx - a / 2:.1f}" y="{cy - a / 2:.1f}" width="{a:.1f}" height="{a:.1f}"'
    else:
        if sh in ('up', 'down'):
            R_ = 1.35 * r; sg = 1 if sh == 'up' else -1; off = 0.18 * R_ * sg
            pts = [(cx, cy - sg * R_ + off), (cx + 0.866 * R_, cy + sg * 0.5 * R_ + off), (cx - 0.866 * R_, cy + sg * 0.5 * R_ + off)]
        elif sh == 'diamond':
            d = 1.35 * r; pts = [(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)]
        elif sh == 'cross':
            a, b = 1.45 * r, 0.5 * r
            pts = [(cx - b, cy - a), (cx + b, cy - a), (cx + b, cy - b), (cx + a, cy - b), (cx + a, cy + b), (cx + b, cy + b),
                   (cx + b, cy + a), (cx - b, cy + a), (cx - b, cy + b), (cx - a, cy + b), (cx - a, cy - b), (cx - b, cy - b)]
        elif sh in ('pent', 'hex'):
            n = 5 if sh == 'pent' else 6; R_ = 1.25 * r
            pts = [(cx + R_ * math.sin(2 * math.pi * k / n), cy - R_ * math.cos(2 * math.pi * k / n)) for k in range(n)]
        else:  # star
            ro, ri = 1.7 * r, 0.72 * r
            pts = [(cx + (ro if k % 2 == 0 else ri) * math.sin(math.pi * k / 5), cy - (ro if k % 2 == 0 else ri) * math.cos(math.pi * k / 5)) for k in range(10)]
        el = f'<polygon class="{cls}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"'
    return el + (f'><title>{escape(title)}</title></{el.split()[0][1:]}>' if title else '/>')


def legend(parts, x, y, items, wrap=1180):
    """items: [(id or None, text, hollow)]; draws marker + text in rows"""
    kx = x
    for i, text, hollow in items:
        w = 24 + 6.4 * len(text) + 22
        if kx + w > x + wrap: kx = x; y += 20
        if i: parts.append(mk(i, kx + 6, y - 4, 5.5, hollow))
        parts.append(f'<text class="sub" x="{kx + 18}" y="{y}">{escape(text)}</text>')
        kx += w
    return y
R = os.path.expanduser('~/decider2')
M = json.load(open(f'{A}/results/metrics.json'))

STYLE = """
<style>
  .viz { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
         --base:#000000; --base-fill:rgba(0,0,0,0.10);
         --cP:#0072b2; --cP-fill:rgba(0,114,178,0.14); --cR:#d55e00; --cR-fill:rgba(213,94,0,0.14);
         --cA:#009e73; --cA-fill:rgba(0,158,115,0.14); --cH:#cc79a7; --cH-fill:rgba(204,121,167,0.18);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  @media (prefers-color-scheme: dark) {
    .viz { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
           --base:#ffffff; --base-fill:rgba(255,255,255,0.12);
           --cP:#0072b2; --cP-fill:rgba(0,114,178,0.25); --cR:#d55e00; --cR-fill:rgba(213,94,0,0.25);
           --cA:#009e73; --cA-fill:rgba(0,158,115,0.25); --cH:#c776a3; --cH-fill:rgba(199,118,163,0.25); }
  }
  .bg { fill: var(--surface); }
  .title { fill: var(--ink); font-size: 16px; font-weight: 600; }
  .sub { fill: var(--ink2); font-size: 12px; }
  .panel { fill: var(--ink); font-size: 13px; font-weight: 600; }
  .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  .name { fill: var(--ink); font-size: 11px; font-weight: 600; }
  .name.base { font-weight: 800; }
  .torso { fill: var(--ink2); font-size: 10px; }
  .val { fill: var(--ink2); font-size: 10.5px; font-variant-numeric: tabular-nums; }
  .note { fill: var(--muted); font-size: 11px; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .bar { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 3; }
  .ring { fill: none; stroke: var(--base); stroke-width: 1.4; }
  .mk { stroke: var(--surface); stroke-width: 1.2; }
  .mk.cP { fill: var(--cP); } .mk.cR { fill: var(--cR); } .mk.cA { fill: var(--cA); } .mk.cH { fill: var(--cH); } .mk.cbase { fill: var(--base); }
  .mk.hollow { fill: var(--surface); stroke-width: 1.8; }
  .mk.hollow.cP { stroke: var(--cP); } .mk.hollow.cR { stroke: var(--cR); } .mk.hollow.cA { stroke: var(--cA); } .mk.hollow.cH { stroke: var(--cH); } .mk.hollow.cbase { stroke: var(--base); }
  .line { fill: none; stroke-width: 2; }
  .line.cP { stroke: var(--cP); } .line.cR { stroke: var(--cR); } .line.cA { stroke: var(--cA); } .line.cH { stroke: var(--cH); }
  .line.cbase { stroke: var(--base); stroke-width: 2.6; }
  .d1 { stroke-dasharray: 7 4; } .d2 { stroke-dasharray: 2 3; }
  .box { stroke-width: 1.6; }
  .box.cP { fill: var(--cP-fill); stroke: var(--cP); } .box.cR { fill: var(--cR-fill); stroke: var(--cR); }
  .box.cA { fill: var(--cA-fill); stroke: var(--cA); } .box.cH { fill: var(--cH-fill); stroke: var(--cH); }
  .box.cbase { fill: var(--base-fill); stroke: var(--base); }
  .whisker { stroke-width: 1.5; } .whisker.cP { stroke: var(--cP); } .whisker.cR { stroke: var(--cR); }
  .whisker.cA { stroke: var(--cA); } .whisker.cH { stroke: var(--cH); } .whisker.cbase { stroke: var(--base); }
  .median { stroke: var(--ink); stroke-width: 2.2; }
  .outlier { fill-opacity: 0.55; stroke: none; }
  .outlier.cP { fill: var(--cP); } .outlier.cR { fill: var(--cR); } .outlier.cA { fill: var(--cA); } .outlier.cH { fill: var(--cH); } .outlier.cbase { fill: var(--base); }
</style>
"""


# ---------------------------------------------------------------- helpers
def svg_open(W, H, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" class="viz" role="img" aria-labelledby="t d">',
            STYLE, f'<title id="t">{escape(title)}</title>', f'<desc id="d">{escape(desc)}</desc>',
            f'<rect class="bg" width="{W}" height="{H}"/>']


def write(name, parts, rows, header):
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/{name}.svg', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(parts + ['</svg>']) + '\n')
    with open(f'{OUT}/{name}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)


def metric_fn(e, suite, key):
    return metric(e, suite, key)


def metric(e, suite, key):
    if e.get('run'):
        return M.get(e['run'], {}).get(suite, {}).get(key)
    return e.get('manual', {}).get(suite, {}).get(key)


def fmt_ms(x):
    return f'{x:,.0f}' if x >= 10 else f'{x:.1f}'


def nice_lin(lo, hi, n=6):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n: step *= m; break
    a = math.floor(lo / step) * step; b = math.ceil(hi / step) * step
    k = round((b - a) / step)
    return a, b, [round(a + i * step, 10) for i in range(k + 1)]


def place(points, box):
    """greedy label placement: points = [(x, y, text, r)]; returns [(x, y, anchor)] for each label"""
    cands = lambda r: [(r + 3, 4, 'start'), (-(r + 3), 4, 'end'), (0, -(r + 5), 'middle'), (0, r + 13, 'middle'),
                       (r + 2, -(r + 3), 'start'), (r + 2, r + 11, 'start'), (-(r + 2), -(r + 3), 'end'), (-(r + 2), r + 11, 'end'),
                       (r + 14, 4, 'start'), (-(r + 14), 4, 'end'), (0, -(r + 17), 'middle'), (0, r + 25, 'middle')]
    placed, out = [], []
    obst = [(x - r, y - r, x + r, y + r) for x, y, _, r in points]

    def bb(x, y, dx, dy, anc, t):
        w = 6.4 * len(t) + 2
        x0 = x + dx - (w if anc == 'end' else w / 2 if anc == 'middle' else 0)
        return (x0, y + dy - 9, x0 + w, y + dy + 2)

    def ov(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    for x, y, t, r in points:
        best = None
        for i, (dx, dy, anc) in enumerate(cands(r)):
            b = bb(x, y, dx, dy, anc, t)
            s = sum(ov(b, p) for p in placed) * 3 + sum(ov(b, o) for o in obst) + i * 0.5
            s += 1e4 * (b[0] < box[0]) + 1e4 * (b[2] > box[2]) + 1e4 * (b[1] < box[1]) + 1e4 * (b[3] > box[3])
            if best is None or s < best[0]: best = (s, dx, dy, anc, b)
        placed.append(best[4]); out.append((x + best[1], y + best[2], best[3]))
    return out


def five(values):
    v = sorted(values)
    q = statistics.quantiles(v, n=4, method='inclusive')
    iqr = q[2] - q[0]; lo_f, hi_f = q[0] - 1.5 * iqr, q[2] + 1.5 * iqr
    inside = [x for x in v if lo_f <= x <= hi_f]
    return dict(n=len(v), min=v[0], q1=q[0], median=q[1], q3=q[2], max=v[-1], p95=v[int(0.95 * (len(v) - 1))],
                lo=inside[0], hi=inside[-1], outliers=[x for x in v if x < lo_f or x > hi_f])


def entry_speed(e):
    """(speedup over same-harness hobson at a 1000-token state, number of questions, note)"""
    ml = e.get('manual_lat')
    if ml: return ml['speedup'], ml.get('q', 1), ml.get('note', '')
    if e.get('lat'):
        c = CURVES[e['lat']]
        return (1.0 if e['id'] == 'hobson' else speedup(e['lat'], 1000)), c['q'], ''
    return None, 1, ''


# ---------------------------------------------------------------- figs 1-2: scatter, full view and zoom
def draw_panel(parts, x0, y0, pw, ph, xr, yr, logx, pts, refs, fmt, pname, zoom=None, label_ok=lambda x, y: True, ylab='Accuracy', xlab='',
               xtickfmt=None, xticklist=None, frontier=None):
    """pts: [(x, y, entry, questions)]; returns CSV rows"""
    if logx:
        lx0, lx1 = math.log10(xr[0]), math.log10(xr[1])
        X = lambda v: x0 + pw * (math.log10(v) - lx0) / (lx1 - lx0)
        xticks = [t for t in (xticklist or (0.1, 0.2, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10)) if xr[0] <= t <= xr[1]]
    else:
        a, b, xticks = nice_lin(*xr, n=5); xr = (a, b)
        X = lambda v: x0 + pw * (v - xr[0]) / (xr[1] - xr[0])
    ya, yb, yticks = nice_lin(*yr, n=6)
    Y = lambda v: y0 + ph * (1 - (v - ya) / (yb - ya))
    parts.append(f'<text class="panel" x="{x0}" y="{y0 - 14}">{escape(pname)}</text>')
    for t in yticks:
        parts.append(f'<line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                     f'<text class="tick" x="{x0 - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{(f"{t:.2f}" if abs(t * 100 - round(t * 100)) < 1e-6 else f"{t:.3f}")}</text>')
    for t in xticks:
        parts.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{y0}" y2="{y0 + ph}"/>'
                     f'<text class="tick" x="{X(t):.1f}" y="{y0 + ph + 16}" text-anchor="middle">{(xtickfmt(t) if xtickfmt else f"{t:g}x" if logx else f"{t:.2f}")}</text>')
    for yv, lab, side in refs:
        if ya <= yv <= yb:
            parts.append(f'<line class="bar" x1="{x0}" x2="{x0 + pw}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}"/>'
                         f'<text class="note" x="{(x0 + pw - 4) if side[0] == "r" else (x0 + 4)}" y="{Y(yv) + (13 if side.endswith("b") else -4):.1f}" '
                         f'text-anchor="{"end" if side[0] == "r" else "start"}">{escape(lab)}</text>')
    parts.append(f'<line class="axis" x1="{x0}" x2="{x0 + pw}" y1="{y0 + ph}" y2="{y0 + ph}"/>'
                 f'<line class="axis" x1="{x0}" x2="{x0}" y1="{y0}" y2="{y0 + ph}"/>'
                 f'<text class="sub" x="{x0 + pw / 2}" y="{y0 + ph + 38}" text-anchor="middle">{escape(xlab)}</text>'
                 f'<text class="sub" transform="translate({x0 - 52} {y0 + ph / 2}) rotate(-90)" text-anchor="middle">{ylab}</text>')
    if zoom:
        zx0, zx1, zy0, zy1 = X(zoom[0]), X(zoom[1]), Y(zoom[3]), Y(zoom[2])
        parts.append(f'<rect x="{zx0:.1f}" y="{zy0:.1f}" width="{zx1 - zx0:.1f}" height="{zy1 - zy0:.1f}" fill="none" class="bar"/>'
                     f'<text class="note" x="{zx1:.1f}" y="{zy0 - 4:.1f}" text-anchor="end">zoomed at right</text>')
    if frontier:
        fp = [(X(x), Y(y)) for x, y in frontier if xr[0] <= x <= xr[1] and ya <= y <= yb]
        if fp:
            d = f'M{fp[0][0]:.1f},{fp[0][1]:.1f}' + ''.join(f' H{x:.1f} V{y:.1f}' for x, y in fp[1:])
            parts.append(f'<path class="line cbase d1" d="{d}" style="stroke-width:1.6;opacity:.55"/>')
    inside = [p for p in pts if xr[0] <= p[0] <= xr[1] and ya <= p[1] <= yb]
    inside.sort(key=lambda p: p[2]['id'] != 'hobson')  # the baseline's label is placed first, so it gets the clearest spot
    lab_of = lambda e, q: DISP.get(e['id'], e['id']) + (f' ({q}q)' if q > 1 else '') + (' (est.)' if (e.get('manual_lat') or {}).get('est') else '')
    pp = [(X(x), Y(y), lab_of(e, q), 15 if e['id'] == 'hobson' else 8) for x, y, e, q in inside]
    labs = place([p for p, q in zip(pp, inside) if label_ok(q[0], q[1])], (x0 + 2, y0 + 2, x0 + pw - 2, y0 + ph - 2))
    li = iter(labs)
    rows, marks = [], []
    for (x, y, e, q), (cx, cy, _, _) in zip(inside, pp):
        ref = e['id'] == 'hobson'
        tip = escape(f"{e['id']}: {e['name']} — {pname}: accuracy {y:.3f}, {fmt(x)}" + (f" ({e['note']})" if e.get('note') else ''))
        mark = (f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="15"/>' + mk('hobson', cx, cy, 7) if ref else
                mk(e['id'], cx, cy, 6, hollow=e['size'] == 'small'))
        lab = ''
        if label_ok(x, y):
            lx, ly, anc = next(li)
            lab = f'<text class="name{" base" if ref else ""}" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anc}">{escape(lab_of(e, q))}</text>'
        marks.append(f'<g><title>{tip}</title><circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="transparent"/>{mark}{lab}</g>')
        rows.append([pname, e['id'], e['name'], e['size'], round(x, 4), round(y, 4)])
    parts += marks[1:] + marks[:1] if inside and inside[0][2]['id'] == 'hobson' else marks  # baseline drawn on top
    return rows


def scatter_grid(name, title, sub, suites, xlab, notes, legend_hollow=False, ylabs=None, **kw):
    """suites: [(suite title, pts, fmt, full (xr, yr), zoom (xr, yr), refs, logx)]: full view left, zoom right"""
    W, pw, ph = 1240, 500, 360
    cols = (92, 92 + pw + 110)
    first = 172; step = ph + 130
    H = first + step * (len(suites) - 1) + ph + 76 + 15 * len(notes)
    parts = svg_open(W, H, title, sub)
    parts += [f'<text class="title" x="24" y="30">{escape(title)}</text>', f'<text class="sub" x="24" y="50">{escape(sub)}</text>']
    y = legend(parts, 24, 78, [(i, t, False) for i, t in CATS] + [('P1', 'hollow: a smaller model (control, or trained from scratch)', True)])
    parts.append(f'<text class="sub" x="24" y="{y + 22}">Every point is labelled with its ID in docs/IDEAS_EXPLORED.md; hover for names and values.'
                 + (' "(4q)", "(15q)": speedup measured with that many questions.' if legend_hollow else '') + '</text>')
    rows = []
    for i, (sname, pts, fmt, full, zoom, refs, logx) in enumerate(suites):
        y0 = first + i * step
        zin = lambda x, y, z=zoom: z[0][0] <= x <= z[0][1] and z[1][0] <= y <= z[1][1]
        k2 = dict(kw); fr = k2.pop('frontiers', None); fr = fr[i] if fr else None
        yl = (ylabs or ['Accuracy'] * len(suites))[i]
        rows += draw_panel(parts, cols[0], y0, pw, ph, full[0], full[1], logx, pts, refs, fmt, f'{sname} · all configurations',
                           zoom=(zoom[0][0], zoom[0][1], zoom[1][0], zoom[1][1]), label_ok=lambda x, y: not zin(x, y), xlab=xlab,
                           ylab=yl, frontier=fr, **k2)
        draw_panel(parts, cols[1], y0, pw, ph, zoom[0], zoom[1], logx, [p for p in pts if zin(p[0], p[1])], refs, fmt,
                   f'{sname} · zoom on the dashed box', xlab=xlab, ylab=yl, frontier=fr, **k2)
    for j, line in enumerate(notes):
        parts.append(f'<text class="note" x="24" y="{H - 16 - 15 * (len(notes) - 1 - j)}">{escape(line)}</text>')
    return parts, rows


def fig1():
    suites = []
    for suite, how, full, zoom in (('JB-all', 'JevBench, 231 tasks', ((0.32, 0.70), (0.25, 0.80)), ((0.325, 0.42), (0.66, 0.76))),
                                   ('REAL-label', '400 real questions, Opus labels', ((0.32, 0.50), (0.62, 0.82)), ((0.32, 0.385), (0.745, 0.80)))):
        pts = [(metric(e, suite, 'brier'), metric(e, suite, 'acc'), e, 1) for e in E
               if e.get('plot_acc', True) and metric(e, suite, 'brier') is not None and metric(e, suite, 'acc') is not None]
        suites.append((f'{suite} ({how})', pts, lambda v: f'Brier {v:.3f}', full, zoom,
                       [(metric(BY_ID['hobson'], suite, 'acc'), 'hobson-v19', 'r')], False))
    parts, rows = scatter_grid('fig1', 'Accuracy against Brier score',
                               'Better is up and to the left: more decisions right, and probabilities closer to the truth.',
                               suites, 'Brier score (lower is better)',
                               ['Brier as JevBench defines it: the sum over the options of (p − y)², averaged over decisions.',
                                'P9 has no prediction file; its point uses the scores in its report. Configurations without a Brier score (R6, R10) are not shown.'])
    write('fig1_accuracy_brier', parts, rows, ['panel', 'id', 'name', 'size', 'brier', 'accuracy'])


def fig2():
    suites = []
    for suite, how, refs, full, zoom in (
            ('JB-all', 'JevBench, 231 tasks', [(.723, 'hobson-v19 .723', 'r')], ((0.08, 12), (0.25, 0.80)), ((0.6, 3.2), (0.66, 0.76))),
            ('REAL-label', '400 real questions, Opus labels', [(.785, 'hobson-v19 .785', 'r'), (.78, 'bar .78', 'rb')], ((0.08, 12), (0.62, 0.82)), ((0.6, 3.2), (0.74, 0.80)))):
        pts = []
        for e in E:
            s, q, _ = entry_speed(e)
            a = metric(e, suite, 'acc')
            if s and a is not None and e.get('plot_acc', True): pts.append((s, a, e, q))
        suites.append((f'{suite} ({how})', pts, lambda v: f'{v:.2f}x the speed of hobson', full, zoom, refs, True))
    parts, rows = scatter_grid('fig2', 'Accuracy against speedup',
                               'Speedup over hobson bf16 at a 1000-token state, each measured on the A10G in the same harness as its own hobson baseline. Up and to the right is better.',
                               suites, 'Speedup over hobson at a 1000-token state (log scale)',
                               ['A label ending "(4q)" or "(15q)" is a speedup measured with that many questions, against hobson-v19 with the same questions.',
                                'R1 exits every decision at layer 16 of the W8A8 model. A11 (114M, from scratch) is compared with full hobson. Same-shape fine-tunes (E1, E2, A10) sit at 1x.',
                                '"(est.)": end-to-end latency estimated from measured GEMM times. M2 points use 55% of the state compiled (A10G bf16).'],
                               legend_hollow=True)
    write('fig2_frontier', parts, rows, ['panel', 'id', 'name', 'size', 'speedup_at_1000', 'accuracy'])


def pareto(pts, maximize):
    """non-dominated (lower latency, better metric) points, sorted by latency"""
    out, best = [], None
    for x, y in sorted(pts):
        if best is None or (y > best if maximize else y < best):
            out.append((x, y)); best = y
    return out


def fig7():
    """Pareto: JevBench accuracy and Brier against latency at a 1000-token state (A10G, one question)."""
    base = 57.1  # hobson bf16 at 1000 tokens, J15 harness; every entry's same-harness speedup is applied to it
    suites, fronts, rows_info = [], [], []
    for metric, maximize, yr, zy in (('acc', True, (0.25, 0.80), (0.63, 0.76)), ('brier', False, (0.30, 0.70), (0.325, 0.45))):
        pts = []
        for e in E:
            s, q, _ = entry_speed(e)
            v = metric_v = metric_fn(e, 'JB-all', metric)
            if not s or v is None or q > 1 or not e.get('plot_acc', True): continue
            pts.append((base / s, v, e, 1))
        fronts.append(pareto([(x, y) for x, y, e, _ in pts if e['size'] != 'small'], maximize))
        refs = [(metric_fn(BY_ID['hobson'], 'JB-all', metric), 'hobson-v19', 'r')] + ([(0.693, 'relaxed bar .693', 'rb')] if maximize else [])
        name = 'JB-all accuracy (JevBench, 231 tasks)' if maximize else 'JB-all Brier score (lower is better)'
        suites.append((name, pts, lambda v: f'{v:.1f} ms at 1000 tokens', ((4, 700), yr), ((15, 70), zy), refs, True))
    parts, rows = scatter_grid('fig7', 'Accuracy and calibration against latency: the Pareto frontier',
                               'Latency at a 1000-token state, one question, A10G: each configuration\'s same-harness speedup applied to hobson-v19\'s 57.1 ms. '
                               'Dashed step line: the Pareto frontier among 2B-class configurations.',
                               suites, 'Latency at a 1000-token state, ms (log scale)',
                               ['Up-left is better for accuracy; down-left is better for Brier. Multi-question measurements (A3, A4, A5) are left out because their latency is not one-question.',
                                '"(est.)": latency from measured GEMM or segment times, not a single end-to-end timing. Smaller models (hollow) are shown but not part of the frontier. R1a (exit at layer 8) is fast but far below the bar.',
                                'Brier as JevBench defines it: the sum over options of (p − y)², averaged over tasks.'],
                               ylabs=['Accuracy', 'Brier score'], xtickfmt=lambda t: f'{t:g}', xticklist=(5, 10, 15, 20, 30, 50, 70, 100, 200, 500),
                               frontiers=fronts)
    write('fig7_pareto_latency', parts, rows, ['panel', 'id', 'name', 'size', 'latency_ms_at_1000', 'value'])


# ---------------------------------------------------------------- fig 3: latency against state length
PANELS3 = [('Precision and early exit', ['P1', 'P2', 'R1']),
           ('Fewer rows or layers', ['R2a', 'A6', 'A7']),
           ('Other architectures', ['A1', 'A2', 'P9']),
           ('Other hardware, smaller models', ['HW3', 'R10a', 'A11']),
           ('4-bit mixes (Oct 6-7)', ['P10', 'P11']),
           ('Decision-native layouts (Oct 6-7)', ['M2x', 'M2y', 'M2z'])]


# curves for IDs whose registry entry has no single curve (label, curve key)
LAT3 = {'M2x': ('segment-isolated state, depth 12, nothing compiled', 'm2/N_k12_c0'),
        'M2y': ('constants isolated, depth 12, nothing compiled', 'm2/C_k12_c0'),
        'M2z': ('segment-isolated state, depth 8, nothing compiled', 'm2/N_k8_c0')}


def fig3():
    W, H = 900, 1337
    pw, ph = 340, 240
    origins = [(80, 182), (510, 182), (80, 584), (510, 584), (80, 986), (510, 986)]
    xlo, xhi, ylo, yhi = 16, 9000, 1, 2500
    X = lambda v, x0: x0 + pw * (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo))
    Y = lambda v, y0: y0 + ph * (1 - (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)))
    parts = svg_open(W, H, 'Latency against state length', 'Measured A10G latency curves, one question, log-log, four panels.')
    parts += ['<text class="title" x="24" y="30">Latency against state length</text>',
              '<text class="sub" x="24" y="50">Measured on the A10G at exact lengths, batch 1, one question, median of ≥12 warm runs. '
              'Log-log; all panels on one scale.</text>',
              '<text class="sub" x="24" y="68">Black line with stars: hobson-v19 (bf16), the baseline, in every panel. Colour and marker follow the category of each ID.</text>']
    rows = []
    ref = CURVES['j15/bf16']
    for (pname, ids), (x0, y0) in zip(PANELS3, origins):
        parts.append(f'<text class="panel" x="{x0}" y="{y0 - 72}">{escape(pname)}</text>')
        for t in (1, 3, 10, 30, 100, 300, 1000):
            parts.append(f'<line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{Y(t, y0):.1f}" y2="{Y(t, y0):.1f}"/>'
                         f'<text class="tick" x="{x0 - 6}" y="{Y(t, y0) + 4:.1f}" text-anchor="end">{t:,}</text>')
        for t in (16, 64, 256, 1000, 4000):
            parts.append(f'<line class="grid" x1="{X(t, x0):.1f}" x2="{X(t, x0):.1f}" y1="{y0}" y2="{y0 + ph}"/>'
                         f'<text class="tick" x="{X(t, x0):.1f}" y="{y0 + ph + 15}" text-anchor="middle">{t:,}</text>')
        parts.append(f'<line class="axis" x1="{x0}" x2="{x0 + pw}" y1="{y0 + ph}" y2="{y0 + ph}"/>'
                     f'<text class="tick" x="{x0 - 6}" y="{y0 - 8}" text-anchor="end">ms</text>'
                     f'<text class="sub" x="{x0 + pw / 2}" y="{y0 + ph + 34}" text-anchor="middle">State tokens</text>')
        series = [('hobson', 'hobson-v19, bf16: the baseline', ref)] + [(i_, LAT3.get(i_, (BY_ID.get(i_, {}).get('name'), None))[0] or BY_ID[i_]['name'],
                                                                         CURVES[LAT3.get(i_, (None, None))[1] or BY_ID[i_]['lat']]) for i_ in ids]
        ky = y0 - 56
        ends, seen = [], {}
        for sid, nm, c in series:
            k = cat(sid); dash = ['', ' d1', ' d2'][seen.get(k, 0)]; seen[k] = seen.get(k, 0) + 1
            hollow = sid != 'hobson' and BY_ID.get(sid, {}).get('size') == 'small'
            pts = sorted(c['pts'].items())
            d = ' '.join(f'{"M" if j == 0 else "L"}{X(t, x0):.1f},{Y(v, y0):.1f}' for j, (t, v) in enumerate(pts))
            parts.append(f'<path class="line c{k}{dash}" d="{d}"/>')
            for t, v in pts:
                parts.append(mk(sid, X(t, x0), Y(v, y0), 3.4 if sid != 'hobson' else 3.6, hollow,
                                f"{DISP.get(sid, sid)} · {nm}: {t:,} state tokens, {fmt_ms(v)} ms"))
                rows.append([pname, sid, nm, t, v, c['src']])
            ends.append((X(pts[-1][0], x0), Y(pts[-1][1], y0), DISP.get(sid, sid), 4))
            lab = f'{DISP.get(sid, sid) if sid == "hobson" else sid} · {nm}' if sid != 'hobson' else nm
            if len(lab) > 58: lab = lab[:57] + '…'
            parts.append(f'<line class="line c{k}{dash}" x1="{x0}" x2="{x0 + 22}" y1="{ky - 4}" y2="{ky - 4}"/>'
                         + mk(sid, x0 + 11, ky - 4, 3.2, hollow) + f'<text class="torso" x="{x0 + 28}" y="{ky}">{escape(lab)}</text>')
            ky += 13
        for (lx, ly, anc), (ex, ey, sid, _) in zip(place(ends, (x0, y0, x0 + pw + 60, y0 + ph)), ends):
            parts.append(f'<text class="name{" base" if sid == "hobson-v19" else ""}" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anc}">{escape(sid)}</text>')
    parts.append('<text class="note" x="24" y="1286">Each curve comes from its own agent\'s harness; hobson-v19 measured in those harnesses is within about 2 ms of the black line '
                 '(J1\'s is 2 ms faster at 64 tokens).</text>'
                 '<text class="note" x="24" y="1301">R1 exits every decision at layer 16. A6 compiles 55% of the state. R10a and A11 are smaller models, shown for scale.</text>'
                 '<text class="note" x="24" y="1316">HW3 runs the exact hobson function on one Inferentia2 NeuronCore (pipelined GDN kernel). M2x, M2y, M2z: the M2 layouts with nothing compiled (M1, all-attention, matches hobson within 1 ms to 1,000 tokens). IDs: docs/IDEAS_EXPLORED.md.</text>')
    write('fig3_latency_length', parts, rows, ['panel', 'id', 'name', 'state_tokens', 'ms', 'source'])



# ---------------------------------------------------------------- per-request token counts
def requests():
    """{suite: [(item id, state tokens, first-question tokens, 64k super state tokens, super question tokens)]}"""
    nt = json.load(open(f'{R}/j7/preds/C.json.ntok.json'))
    sup = json.load(open(f'{R}/j7/preds/TL64k.json.ntok.json')) if os.path.exists(f'{R}/j7/preds/TL64k.json.ntok.json') else \
        json.load(open(f'{R}/j7/preds/T64k.json.ntok.json'))
    out = {}
    for s in ('JB-all', 'REAL-agree', 'LONG'):
        out[s] = [(it['id'], nt[it['id']]['s_orig'], nt[it['id']]['q_orig'][0], sup[it['id']]['s'], sup[it['id']]['q'][0])
                  for it in EK.load_suite(s) if it['id'] in nt and it['id'] in sup]
    return out


BOX4 = [('hobson', 'j15/bf16'), ('P1', 'j15/b8'), ('P2', 'j15/w4a4'), ('R1', 'j15/exit16'), ('R2a', 'j3/DT-A8'),
        ('A7', 'super'), ('A1', 'j1/encoder')]


def modeled(key, s, q, ss, sq):
    if key == 'super':  # hobson's own curve at the super-token row count
        c = CURVES['j7/hobson']; return curve_at('j7/hobson', max(1, ss + sq - c['qref']))
    return curve_at(key, max(1, s + q - CURVES[key]['qref']))


def box_svg(name, title, subs, groups, note, scale_lo=None, scale_hi=None):
    """groups: [(panel title, how, [(id, label1, label2, values)])]; one log scale for all panels"""
    W, left, panel_w = 780, 64, 690
    first_top, plot_h, step = 122, 230, 340
    H = first_top + step * (len(groups) - 1) + plot_h + 74 + 15 * len(note)
    st = [[(i, l1, l2, five(v)) for i, l1, l2, v in g[2]] for g in groups]
    lo = scale_lo or min(s['min'] for g in st for *_, s in g) * 0.85
    hi = scale_hi or max(s['max'] for g in st for *_, s in g) * 1.15
    ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if lo <= t <= hi]
    parts = svg_open(W, H, title, ' '.join(subs))
    parts += [f'<text class="title" x="24" y="30">{escape(title)}</text>'] + \
             [f'<text class="sub" x="24" y="{50 + 16 * i}">{escape(s)}</text>' for i, s in enumerate(subs)]
    rows = []
    for p, ((gname, how, _), stats) in enumerate(zip(groups, st)):
        top = first_top + p * step; bottom = top + plot_h
        Y = lambda v, top=top: top + plot_h * (1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
        parts.append(f'<text class="panel" x="{left}" y="{top - 22}">{escape(gname)}<tspan class="sub" dx="6">· {escape(how)}</tspan></text>')
        for t in ticks:
            parts.append(f'<line class="grid" x1="{left}" x2="{left + panel_w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                         f'<text class="tick" x="{left - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:,}</text>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{top - 6}" text-anchor="end">ms</text>'
                     f'<line class="axis" x1="{left}" x2="{left + panel_w}" y1="{bottom}" y2="{bottom}"/>')
        slot = panel_w / max(len(stats), 1)
        for i, (sid, l1, l2, s) in enumerate(stats):
            cx = left + slot * (i + 0.5); bw = 30
            cls = f'c{cat(sid)}'
            tip = escape(f"{sid} ({l1} {l2}) on {gname} (n={s['n']}): min {fmt_ms(s['min'])}, Q1 {fmt_ms(s['q1'])}, median {fmt_ms(s['median'])}, "
                         f"Q3 {fmt_ms(s['q3'])}, p95 {fmt_ms(s['p95'])}, max {fmt_ms(s['max'])} ms")
            g = [f'<g><title>{tip}</title>',
                 f'<rect x="{cx - bw}" y="{Y(s["hi"]) - 4:.1f}" width="{2 * bw}" height="{Y(s["lo"]) - Y(s["hi"]) + 8:.1f}" fill="transparent"/>',
                 f'<line class="whisker {cls}" x1="{cx}" x2="{cx}" y1="{Y(s["hi"]):.1f}" y2="{Y(s["q3"]):.1f}"/>',
                 f'<line class="whisker {cls}" x1="{cx}" x2="{cx}" y1="{Y(s["q1"]):.1f}" y2="{Y(s["lo"]):.1f}"/>',
                 f'<line class="whisker {cls}" x1="{cx - 7}" x2="{cx + 7}" y1="{Y(s["hi"]):.1f}" y2="{Y(s["hi"]):.1f}"/>',
                 f'<line class="whisker {cls}" x1="{cx - 7}" x2="{cx + 7}" y1="{Y(s["lo"]):.1f}" y2="{Y(s["lo"]):.1f}"/>',
                 f'<rect class="box {cls}" x="{cx - bw / 2}" y="{Y(s["q3"]):.1f}" width="{bw}" height="{max(1.5, Y(s["q1"]) - Y(s["q3"])):.1f}" rx="3"/>',
                 f'<line class="median" x1="{cx - bw / 2}" x2="{cx + bw / 2}" y1="{Y(s["median"]):.1f}" y2="{Y(s["median"]):.1f}"/>']
            g += [f'<circle class="outlier {cls}" cx="{cx}" cy="{Y(o):.1f}" r="2.4"/>' for o in s['outliers']]
            g.append(f'<text class="val" x="{cx + bw / 2 + 4}" y="{Y(s["median"]) + 3.5:.1f}">{fmt_ms(s["median"])}</text></g>')
            parts += g
            nm_ = DISP.get(sid, sid)
            parts.append(mk(sid, cx - 3.3 * len(nm_) - 10, bottom + 14, 4.5)
                         + f'<text class="name{" base" if sid == "hobson" else ""}" x="{cx}" y="{bottom + 18}" text-anchor="middle">{escape(nm_)}</text>'
                         f'<text class="torso" x="{cx}" y="{bottom + 32}" text-anchor="middle">{escape(l1)}</text>'
                         f'<text class="torso" x="{cx}" y="{bottom + 44}" text-anchor="middle">{escape(l2)}</text>')
            rows.append([gname, sid, f'{l1} {l2}'.strip()] + [round(s[k], 2) for k in ('n', 'min', 'q1', 'median', 'q3', 'max', 'p95', 'lo', 'hi')] + [len(s['outliers'])])
    for i, line in enumerate(note):
        parts.append(f'<text class="note" x="24" y="{H - 12 - 15 * (len(note) - 1 - i)}">{escape(line)}</text>')
    write(name, parts, rows, ['benchmark', 'id', 'name', 'n', 'min', 'q1', 'median', 'q3', 'max', 'p95', 'lo', 'hi', 'outliers'])
    return rows


LAB4 = {'hobson': ('baseline', 'bf16'), 'P1': ('W8A8', ''), 'P2': ('W4A4', '(inaccurate)'), 'R1': ('W8A8 +', 'exit at 16'),
        'R2a': ('depth split', '8 layers'), 'A7': ('vocabulary', '64k'), 'A1': ('T5Gemma', 'encoder')}


def fig4():
    req = requests()
    groups = []
    for gname, suites, how in (('JevBench', ['JB-all'], '231 public tasks, first question'),
                               ('Real traffic', ['REAL-agree', 'LONG'], 'REAL-agree + LONG eval requests, first question')):
        rs = [r for s in suites for r in req[s]]
        groups.append((gname, f'{how} (n={len(rs)})', [(i, *LAB4[i], [modeled(k, *r[1:]) for r in rs]) for i, k in BOX4]))
    box_svg('fig4_latency_box_model', 'Per-request latency by configuration, modeled',
            ['Each request\'s latency read off its configuration\'s measured A10G curve at the request\'s own token count (log-log interpolation).',
             'Box: quartiles · line: median · whiskers: 1.5 × IQR · dots: beyond. Both panels on one log scale.'],
            groups, ['Modeled, not timed per request: each curve was measured at exact lengths with one question.',
                     'A7 uses each request\'s own 64k super-token count. R1 exits every decision at layer 16 (fig 5 has the measured cascade).',
                     'IDs refer to docs/IDEAS_EXPLORED.md.'])


def fig5():
    j15 = {}
    for f, key in (('res_dcasc_16', 'R1'), ('res_dcasc_816', 'R1b'), ('res_casc_w4', 'P8')):
        for line in open(f'{R}/j15/res/{f}.jsonl'):
            r = json.loads(line); j15.setdefault(key, []).append(r['casc'])
            if key == 'R1':
                j15.setdefault('P1', []).append(r['b8'])
                j15.setdefault('hob', []).append(curve_at('j15/bf16', r['T']))  # J15's own measured bf16 curve at the request's length
    xr = [json.loads(l) for l in open(f'{R}/q2/res/res_xreal.jsonl')]
    j9 = json.load(open(f'{R}/j9/lat_real_affine.json'))['rows']
    groups = [('J15 requests', f'120 real requests (54 REAL, 18 LONG, 48 JevBench), timed end to end',
               [('hobson', 'baseline, bf16', '(modeled)', j15['hob']), ('P1', 'W8A8', '', j15['P1']), ('R1', 'W8A8 + exit', 'cascade {16}', j15['R1']),
                ('R1b', 'W8A8 + exit', 'cascade {8, 16}', j15['R1b']), ('P8', 'W4A4 draft', '+ W8A8 verify', j15['P8'])]),
              ('J9 requests', f'{len(j9)} real requests (REAL-agree + LONG), bf16, timed end to end',
               [('hobson', 'baseline', 'bf16', [r['native'] for r in j9]), ('A6', 'compiled', 'documents', [r['compiled'] for r in j9])]),
              ('Same 120 requests, box q2b', 'W8A8 base; all four timed on one box, deployed kernels',
               [('hobson', 'baseline, bf16', '(modeled)', [curve_at('j15/bf16', r['T']) for r in xr]),
                ('P1', 'W8A8', '', [r['b8'] for r in xr]), ('P11', 'k64rr', '', [r['k64'] for r in xr]),
                ('R1', 'W8A8 + exit', 'cascade {16}', [r['b8x'] for r in xr]), ('C1', 'k64rr + exit', 'cascade {16}', [r['casc'] for r in xr])])]
    box_svg('fig5_latency_box_meas', 'Per-request latency, measured end to end',
            ['Every request timed on the A10G at its exact length (15-20 warm repetitions each, median).',
             'Box: quartiles · line: median · whiskers: 1.5 × IQR · dots: beyond. All panels on one log scale.'],
            groups, ['hobson-v19 in the J15 and q2b panels is read off J15\'s measured bf16 curve at each request\'s length; J15 did not time bf16 per request.',
                     'R1b exits at layer 8 or 16 (it changes 0.15% of decisions). P8 matches W8A8\'s decisions (re-runs 71% of requests).',
                     'J9 times the first question of each request; A6 compiles the documents of each request\'s state. IDs refer to docs/IDEAS_EXPLORED.md.'])


# ---------------------------------------------------------------- fig 6: measured latency against state length
def fig6():
    j15 = [json.loads(l) for l in open(f'{R}/j15/res/res_dcasc_16.jsonl')]
    j9 = json.load(open(f'{R}/j9/lat_real_affine.json'))['rows']
    xr = [json.loads(l) for l in open(f'{R}/q2/res/res_xreal.jsonl')]
    data = [('J15 requests', '120 real requests, timed on box q2b', [('P1', 'W8A8', [(r['T'], r['b8'], r['id']) for r in xr]),
                                                               ('R1', 'W8A8 + exit {16}', [(r['T'], r['b8x'], r['id']) for r in xr]),
                                                               ('C1', 'k64rr + exit {16}', [(r['T'], r['casc'], r['id']) for r in xr])]),
            ('J9 requests', f'{len(j9)} real requests, bf16', [('hobson', 'bf16, the baseline', [(r['T'], r['native'], r['id']) for r in j9]),
                                                                 ('A6', 'compiled documents', [(r['T'], r['compiled'], r['id']) for r in j9])])]
    W, left, pw = 760, 72, 652
    first_top, ph, step = 150, 250, 370
    H = first_top + step + ph + 90
    xhi = math.ceil(max(t for _, _, ss in data for *_, s in ss for t, _, _ in s) / 1000) * 1000
    yhi = math.ceil(max(v for _, _, ss in data for *_, s in ss for _, v, _ in s) * 1.05 / 50) * 50
    X = lambda v: left + pw * v / xhi
    parts = svg_open(W, H, 'Request latency against state length, measured', 'Scatter of measured per-request latency against state tokens.')
    parts += ['<text class="title" x="24" y="30">Request latency against state length, measured</text>',
              '<text class="sub" x="24" y="50">Every point is one real request timed end to end on the A10G at its exact length. Linear scales.</text>']
    rows = []
    for p, (gname, how, series) in enumerate(data):
        top = first_top + p * step; base = top + ph
        Y = lambda v, top=top: top + ph * (1 - v / yhi)
        items = ([('hobson', 'hobson-v19 bf16, measured curve (same harness): the baseline', False)] if p == 0 else []) + \
                [(sid, f'{DISP.get(sid, sid)} · {nm}', False) for sid, nm, _ in series]
        legend(parts, 24, top - 14, items, wrap=720)
        parts.append(f'<text class="panel" x="{left}" y="{top - 40}">{escape(gname)}<tspan class="sub" dx="6">· {escape(how)}</tspan></text>')
        for t in range(0, yhi + 1, 50 if yhi <= 400 else 100):
            parts.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                         f'<text class="tick" x="{left - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
        for t in range(0, xhi + 1, 1000):
            parts.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{base}"/>'
                         f'<text class="tick" x="{X(t):.1f}" y="{base + 16}" text-anchor="middle">{t:,}</text>')
        parts.append(f'<line class="axis" x1="{left}" x2="{left + pw}" y1="{base}" y2="{base}"/>'
                     f'<text class="tick" x="{left - 34}" y="{top + 4}" text-anchor="end">ms</text>'
                     f'<text class="sub" x="{left + pw / 2}" y="{base + 34}" text-anchor="middle">State tokens</text>')
        for sid, nm, pts in series:
            for t, v, rid in pts:
                parts.append(mk(sid, X(t), Y(v), 3.6, title=f"{DISP.get(sid, sid)} · {rid}: {t:,} tokens, {fmt_ms(v)} ms"))
                rows.append([gname, sid, rid, t, v])
        for sid, nm, pts in series:
            t, v, _ = max(pts)
            parts.append(f'<text class="name{" base" if sid == "hobson" else ""}" x="{X(t) + 9:.1f}" y="{Y(v) + 4:.1f}">{escape(DISP.get(sid, sid))}</text>')
        if p == 0:  # hobson-v19's measured bf16 curve from the same harness, for the baseline
            hp = [(t, v) for t, v in sorted(CURVES['j15/bf16']['pts'].items()) if t <= xhi]
            parts.append(f'<path class="line cbase" d="{" ".join(f"{chr(77) if j == 0 else chr(76)}{X(t):.1f},{Y(v):.1f}" for j, (t, v) in enumerate(hp))}"/>')
            t, v = hp[-1]
            parts.append(f'<text class="name base" x="{X(t) - 6:.1f}" y="{Y(v) - 8:.1f}" text-anchor="end">hobson-v19 bf16</text>')
    parts.append(f'<text class="note" x="24" y="{H - 26}">J15: REAL, LONG and JevBench requests; J9: REAL-agree and LONG requests, first question. '
                 'Hover a point for its request.</text>'
                 f'<text class="note" x="24" y="{H - 11}">IDs refer to docs/IDEAS_EXPLORED.md.</text>')
    write('fig6_length_scatter', parts, rows, ['set', 'id', 'request', 'state_tokens', 'ms'])


def png(theme='light'):
    """PNG copies for documents: rsvg-convert does not resolve CSS custom properties, so substitute them first"""
    import re, shutil, subprocess
    if not shutil.which('rsvg-convert'): return
    block = STYLE.split('@media')[0] if theme == 'light' else STYLE.split('@media')[1]
    var = dict(re.findall(r'--([\w-]+):\s*([^;]+);', block))
    os.makedirs(f'{OUT}/png', exist_ok=True)
    for f in sorted(os.listdir(OUT)):
        if not f.endswith('.svg'): continue
        svg = open(f'{OUT}/{f}', encoding='utf-8').read()
        svg = re.sub(r'@media \(prefers-color-scheme: dark\) \{.*?\n  \}\n', '', svg, flags=re.S)
        svg = re.sub(r'var\(--([\w-]+)\)', lambda m: var.get(m.group(1), '#888'), svg)
        tmp = f'{OUT}/png/.{f}'
        open(tmp, 'w', encoding='utf-8').write(svg)
        subprocess.run(['rsvg-convert', '-w', '1500', tmp, '-o', f'{OUT}/png/{f[:-4]}{"" if theme == "light" else "_dark"}.png'], check=True)
        os.remove(tmp)


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f(); print('wrote', f.__name__)
    png('light')
