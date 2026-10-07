"""J9 library: compile the deployment.

A gate state = deployment constants (the hook's frame, KB document bodies, hook notes, tool schemas) interleaved with dynamic text
(conversation, tool results). Text-level segmentation (`pieces`) cuts a rendered state into
  frame : the hook's framing text up to the first section header (a true prefix: compiled exactly, function-preserving)
  blk   : a deployment-constant block (KB document body, hook note, tool schema), keyed by its text; compiled ONCE per deployment,
          independently of any request, in a universal context U = "<state>\n"
  dyn   : everything else (computed per request)
Cuts are made after a maximal newline run, so concatenated piece tokenizations equal the whole-state tokenization (checked).

Compiled block (per layer): attention K (post-norm, PRE-RoPE) and V; GDN affine transfer of the block: S_out = A S_in + B, stored as
E = A S_U + B (the final state from U) and A (the transfer, from a v=0 run with S_in = I); its last 3 raw conv rows.
Runtime composition of blocks in ANY order: S <- A_i (S - S_U) + E_i  (exact algebra of the delta rule, given the block's activations);
attention K re-rotated to the block's runtime position (exact).

Layouts (the request's tokens after compile):
  'native' : hobson's own forward (teacher).
  'R'      : [frame][blocks in order of first appearance][dyn pieces in order][question]   (constants first; one live segment)
  'S'      : native order, blocks spliced in place (several live segments)
comp: 'affine' (A_i (S - S_U) + E_i) | 'last' (S <- E_i) | 'skip' (blocks do not touch the GDN state; attention only)
"""
import os, sys, math, json, re, hashlib, collections, bisect
sys.path[:0] = [os.path.expanduser('~/work/j9'), os.path.expanduser('~/work/h7'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
import h3lib as H
from h3lib import rms_zc, DENSE, chunk_gated_delta_rule

# ============================================================ segmentation (text)
HDR = re.compile(r'^--- .* ---\n?$')
NUM = re.compile(r'^(\s*)\d+\. ')
IDL = re.compile(r'^\s+ID: doc_\S+\s*\n?$')
SCORE = re.compile(r'^\s+Score: -?[\d.]+\s*\n?$')
ROLE = re.compile(r'^(user|assistant|tool result|tool:|system note|assistant called|\[\.\.\. |Not shown:|Knowledge-base documents|Search results|result:)')
TRUNC = ' ...[truncated]'


def norm_line(ln):
    return NUM.sub(r'\1#. ', ln.rstrip('\n'))


class LineLib:
    """lines that occur in states of >= k distinct TRAIN tasks (train_pool): the deployment's recurring text"""

    def __init__(self, pool_path, k=3):
        seen = collections.defaultdict(set)
        for l in open(pool_path):
            d = json.loads(l)
            for ln in set(d['state'].split('\n')):
                if len(ln.strip()) >= 2: seen[norm_line(ln)].add(d['task'])
        self.lib = {ln for ln, ts in seen.items() if len(ts) >= k}
        self.libs = sorted(self.lib)

    def const(self, ln):
        n = norm_line(ln)
        if n in self.lib: return True
        if n.endswith(TRUNC.strip()):
            p = n[:-len(TRUNC)] if n.endswith(TRUNC) else n[:-len(TRUNC.strip())]
            if len(p) < 8: return False
            i = bisect.bisect_left(self.libs, p)
            return i < len(self.libs) and self.libs[i].startswith(p)
        return False


def split_lines(text):
    return re.findall(r'[^\n]*\n|[^\n]+$', text)


MARK = re.compile(r'^(\[\.\.\. \d+ messages? omitted \.\.\.\]|\[earlier text omitted\]|\(none\)|none yet|\(no conversation yet\))\s*$')
NOTE0 = re.compile(r'^(system note from the steering hook|tool result: (GUIDANCE|DENIED|CONFIRMATION_FAILED):|.*\[Note from the supervising system)')
TOOL0 = re.compile(r'^(Tool unlocked: |tool result \(unlock_discoverable_agent_tool\): Tool unlocked)')


def pieces(rendered, LL, strict=None):
    """rendered = render_state(state) text. -> list of [kind, text] (kind frame | blk | dyn), ''.join(texts) == rendered
    strict (default: env J9_STRICT=1): blocks only from deployment-constant kinds -- KB document bodies, steering-hook notes, unlocked-tool schemas;
    omission markers, section headers and recurring conversation / tool-result lines stay dynamic (a canonical, bounded library)."""
    if strict is None: strict = os.environ.get('J9_STRICT') == '1'
    lines = split_lines(rendered)
    n = len(lines)
    blank = [not ln.strip() for ln in lines]
    kind = ['dyn'] * n
    i = 0
    # frame: '<state>\n' + framing lines up to and including the first section header (+ blank lines after it)
    if n > 1 and not lines[1].startswith('[earlier text omitted]'):
        j = 0
        while j < n and not HDR.match(lines[j]) and j < 12: j += 1
        if j < n and HDR.match(lines[j]):
            for t in range(j + 1): kind[t] = 'frame'
            t = j + 1
            while t < n and blank[t]: kind[t] = 'frame'; t += 1
            i = t
    # doc bodies (structural: inside a KB entry after its ID / Score lines)
    inbody = [False] * n; t = i
    while t < n:
        if IDL.match(lines[t]):
            u = t + 1
            if u < n and SCORE.match(lines[u]): u += 1
            if u < n and lines[u].lstrip().startswith('Content:'):
                while u < n:
                    ln = lines[u]
                    if u > t + 1 and (HDR.match(ln) or ROLE.match(ln) or (NUM.match(ln) and u + 1 < n and IDL.match(lines[u + 1]))): break
                    inbody[u] = True; u += 1
            t = u
        else:
            t += 1
    const = [False] * n
    ctx = [None] * n; cur = None
    for t in range(i, n):
        ln = lines[t]
        if HDR.match(ln):
            cur = 'tool' if 'TOOL JUST UNLOCKED' in ln else None
        elif NOTE0.match(ln): cur = 'note'
        elif TOOL0.match(ln): cur = 'tool'
        elif ROLE.match(ln) or MARK.match(ln): cur = None
        ctx[t] = cur
    for t in range(i, n):
        if blank[t]: continue
        if strict:
            if MARK.match(lines[t]) or HDR.match(lines[t]): continue
            const[t] = inbody[t] or (ctx[t] in ('note', 'tool') and LL.const(lines[t]) and not IDL.match(lines[t]) and not SCORE.match(lines[t]) and not NUM.match(lines[t]))
        else:
            const[t] = inbody[t] or (LL.const(lines[t]) and not IDL.match(lines[t]) and not SCORE.match(lines[t]) and not NUM.match(lines[t]))
    # runs of constant lines (blank lines attach to the preceding piece); a new run starts at every 'Content:' line
    t = i
    while t < n:
        if const[t]:
            u = t
            while u < n and (const[u] or blank[u]) and not (u > t and lines[u].lstrip().startswith('Content:') and const[u]):
                u += 1
            for w in range(t, u): kind[w] = 'blk'
            # a run must start right after a newline run of the previous piece: if the previous line is blank it already ends the dyn piece
            t = u
        else:
            t += 1
    # trailing '</state>' stays dyn; assemble
    out = []
    for t in range(n):
        k = kind[t]
        if blank[t] and t > 0: k = kind[t - 1] if out else k   # blank lines attach to the preceding piece
        if out and out[-1][0] == k and not (k == 'blk' and lines[t].lstrip().startswith('Content:') and not blank[t]):
            out[-1][1] += lines[t]
        else:
            out.append([k, lines[t]])
    assert ''.join(p[1] for p in out) == rendered
    return out


CTX = {}     # block key -> compile-context token ids (context-primed compile, J9_PRIME=1): the KB entry's own header '1. Title / ID: doc_..'
_HEAD = re.compile(r'(?m)^\d+\. (.+)\n[ \t]+ID: (doc_\S+)[ \t]*\n(?:[ \t]+Score: [-\d.]+[ \t]*\n)?\Z')


def tokenize_pieces(tok, ps, minb=32, prime=None):
    """-> list of (kind, ids, key). Blocks shorter than minb tokens are merged back into dyn (re-tokenized with the neighbours).
    prime (default env J9_PRIME=1): a KB document body is compiled after its own canonical header (rank 1, no score) instead of U alone;
    the key then covers header + body, and CTX[key] holds the header's token ids."""
    if prime is None: prime = os.environ.get('J9_PRIME') == '1'
    toks = [tok(p[1], add_special_tokens=False)['input_ids'] for p in ps]
    merged = []
    for (k, txt), ids in zip(ps, toks):
        if k == 'blk' and len(ids) < minb: k = 'dyn'
        if merged and merged[-1][0] == k == 'dyn':
            merged[-1][1] += txt
        else:
            merged.append([k, txt])
    out = []; prev = ''
    for k, txt in merged:
        ids = tok(txt, add_special_tokens=False)['input_ids']
        key = None
        if k != 'dyn':
            ctx = ''
            if prime and k == 'blk' and txt.lstrip().startswith('Content:'):
                mm = _HEAD.search(prev[-600:])
                if mm: ctx = f"1. {mm.group(1)}\n   ID: {mm.group(2)}\n"
            key = hashlib.sha1((ctx + '\x00' + txt).encode()).hexdigest()[:16]
            if ctx and key not in CTX: CTX[key] = tok(ctx, add_special_tokens=False)['input_ids']
        out.append((k, ids, key))
        prev = txt
    return out


# ============================================================ model
def _rope(t, cos, sin):
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


def gconv(raw, idx, w):
    """depthwise causal conv (kernel 4) + SiLU with explicit per-row input indices. raw [T, C] bf16, idx [T, 4] (-1 = zero), w [C, 4]"""
    pad = torch.cat([raw, torch.zeros(1, raw.shape[1], device=raw.device, dtype=raw.dtype)], 0)
    ii = torch.where(idx < 0, torch.full_like(idx, raw.shape[0]), idx)
    wf = w.float()
    acc = pad[ii[:, 0]].float() * wf[:, 0] + pad[ii[:, 1]].float() * wf[:, 1] + pad[ii[:, 2]].float() * wf[:, 2] + pad[ii[:, 3]].float() * wf[:, 3]
    return F.silu(acc).to(raw.dtype)


class Plan:
    """row bookkeeping for one compiled-layout forward.
    rows: [U (u rows)] [unique blocks, each once] [stream rows (frame + dyn pieces + question) in runtime order]
    segs: stream segments; after segment j the blocks in placed[j] are applied (GDN composition, attention K at runtime positions)."""
    pass


def build_plan(m, req, qids, layout, dev):
    """req: list of (kind, ids, key); qids: question token ids (incl. final <answer>). Returns Plan with tensors on dev.
    Also returns native index of every stream row (for distillation against hobson's native rows)."""
    U = m.U
    P = Plan(); P.layout = layout
    blocks = []; bkey = {}; bctx = []
    for k, ids, key in req:
        if k == 'blk' and key not in bkey:
            cx = CTX.get(key, [])
            bkey[key] = len(blocks); blocks.append(list(cx) + list(ids)); bctx.append(len(cx))
    # native positions of pieces
    nat = []; c = 0
    for k, ids, key in req:
        nat.append(c); c += len(ids)
    Nst = c
    # stream order and placements
    stream = []          # list of (ids, native_start)  in runtime order, split into segments
    segs = [[]]; placed = [[]]
    if layout == 'R':
        has_frame = any(k == 'frame' for k, _, _ in req)
        first = None
        for t, ((k, ids, key), n0) in enumerate(zip(req, nat)):
            if k == 'frame' or (not has_frame and k == 'dyn' and first is None):
                segs[0].append((ids, n0)); first = t
        segs.append([]); placed.append([])
        seen = set()
        for k, ids, key in req:
            if k == 'blk' and key not in seen:
                seen.add(key); placed[0].append(bkey[key])
        for t, ((k, ids, key), n0) in enumerate(zip(req, nat)):
            if k == 'dyn' and t != first: segs[1].append((ids, n0))
        segs[1].append((qids, Nst))
    elif layout == 'S':
        for (k, ids, key), n0 in zip(req, nat):
            if k == 'blk':
                placed[-1].append(bkey[key]); segs.append([]); placed.append([])
            else:
                segs[-1].append((ids, n0))
        segs[-1].append((qids, Nst))
    else:
        raise ValueError(layout)
    # rows
    u = len(U); ids_all = list(U); pos_all = list(range(u))
    brow = []
    for b in blocks:
        brow.append((len(ids_all), len(b))); ids_all += b; pos_all += list(range(u, u + len(b)))
    s0 = len(ids_all)
    # runtime positions: walk segments and placements
    rpos = 0; seg_rows = []; plc = []    # plc: list of (block idx, runtime pos0, after segment j)
    native_idx = []
    for j, sg in enumerate(segs):
        r0 = len(ids_all)
        for ids, n0 in sg:
            ids_all += ids; pos_all += list(range(rpos, rpos + len(ids))); native_idx += list(range(n0, n0 + len(ids))); rpos += len(ids)
        seg_rows.append((r0, len(ids_all) - r0))
        for bi in placed[j]:
            plc.append((bi, rpos, j)); rpos += brow[bi][1] - bctx[bi]
    P.ids = ids_all; P.pos = pos_all; P.u = u; P.brow = brow; P.s0 = s0; P.seg_rows = seg_rows; P.plc = plc
    import itertools
    P.cu = torch.tensor([0] + list(itertools.accumulate(L for _, L in brow)), device=dev, dtype=torch.long)
    P.fast = (layout == 'R' and len(seg_rows) == 2)
    P.native_idx = native_idx; P.T = len(ids_all); P.nq = len(qids)
    P.n_live = len(ids_all) - s0
    # conv indices: each row's 4 inputs (row indices into all rows)
    idx = []
    def hist_ctx(prev_rows):
        return prev_rows[-3:] if len(prev_rows) >= 3 else [-1] * (3 - len(prev_rows)) + prev_rows
    urows = list(range(u))
    for t in range(u):
        idx.append([t - 3 + j if t - 3 + j >= 0 else -1 for j in range(3)] + [t])
    for (r0, L) in brow:
        h = hist_ctx(urows)
        for t in range(L):
            prev = (h + list(range(r0, r0 + t)))[-3:]
            idx.append(prev + [r0 + t])
    last_rows = []   # rows (in all-rows index) that precede the current stream position, for conv history (blocks included)
    for j, (r0, L) in enumerate(seg_rows):
        for t in range(L):
            prev = hist_ctx(last_rows)
            idx.append(prev + [r0 + t]); last_rows.append(r0 + t)
        for bi, rp, jj in plc:
            if jj == j:
                b0, bl = brow[bi]; last_rows += list(range(b0, b0 + bl))
    P.cidx = torch.tensor(idx, device=dev, dtype=torch.long)
    # fast conv: runs (U, each block, each stream segment), each preceded by its 3 history rows, one fla conv call
    cin = []; cout = [0] * len(ids_all)
    runs = [(0, u, [-1, -1, -1])]
    for (r0, L) in brow: runs.append((r0, L, hist_ctx(urows)))
    lr = []
    for j, (r0, L) in enumerate(seg_rows):
        runs.append((r0, L, hist_ctx(lr)))
        lr += list(range(r0, r0 + L))
        for bi, rp, jj in plc:
            if jj == j:
                b0, bl = brow[bi]; lr += list(range(b0, b0 + bl))
    for r0, L, hh in runs:
        if L == 0: continue
        cin += hh
        for t in range(L):
            cout[r0 + t] = len(cin); cin.append(r0 + t)
    P.cin = torch.tensor(cin, device=dev, dtype=torch.long); P.cout = torch.tensor(cout, device=dev, dtype=torch.long)
    # attention masks
    T = P.T; pos_t = torch.tensor(pos_all, device=dev)
    # stream rows: keys = [placed block instances (runtime pos)] + stream rows ; visible iff key pos < query pos (blocks) / <= (stream)
    kp = []; krow = []
    for bi, rp, j in plc:
        b0, bl = brow[bi]; nc = bctx[bi]; kp += list(range(rp, rp + bl - nc)); krow += list(range(b0 + nc, b0 + bl))
    P.krow = torch.tensor(krow, device=dev, dtype=torch.long); P.kpos = torch.tensor(kp, device=dev, dtype=torch.float32)
    P.kseg = [[] for _ in seg_rows]; c = 0
    for bi, rp, j in plc:
        bl = brow[bi][1] - bctx[bi]; P.kseg[j].append((c, c + bl)); c += bl
    return P


class J9(H.H3):
    """hobson-v19 (+ optional LoRA, head) with compiled-constant layouts.
    compile_only=True: the LoRA is a COMPILE ADAPTER: it acts only on the compile rows (U + blocks = rows [0, P.s0)), so the live path
    (frame, dynamic state, question) runs exactly hobson's weights; the adapter only shapes the cached K/V and GDN states of the constants."""
    compile_only = False
    _nrm = None

    def lin(self, x, i, nm, qc, xn=None):
        if self._nrm is None or self.lora is None:
            return super().lin(x, i, nm, qc, xn)
        W = self.L[i][nm]; lo = self.lora[i][nm]; n = self._nrm
        y = x @ W.t()
        if n <= 0: return y
        d = ((x[:n] @ lo.A.t().to(x.dtype)) @ lo.B.t().to(x.dtype)) * self.lora_scale
        return torch.cat([y[:n] + d, y[n:]], 0)

    def setup_u(self, tok):
        self.U = tok('<state>\n', add_special_tokens=False)['input_ids']

    def cos_sin(self, pos):
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        return fr.cos().to(torch.bfloat16), fr.sin().to(torch.bfloat16)

    def layer_c(self, i, x, P, comp, cs_all, cs_k):
        self._nrm = getattr(P, 'nrm', None)
        try:
            return self._layer_c(i, x, P, comp, cs_all, cs_k)
        finally:
            self._nrm = None

    def _layer_c(self, i, x, P, comp, cs_all, cs_k):
        d = self.L[i]; T = x.shape[0]; eps = self.eps; u = P.u; s0 = P.s0
        h = self.bnorm(x, i, 0)
        proj = self.lin(h, i, 'Win', DENSE)
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()).to(torch.bfloat16); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            rp = torch.cat([raw, torch.zeros(1, raw.shape[1], device=raw.device, dtype=raw.dtype)], 0)
            xin = rp[torch.where(P.cin < 0, torch.full_like(P.cin, raw.shape[0]), P.cin)]
            cy = H.fla_conv(xin[None].contiguous(), d['conv_w'], None, activation='silu')
            cy = cy[0] if isinstance(cy, tuple) else cy
            cv = cy[0][P.cout]
            q, k, v = cv.split(2048, dim=-1)
            q = q.reshape(T, 16, 128); k = k.reshape(T, 16, 128); v = v.reshape(T, 16, 128)
            outs = [None] * 3
            # U
            oU, SU = chunk_gated_delta_rule(q[None, :u], k[None, :u], v[None, :u], g[None, :u], beta[None, :u], use_qk_l2norm_in_kernel=True, output_final_state=True)
            outs[0] = oU[0]
            # blocks (varlen, from S_U)
            nb = len(P.brow)
            if nb:
                cu = P.cu
                sl = slice(u, s0)
                oB, E = chunk_gated_delta_rule(q[None, sl], k[None, sl], v[None, sl], g[None, sl], beta[None, sl], initial_state=SU.expand(nb, -1, -1, -1).contiguous(),
                                               use_qk_l2norm_in_kernel=True, output_final_state=True, cu_seqlens=cu)
                outs[1] = oB[0]
                if comp == 'affine':
                    I0 = torch.eye(128, device=x.device, dtype=torch.float32)[None, None].expand(nb, 16, 128, 128).contiguous()
                    _, A = chunk_gated_delta_rule(q[None, sl], k[None, sl], torch.zeros_like(v[None, sl]), g[None, sl], beta[None, sl], initial_state=I0,
                                                  use_qk_l2norm_in_kernel=True, output_final_state=True, cu_seqlens=cu)
            # stream segments
            S = None; so = []
            for j, (r0, L) in enumerate(P.seg_rows):
                if L:
                    sl = slice(r0, r0 + L)
                    o_, S = chunk_gated_delta_rule(q[None, sl], k[None, sl], v[None, sl], g[None, sl], beta[None, sl], initial_state=S,
                                                   use_qk_l2norm_in_kernel=True, output_final_state=True)
                    so.append(o_[0])
                for bi, rp, jj in P.plc:
                    if jj != j or comp == 'skip': continue
                    if comp == 'last' or S is None: S = E[bi:bi + 1]
                    else: S = torch.matmul(A[bi:bi + 1], S - SU) + E[bi:bi + 1]
            outs[2] = torch.cat(so, 0)
            o = torch.cat([t for t in outs if t is not None], 0)
            of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(x.dtype).reshape(T, 2048)
        else:
            qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
            cos, sin = cs_all
            qr = _rope(qh, cos, sin); kr = _rope(kk, cos, sin)
            outs = []
            # U causal
            outs.append(F.scaled_dot_product_attention(qr[None, :u].transpose(1, 2), kr[None, :u].transpose(1, 2), v[None, :u].transpose(1, 2), is_causal=True, enable_gqa=True)[0].transpose(0, 1))
            from torch.nn.attention.bias import causal_lower_right
            for (r0, L) in P.brow:
                Kb = torch.cat([kr[:u], kr[r0:r0 + L]], 0); Vb = torch.cat([v[:u], v[r0:r0 + L]], 0)
                outs.append(F.scaled_dot_product_attention(qr[None, r0:r0 + L].transpose(1, 2), Kb[None].transpose(1, 2), Vb[None].transpose(1, 2),
                                                           attn_mask=causal_lower_right(L, u + L), enable_gqa=True)[0].transpose(0, 1))
            # stream: keys in runtime order = interleaved stream segments and placed block instances (K re-rotated to runtime positions);
            # segment j's queries see a prefix of that order -> one lower-right causal flash call per segment
            parts_k = []; parts_v = []; ends = []; tot = 0
            ck, sk = cs_k if cs_k is not None else (None, None)
            kb_all = _rope(kk[P.krow], ck, sk) if len(P.krow) else None
            for j, (r0, L) in enumerate(P.seg_rows):
                parts_k.append(kr[r0:r0 + L]); parts_v.append(v[r0:r0 + L]); tot += L; ends.append(tot)
                for (a0, a1) in P.kseg[j]:
                    parts_k.append(kb_all[a0:a1]); parts_v.append(v[P.krow[a0:a1]]); tot += a1 - a0
            K = torch.cat(parts_k, 0); V = torch.cat(parts_v, 0)
            for j, (r0, L) in enumerate(P.seg_rows):
                if L == 0: continue
                e = ends[j]
                outs.append(F.scaled_dot_product_attention(qr[None, r0:r0 + L].transpose(1, 2), K[None, :e].transpose(1, 2), V[None, :e].transpose(1, 2),
                                                           attn_mask=causal_lower_right(L, e), enable_gqa=True)[0].transpose(0, 1))
            o = (torch.cat(outs, 0) * torch.sigmoid(gate)).reshape(T, 2048)
        x = x + self.lin(o, i, 'Wo', DENSE)
        h2 = self.bnorm(x, i, 1)
        gu = self.lin(h2, i, 'Wgu', DENSE); I = d['I']
        x = x + self.lin(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd', DENSE)
        return x

    def fwd_c(self, P, comp='affine', ckpt=False, keep=(), keep_rows=None):
        P.nrm = P.s0 if self.compile_only else None     # read by layer_c (also on checkpoint recompute)
        return self._fwd_c(P, comp, ckpt, keep, keep_rows)

    def _fwd_c(self, P, comp='affine', ckpt=False, keep=(), keep_rows=None):
        ids_t = torch.as_tensor(P.ids, device=self.dev)
        x = F.embedding(ids_t, self.embed)
        cs_all = self.cos_sin(torch.as_tensor(P.pos, device=self.dev, dtype=torch.float32))
        cs_k = self.cos_sin(P.kpos) if len(P.krow) else None
        self.kept = {}
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self.layer_c, i, x, P, comp, cs_all, cs_k, use_reentrant=False)
            else:
                x = self.layer_c(i, x, P, comp, cs_all, cs_k)
            if i in keep: self.kept[i] = x[keep_rows]
        return rms_zc(x, self.norm_w, self.eps)

    def logits_c(self, P, h, opt, kind, head=None):
        """opt: option token offsets within the question; the question is the last P.nq stream rows"""
        head = head or self.head or self.head0
        q0 = P.T - P.nq
        oi = torch.tensor([q0 + o for o in opt], device=self.dev)
        return head(h[P.T - 1].float()[None], h[oi].float()[None])[0] / self.temp(kind)

    def native_logits(self, s, q, opt, kind, head=None, keep=(), keep_rows=None):
        """hobson's own layout, one sequence [state][question]"""
        head = head or self.head or self.head0
        out = self.forward(list(s) + list(q), keep=keep)
        h, kept = out if keep else (out, None)
        if keep: self.kept = {i: kept[i][keep_rows] for i in keep}
        T = h.shape[0]; q0 = len(s)
        oi = torch.tensor([q0 + o for o in opt], device=self.dev)
        return head(h[T - 1].float()[None], h[oi].float()[None])[0] / self.temp(kind)
