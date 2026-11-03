"""Bit-flip attack on the integer keyword-spotting models.

The models are stored as signed integers: 8-bit weights for the int8 models, 4-bit weights (values in
[-7,7], packed in a byte) for the w4a8 / w4a4 models. A flip toggles one bit of a weight's b-bit two's
complement field (b = 8 or 4), so the flipped value stays inside the b-bit range.

Two attackers:
  * pbs   -- progressive bit search (Rakin et al., ICCV 2019): per iteration, rank candidate weight bits by
             a gradient taken on a differentiable surrogate of the integer net, then evaluate the top
             candidate of every weight layer with the exact integer simulator and keep the single flip that
             raises cross-entropy the most on the attack batch.
  * rand  -- flip a uniformly random bit of a uniformly random weight each iteration (baseline).

Loss/accuracy are the exact integer simulator (qmodel.run_int); gradients are only used to rank candidates.
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from common import ASSETS, load_norm
import qmodel as Q
from blob import load_blob, quant_input


# ------------------------------------------------------------------ model loading
def load_model(key, variant):
    base = os.path.join(ASSETS, 'models', variant, key)
    js = json.load(open(base + '.json'))
    net, qops, s_in, _ = load_blob(base + '.bin')
    hi0 = 7 if variant == 'w4a4' else 127            # 4-bit inputs for w4a4, else int8 inputs
    wbits = 4 if variant.startswith('w4') else 8
    base_acc = js['int8_acc'] if variant == 'int8' else js['acc']
    wops = [i for i, d in enumerate(qops) if 'Wq' in d]      # conv / gapfc ops carry weights
    total_bits = int(sum(qops[i]['Wq'].size for i in wops) * wbits)
    return dict(key=key, variant=variant, js=js, net=net, qops=qops, s_in=s_in, hi0=hi0,
                wbits=wbits, base_acc=base_acc, wops=wops, total_bits=total_bits, s_in_json=js['s_in'])


# ------------------------------------------------------------------ one-bit flip in a b-bit field
def flip_value(v, bit, b):
    """Toggle `bit` of the b-bit two's-complement encoding of integer v; return the signed result."""
    mask = (1 << b) - 1
    u = (int(v) & mask) ^ (1 << bit)
    return u - (1 << b) if (u >> (b - 1)) & 1 else u


def _flip_deltas(W, b):
    """For int array W, return deltas[j] = (value after flipping bit j) - W, shape (b,)+W.shape."""
    mask = (1 << b) - 1
    sign = 1 << (b - 1)
    u = (W.astype(np.int64) & mask)
    out = np.empty((b,) + W.shape, np.int64)
    for j in range(b):
        fu = u ^ (1 << j)
        dec = np.where(fu & sign, fu - (1 << b), fu)
        out[j] = dec - W.astype(np.int64)
    return out


# ------------------------------------------------------------------ exact integer forward
def run_logits(m, xq, bs=256):
    out = []
    for i in range(0, len(xq), bs):
        out.append(Q.run_int(m['net'], m['qops'], xq[i:i + bs])[1])
    return np.concatenate(out)


def _softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ce_acc(logits, labels):
    p = _softmax(logits)
    ce = float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())
    acc = float((logits.argmax(1) == labels).mean())
    return ce, acc


# ------------------------------------------------------------------ differentiable surrogate (for ranking only)
class Surrogate:
    """Mirror of qmodel.run_int with float arithmetic and clamp-as-STE, used only to get d(CE)/d(weight)."""
    def __init__(self, m):
        self.m = m
        self.net, self.qops = m['net'], m['qops']
        self.params = {}            # op idx -> torch leaf matching the stored Wq layout
        for i in m['wops']:
            self.params[i] = torch.zeros(self.qops[i]['Wq'].shape, dtype=torch.float64, requires_grad=True)

    def _sync(self):
        for i, p in self.params.items():
            p.data = torch.from_numpy(self.qops[i]['Wq'].astype(np.float64))

    def _hi(self, d, to):
        return d.get('out_hi') or (127 if to.signed else 255)

    def forward(self, xq):
        net = self.net
        t0 = net.t[0]
        x = torch.from_numpy(np.asarray(xq, np.float64)).reshape(len(xq), t0.h, t0.w, t0.c).permute(0, 3, 1, 2)
        ts = {0: x}
        for i, d in enumerate(self.qops):
            ty = d['type']; to = net.t[d['out']]
            if ty == Q.OP_CONV:
                a = ts[d['a']]
                if d['groups'] > 1:                                   # depthwise: stored (kh,kw,c)
                    w = self.params[i].permute(2, 0, 1).unsqueeze(1)  # (c,1,kh,kw)
                else:
                    w = self.params[i].permute(0, 3, 1, 2)            # (cout,cin/g,kh,kw)
                acc = F.conv2d(a, w, torch.from_numpy(d['Bq'].astype(np.float64)), d['stride'], d['pad'], groups=d['groups'])
                if d['relu']:
                    acc = F.relu(acc)
                scale = torch.from_numpy((d['M'].astype(np.float64)) / (2.0 ** d['SH'].astype(np.float64))).view(1, -1, 1, 1)
                v = acc * scale
                if d['boff'] is not None:
                    v = v + torch.from_numpy(d['boff'].astype(np.float64)).view(1, -1, 1, 1)
                hi = self._hi(d, to)
                ts[d['out']] = torch.clamp(v, -hi if to.signed else 0, hi)
            elif ty == Q.OP_ADD:
                a, bb = ts[d['a']], ts[d['bt']]
                v = (a * d['ma'] + bb * d['mb']) / (2.0 ** d['ash'])
                if d['relu']:
                    v = F.relu(v)
                hi = self._hi(d, to)
                ts[d['out']] = torch.clamp(v, -hi if to.signed else 0, hi)
            elif ty == Q.OP_GAPFC:
                a = ts[d['a']]
                S = a.sum((2, 3))                                     # GAP over spatial
                acc = S @ self.params[i].t()
                logits = acc * torch.from_numpy(d['fscale'].astype(np.float64)) + torch.from_numpy(d['fbias'].astype(np.float64))
                ts[d['out']] = logits
            else:
                raise ValueError('unexpected op %d in surrogate' % ty)
        return ts[self.qops[-1]['out']]

    def grads(self, xq, labels):
        self._sync()
        for p in self.params.values():
            if p.grad is not None:
                p.grad = None
        logits = self.forward(xq)
        loss = F.cross_entropy(logits, torch.from_numpy(labels.astype(np.int64)))
        loss.backward()
        return {i: self.params[i].grad.detach().numpy() for i in self.params}


# ------------------------------------------------------------------ attacks
def _record(hist, flip_no, ce, acc, extra):
    hist.append(dict(flip=flip_no, ce=ce, acc=acc, **extra))


def attack_pbs(m, Xa_q, ya, Xe_q, ye, target_acc, max_flips):
    """Progressive bit search. Returns (history list, flips_to_target or None)."""
    sur = Surrogate(m)
    b = m['wbits']
    hist = []
    ce, acc = ce_acc(run_logits(m, Xe_q), ye)
    _record(hist, 0, ce, acc, dict(layer=-1, widx=-1, bit=-1))
    reached = None
    for step in range(1, max_flips + 1):
        g = sur.grads(Xa_q, ya)
        base_ce, _ = ce_acc(run_logits(m, Xa_q), ya)
        best = None                      # (attack_ce, op_idx, flat_widx, bit, delta)
        for i in m['wops']:
            W = m['qops'][i]['Wq']
            deltas = _flip_deltas(W, b)                       # (b,)+W.shape
            score = g[i][None] * deltas                       # first-order dCE estimate
            flat = score.reshape(b, -1)
            jbit = int(flat.max(1).argmax())                  # best bit plane for this layer
            widx = int(flat[jbit].argmax())
            delta = int(deltas[jbit].reshape(-1)[widx])
            if delta == 0:
                continue
            Wf = W.reshape(-1)
            old = int(Wf[widx]); Wf[widx] = old + delta       # tentatively apply
            cand_ce, _ = ce_acc(run_logits(m, Xa_q), ya)
            Wf[widx] = old                                    # restore
            if best is None or cand_ce > best[0]:
                best = (cand_ce, i, widx, jbit, delta)
        if best is None:
            break
        cand_ce, i, widx, jbit, delta = best
        W = m['qops'][i]['Wq']; Wf = W.reshape(-1)
        old = int(Wf[widx])
        assert flip_value(old, jbit, b) == old + delta, 'flip/delta mismatch'
        Wf[widx] = old + delta                                # commit the flip
        ce, acc = ce_acc(run_logits(m, Xe_q), ye)
        _record(hist, step, ce, acc, dict(layer=i, widx=widx, bit=jbit, attack_ce=cand_ce))
        if reached is None and acc <= target_acc:
            reached = step
            break
    return hist, reached


def attack_rand(m, Xe_q, ye, target_acc, max_flips, seed):
    rng = np.random.RandomState(seed)
    b = m['wbits']
    hist = []
    ce, acc = ce_acc(run_logits(m, Xe_q), ye)
    _record(hist, 0, ce, acc, dict(layer=-1, widx=-1, bit=-1))
    sizes = np.array([m['qops'][i]['Wq'].size for i in m['wops']], float)
    probs = sizes / sizes.sum()
    reached = None
    for step in range(1, max_flips + 1):
        i = m['wops'][rng.choice(len(m['wops']), p=probs)]   # weight picked uniformly over all weights
        W = m['qops'][i]['Wq']; Wf = W.reshape(-1)
        widx = rng.randint(Wf.size); bit = rng.randint(b)
        Wf[widx] = flip_value(int(Wf[widx]), bit, b)
        ce, acc = ce_acc(run_logits(m, Xe_q), ye)
        _record(hist, step, ce, acc, dict(layer=i, widx=int(widx), bit=int(bit)))
        if reached is None and acc <= target_acc:
            reached = step
            break
    return hist, reached
