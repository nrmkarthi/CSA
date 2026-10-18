"""Layer 3 (attack classifier) and Layer 4 (situation-awareness engine) of XAI-CSA.

Perception    : alert events = maximal runs of fused score > 1 (gaps <= GAP_MERGE s merged).
Comprehension : event correlation into incidents (temporal + asset-graph affinity),
                incident phase (normal / active / recovery) and affected-process estimation.
Projection    : attack continuation (is the attack still active at t+H?) and
                asset propagation (which tags will become anomalous within H s).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from common import segments, ewma, PROCS

GAP_MERGE = 10
REC_WIN = 600         # reference recovery window after an attack ends (s)
QUIET_CLOSE = 600     # incident closes after this many quiet seconds
H_PROJ = 60
TAG_ON = 0.05         # tag counted as anomalous when its attribution exceeds 5 % of the alert threshold


# --------------------------------------------------------------------------- features
def proc_agg(phi, proc_of):
    return np.stack([phi[:, proc_of == k].sum(1) for k in range(4)], 1)


def lagged(x, k):
    return np.vstack([np.repeat(x[:1], k, 0), x[:-k]]) if x.ndim == 2 else np.r_[np.repeat(x[:1], k), x[:-k]]


def frame_features(phi, proc_of):
    s = phi.sum(1)
    pa = proc_agg(phi, proc_of)
    pa_share = pa / (pa.sum(1, keepdims=True) + 1e-9)
    slope10 = s - lagged(s, 10)
    slope60 = s - lagged(s, 60)
    from scipy.ndimage import maximum_filter1d
    smax = maximum_filter1d(s, 121, origin=60)            # causal max over last 121 s
    # time since the score last exceeded 1 (capped) and time above threshold in last 300 s
    above = s > 1
    last = np.full(len(s), 3600.0); c = 3600.0
    for i, a in enumerate(above):
        c = 0.0 if a else min(c + 1, 3600.0); last[i] = c
    frac300 = np.convolve(above.astype(float), np.ones(300), 'full')[:len(s)] / 300
    long = ewma(pa, 0.005)                                # slow (situation-level) accumulation
    long_share = long / (long.sum(1, keepdims=True) + 1e-9)
    F = np.hstack([phi, pa, pa_share, long_share, np.c_[s, slope10, slope60, smax, last, frac300]])
    return F.astype(np.float32)


def ref_states(L):
    """Reference incident state: 0 normal, 1 active attack, 2 recovery (<= REC_WIN s after attack)."""
    y = L[:, 0]
    st = np.zeros(len(y), np.int8)
    for s, e in segments(y):
        st[e:min(len(y), e + REC_WIN)] = np.where(st[e:min(len(y), e + REC_WIN)] == 0, 2, st[e:min(len(y), e + REC_WIN)])
    st[y == 1] = 1
    return st


# --------------------------------------------------------------------------- classifier
class SituationClassifier:
    """Phase classifier (normal/active/recovery) + per-process target classifiers (P1..P3)."""
    def __init__(self, seed=0, use_xai_feats=True):
        self.seed = seed; self.use_xai = use_xai_feats

    def _sel(self, F, nfeat):
        if self.use_xai:
            return F
        return F[:, :nfeat]  # only raw per-tag attribution (no situation-level context)

    def fit(self, Fs, Ls, nfeat, exclude_proc=None):
        rng = np.random.default_rng(self.seed)
        Xs, Ys, Ts = [], [], []
        for F, L in zip(Fs, Ls):
            st = ref_states(L)
            cand = (F[:, -6] > 0.3) | (st > 0)            # informative region + random normal subsample
            keep = cand | (rng.random(len(F)) < 0.03)
            if exclude_proc is not None:                   # zero-day: remove all attacks touching that process
                bad = np.zeros(len(F), bool)
                for s, e in segments(L[:, 0]):
                    if L[s:e, exclude_proc + 1].any():
                        bad[s:min(len(F), e + REC_WIN)] = True
                keep &= ~bad
            Xs.append(self._sel(F[keep], nfeat)); Ys.append(st[keep]); Ts.append(L[keep, 1:4])
        X = np.vstack(Xs); y = np.concatenate(Ys); T = np.vstack(Ts)
        self.phase = HGB(max_iter=200, learning_rate=0.1, random_state=self.seed, class_weight='balanced').fit(X, y)
        act = y == 1
        self.target = []
        for k in range(3):
            if T[act, k].min() == T[act, k].max():
                self.target.append(None); continue
            self.target.append(HGB(max_iter=150, random_state=self.seed).fit(X[act], T[act, k]))
        return self

    def predict(self, F, nfeat):
        X = self._sel(F, nfeat)
        ph = self.phase.predict_proba(X)
        if ph.shape[1] < 3:
            ph = np.c_[ph, np.zeros((len(ph), 3 - ph.shape[1]))]
        tg = np.stack([m.predict_proba(X)[:, 1] if m is not None else np.zeros(len(X)) for m in self.target], 1)
        return ph, tg


# --------------------------------------------------------------------------- perception
def alert_events(score):
    segs = segments(score > 1)
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= GAP_MERGE:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# --------------------------------------------------------------------------- comprehension
def graph_diffuse(v, A, beta=0.5):
    """One-hop diffusion of a tag-level vector over the asset-dependency graph."""
    return (1 - beta) * v + beta * (A @ v)


def correlate(events, phi, A, mode='graph_phase', dt=300, gamma=0.6, dt_small=30, ph=None):
    """Return incident id per alert event.  mode in {'none','temporal','graph'}."""
    inc = []; ends = []; vecs = []
    for j, (s, e) in enumerate(events):
        v = phi[s:e].sum(0); v = graph_diffuse(v / (np.linalg.norm(v) + 1e-9), A)
        v /= np.linalg.norm(v) + 1e-9
        best = None
        if mode != 'none' and inc:
            k = inc[-1] if mode == 'temporal' else None
            cands = range(len(ends)) if mode in ('graph', 'graph_phase') else [inc[-1]]
            onset = False
            if mode == 'graph_phase' and ph is not None:
                # a fresh attack onset: the plant looked non-attacked just before the event and the
                # comprehension classifier sees an active attack at the start of the event
                pre = ph[max(0, s - 10):s + 1]
                onset = pre[:, 1].mean() < 0.3 and ph[s:min(e, s + 30), 1].max() > 0.5
            for c in cands:
                gap = s - ends[c]
                if gap > dt or onset:
                    continue
                if mode == 'temporal' or gap <= dt_small or float(v @ vecs[c]) >= gamma:
                    best = c if best is None or ends[c] > ends[best] else best
        if best is None:
            ends.append(e); vecs.append(v); inc.append(len(ends) - 1)
        else:
            ends[best] = max(ends[best], e)
            vecs[best] = vecs[best] + v; vecs[best] /= np.linalg.norm(vecs[best]) + 1e-9
            inc.append(best)
    return np.array(inc)


def event_truth(events, L):
    """Ground-truth cluster of each alert event: attack-episode id if it overlaps the attack or its
    recovery window, else a unique negative id (false alert)."""
    eps = segments(L[:, 0])
    lab = []
    for j, (s, e) in enumerate(events):
        g = -(j + 1)
        for k, (a, b) in enumerate(eps):
            if s < b + REC_WIN and e > a:
                g = k; break
        lab.append(g)
    return np.array(lab)


def pair_counts(pred, true):
    n = len(pred)
    if n < 2:
        return np.zeros(3)
    P = pred[:, None] == pred[None, :]; T = true[:, None] == true[None, :]
    iu = np.triu_indices(n, 1); P, T = P[iu], T[iu]
    return np.array([(P & T).sum(), (P & ~T).sum(), (~P & T).sum()], float)


def bcubed(pred, true):
    """B-cubed precision / recall per element (lists, to be pooled across files)."""
    pr, rc = [], []
    for i in range(len(pred)):
        c = pred == pred[i]; g = true == true[i]
        pr.append((c & g).sum() / c.sum()); rc.append((c & g).sum() / g.sum())
    return pr, rc


def pairwise_f1(pred, true):
    n = len(pred)
    if n < 2:
        return dict(precision=1.0, recall=1.0, f1=1.0)
    P = pred[:, None] == pred[None, :]; T = true[:, None] == true[None, :]
    iu = np.triu_indices(n, 1)
    P, T = P[iu], T[iu]
    tp = (P & T).sum(); fp = (P & ~T).sum(); fn = (~P & T).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(precision=float(pr), recall=float(rc), f1=float(2 * pr * rc / max(pr + rc, 1e-12)))


def incidents_from(events, inc_ids, T):
    """Per-second incident membership: incident open from first alert until QUIET_CLOSE s after last alert."""
    open_id = np.full(T, -1)
    spans = {}
    for (s, e), k in zip(events, inc_ids):
        a, b = spans.get(k, (s, e)); spans[k] = (min(a, s), max(b, e))
    for k, (a, b) in spans.items():
        open_id[a:min(T, b + QUIET_CLOSE)] = np.where(open_id[a:min(T, b + QUIET_CLOSE)] < 0, k, open_id[a:min(T, b + QUIET_CLOSE)])
    return open_id, spans


def build_graph(det, thr=0.1):
    """Asset-dependency graph over all tags: |partial correlation| edges between non-constant tags
    plus structural edges between tags of the same sub-process (weight 0.1)."""
    nf = len(det.feats)
    A = np.zeros((nf, nf))
    pc = np.abs(det.gcr.partial_corr()); np.fill_diagonal(pc, 0)
    pc[pc < thr] = 0
    ik = det.idx_keep
    A[np.ix_(ik, ik)] = pc
    same = det.proc_of[:, None] == det.proc_of[None, :]
    A = np.maximum(A, 0.1 * same); np.fill_diagonal(A, 0)
    A = A / (A.sum(1, keepdims=True) + 1e-9)
    return A
