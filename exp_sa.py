"""RQ1: detection, event correlation, situation-awareness (perception/comprehension/projection), triage."""
import numpy as np, sys, time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from core import *
from sa import (pair_counts, bcubed, event_truth, ref_states, proc_agg, TAG_ON, H_PROJ, REC_WIN)
from response import CRIT

TAG = sys.argv[1] if len(sys.argv) > 1 else 'central'
VER = sys.argv[2] if len(sys.argv) > 2 else 'hai-21.03'
import os
SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else SEEDS


def detection_metrics(S):
    y = np.concatenate([f['L'][:, 0] for f in S]).astype(bool)
    s = np.concatenate([f['s'] for f in S])
    ph = np.vstack([f['ph'] for f in S])
    out = {}
    variants = {
        'anomaly_only': (s > 1, s),
        'classifier_only': (ph[:, 1] > 0.5, ph[:, 1]),
        'hybrid': ((s > 1) & ~(ph[:, 2] > 0.5), s * (1 - ph[:, 2])),
    }
    for k, (pred, sc) in variants.items():
        m = point_metrics(y, pred, sc)
        dets, lats = [], []
        off = 0
        for f in S:
            T = len(f['s']); d, l = event_metrics(f['L'][:, 0], pred[off:off + T]); dets += list(d); lats += list(l); off += T
        m['event_recall'] = float(np.mean(dets)); m['latency_median'] = float(np.nanmedian(lats)); m['latency_mean'] = float(np.nanmean(lats))
        out[k] = m
    # attack-category (targeted sub-process) classification on detected attack seconds
    T3 = np.vstack([f['L'][:, 1:4] for f in S]); tg = np.vstack([f['tg'] for f in S])
    sel = y & (s > 1)
    f1s = [f1_score(T3[sel, k], tg[sel, k] > 0.5, zero_division=0) for k in range(3)]
    out['target_macro_f1'] = float(np.mean(f1s)); out['target_f1_per_proc'] = [float(v) for v in f1s]
    return out


def correlation_metrics(det, files, A):
    res = {}
    for mode in ['none', 'temporal', 'graph', 'graph_phase']:
        pc = np.zeros(3); bp, br = [], []; n_inc = 0; n_ev = 0; hours = 0; true_inc = 0
        for f in files:
            S = situation(det, f, A, mode if mode != 'none' else 'alert')
            if mode == 'none':
                S['inc'] = np.arange(len(S['events']))
            tr = event_truth(S['events'], S['L'])
            pc += pair_counts(S['inc'], tr); p_, r_ = bcubed(S['inc'], tr); bp += p_; br += r_
            n_inc += len(set(S['inc'].tolist())); n_ev += len(S['events']); hours += len(S['s']) / 3600
            # incident precision: incident overlaps an active attack interval
            for k in set(S['inc'].tolist()):
                evs = [S['events'][j] for j in np.where(S['inc'] == k)[0]]
                if any(S['L'][a:b, 0].any() for a, b in evs):
                    true_inc += 1
        P = pc[0] / max(pc[0] + pc[1], 1); R = pc[0] / max(pc[0] + pc[2], 1)
        BP, BR = np.mean(bp), np.mean(br)
        res[mode] = dict(pair_f1=float(2 * P * R / max(P + R, 1e-12)), pair_p=float(P), pair_r=float(R),
                         bcubed_f1=float(2 * BP * BR / (BP + BR)), incidents_per_day=n_inc / hours * 24,
                         alerts_per_day=n_ev / hours * 24, incident_precision=true_inc / max(n_inc, 1))
    return res


def state_metrics(S):
    ref = np.concatenate([ref_states(f['L']) for f in S])
    s = np.concatenate([f['s'] for f in S]); ph = np.vstack([f['ph'] for f in S])
    op = np.concatenate([f['open_id'] for f in S])
    det_only = (s > 1).astype(int)                                  # no recovery notion
    sa = np.where(op >= 0, np.argmax(ph, 1), 0)
    out = {}
    for k, est in [('detection_only', det_only), ('sa_engine', sa)]:
        out[k] = dict(acc=float((est == ref).mean()), macro_f1=float(f1_score(ref, est, average='macro')),
                      f1_per_state=[float(v) for v in f1_score(ref, est, average=None, labels=[0, 1, 2])])
    # full-state accuracy during active attack (state AND targeted process set correct)
    T3 = np.vstack([f['L'][:, 1:4] for f in S]).astype(bool); tg = np.vstack([f['tg'] for f in S]); gs = np.vstack([f['gs'] for f in S])
    act = ref == 1
    cons = 0.5 * tg + 0.5 * gs
    for k, est, tset in [('detection_only', det_only, None), ('sa_clf', sa, tg > 0.5), ('sa_consensus', sa, cons > 0.5)]:
        if tset is None:
            # detection-only: single process = argmax of instantaneous attribution
            phi_p = np.vstack([proc_agg(f['phi'], DET.proc_of)[:, :3] for f in S])
            tset = np.zeros_like(T3); tset[np.arange(len(tset)), np.argmax(phi_p, 1)] = True
        ok = (est[act] == 1) & (tset[act] == T3[act]).all(1)
        out[k + '_fullstate_acc_active'] = float(ok.mean())
    return out


def impact_metrics(S):
    """Asset-impact estimation per detected attack episode (Jaccard of affected-process sets)."""
    res = {'detection_only': [], 'sa_clf': [], 'sa_consensus': []}
    for f in S:
        L = f['L']; s = f['s']
        pp = proc_agg(f['phi'], DET.proc_of)[:, :3]
        for a, b in segments(L[:, 0]):
            true = set(k for k in range(3) if L[a:b, k + 1].any())
            hit = np.where(s[a:b] > 1)[0]
            if len(hit) == 0:
                continue
            t0 = a + hit[0]; win = a + hit
            ests = {'detection_only': {int(np.argmax(pp[t0]))},
                    'sa_clf': set(np.where(f['tg'][win].mean(0) > 0.5)[0].tolist()) or {int(np.argmax(f['tg'][win].mean(0)))},
                    'sa_consensus': set(np.where((0.5 * f['tg'][win] + 0.5 * f['gs'][win]).mean(0) > 0.5)[0].tolist())
                    or {int(np.argmax((0.5 * f['tg'][win] + 0.5 * f['gs'][win]).mean(0)))}}
            for k, e in ests.items():
                res[k].append(len(e & true) / len(e | true))
    return {k: dict(jaccard=float(np.mean(v)), exact=float(np.mean(np.array(v) == 1))) for k, v in res.items()}


def proj_points(f, step=10):
    T = len(f['s']); pts = np.arange(0, T - H_PROJ, step)
    return pts[f['open_id'][pts] >= 0]


def projection_features(f, pts):
    s = f['s']; ph = f['ph']; op = f['open_id']
    start = {}
    for t in range(len(op)):
        if op[t] >= 0 and op[t] not in start:
            start[op[t]] = t
    age = np.array([t - start[op[t]] for t in pts])
    sl = s[pts] - s[np.maximum(pts - 10, 0)]
    pam = np.array([ph[max(0, t - 60):t + 1, 1].mean() for t in pts])
    return np.c_[ph[pts], s[pts], sl, age, pam]


def projection_metrics(S):
    K = len(S)
    P = [proj_points(f) for f in S]
    Y = [f['L'][p + H_PROJ, 0] for f, p in zip(S, P)]
    X = [projection_features(f, p) for f, p in zip(S, P)]
    preds = {'persistence': [], 'phase': [], 'sa_hazard': []}; ys = []; probs = {'phase': [], 'sa_hazard': [], 'persistence': []}
    for k in range(K):
        tr = [j for j in range(K) if j != k]
        m = HGB(max_iter=150, random_state=0).fit(np.vstack([X[j] for j in tr]), np.concatenate([Y[j] for j in tr]))
        pr = m.predict_proba(X[k])[:, 1]
        preds['sa_hazard'].append(pr > 0.5); probs['sa_hazard'].append(pr)
        preds['phase'].append(S[k]['ph'][P[k], 1] > 0.5); probs['phase'].append(S[k]['ph'][P[k], 1])
        preds['persistence'].append(S[k]['s'][P[k]] > 1); probs['persistence'].append(S[k]['s'][P[k]])
        ys.append(Y[k])
    y = np.concatenate(ys)
    cont = {}
    for k in preds:
        pr = np.concatenate(preds[k]); sc = np.concatenate(probs[k])
        m = point_metrics(y, pr, sc); m['roc_auc'] = float(roc_auc_score(y, sc)); cont[k] = m
    # ---- asset propagation: which tags become newly anomalous within H s
    def tag_sets(f, t):
        phi = f['phi']
        recent = (phi[max(0, t - 30):t + 1] > TAG_ON).any(0)
        fut = (phi[t + 1:t + 1 + H_PROJ] > TAG_ON).any(0) & ~recent
        return recent, fut
    nf = S[0]['phi'].shape[1]
    out_s = {m: [] for m in ['prior', 'graph', 'transition', 'hybrid']}; out_y = []; rec5 = {m: [] for m in out_s}
    stats = []
    for f, p in zip(S, P):
        R_ = []; F_ = []; C_ = []
        for t in p:
            r, fu = tag_sets(f, t); R_.append(r); F_.append(fu); C_.append(np.clip(f['phi'][t], 0, None))
        stats.append((np.array(R_), np.array(F_), np.array(C_)))
    for k in range(K):
        tr = [j for j in range(K) if j != k]
        Rtr = np.vstack([stats[j][0] for j in tr]); Ftr = np.vstack([stats[j][1] for j in tr])
        prior = Ftr.mean(0)
        M = (Rtr.T.astype(float) @ Ftr.astype(float)) / (Rtr.sum(0)[:, None] + 1)   # P(i new | j recent)
        Rk, Fk, Ck = stats[k]
        for r, fu, c in zip(Rk, Fk, Ck):
            cand = ~r
            if cand.sum() == 0:
                continue
            g = AG.T @ (c / (c.sum() + 1e-9))
            tr_s = (r[:, None] * M).max(0) if r.any() else prior
            from scipy.stats import rankdata
            hyb = rankdata(g) + rankdata(tr_s)
            sc = dict(prior=prior, graph=g, transition=tr_s, hybrid=hyb)
            for m in out_s:
                out_s[m].append(sc[m][cand])
                if fu.any():
                    top = np.where(cand)[0][np.argsort(-sc[m][cand])[:5]]
                    rec5[m].append(fu[top].sum() / fu.sum())
            out_y.append(fu[cand])
    yy = np.concatenate(out_y)
    prop = {m: dict(ap=float(average_precision_score(yy, np.concatenate(out_s[m]))),
                    auc=float(roc_auc_score(yy, np.concatenate(out_s[m]))), recall_at5=float(np.mean(rec5[m])))
            for m in out_s}
    prop['base_rate'] = float(yy.mean())
    return dict(continuation=cont, propagation=prop)


def triage_metrics(S):
    ys, base, risk = [], [], []
    for f in S:
        for a, b in f['events']:
            ys.append(int(f['L'][a:b, 0].any()))
            base.append(f['s'][a:b].max())
            w = slice(a, min(b, a + 30))   # decision made within the first 30 s of the alert
            risk.append(float((f['ph'][w, 1] * (f['tg'][w] @ CRIT)).max()))
    ys = np.array(ys)
    return dict(n_events=int(len(ys)), frac_true=float(ys.mean()),
                peak_score=dict(auc=float(roc_auc_score(ys, base)), ap=float(average_precision_score(ys, base))),
                sa_risk=dict(auc=float(roc_auc_score(ys, risk)), ap=float(average_precision_score(ys, risk))))


def splice_campaigns(S, rng, gap_lo=10, gap_hi=120, pre=60, post=120):
    """Compressed-campaign stress test: attack windows (with recovery tails) from the recorded files are
    concatenated with short real normal gaps so that independent incidents become temporally adjacent."""
    blocks, normals = [], []
    for f in S:
        L = f['L']; T = len(L)
        eps = segments(L[:, 0])
        busy = np.zeros(T, bool)
        for a, b in eps:
            lo, hi = max(0, a - pre), min(T, b + post)
            blocks.append((f, lo, hi)); busy[max(0, a - pre - 600):min(T, b + REC_WIN + 600)] = True
        normals.append((f, np.where(~busy)[0]))
    order = rng.permutation(len(blocks))
    phis, Ls, ids, phs = [], [], [], []
    for n, j in enumerate(order):
        f, lo, hi = blocks[j]
        g = int(rng.integers(gap_lo, gap_hi + 1))
        fn, idx = normals[rng.integers(len(normals))]
        st = idx[rng.integers(len(idx) - g)] if len(idx) > g else idx[0]
        if np.all(np.diff(idx[np.searchsorted(idx, st):np.searchsorted(idx, st) + g]) == 1):
            phis.append(fn['phi'][st:st + g]); Ls.append(np.zeros((g, 4), np.int8)); ids.append(np.full(g, -1)); phs.append(fn['ph'][st:st + g])
        phis.append(f['phi'][lo:hi]); Ls.append(f['L'][lo:hi]); ids.append(np.full(hi - lo, n)); phs.append(f['ph'][lo:hi])
    return np.vstack(phis), np.vstack(Ls), np.concatenate(ids), np.vstack(phs)


def splice_metrics(S, A, n_rep=10, gammas=(0.0, 0.2, 0.4, 0.6, 0.8), seed=0):
    from sa import alert_events, correlate, segments as seg
    rng = np.random.default_rng(seed)
    res = {'none': np.zeros(3)}
    bc = {k: [[], []] for k in res}
    for r in range(n_rep):
        phi, L, bid, PH = splice_campaigns(S, rng)
        ev = alert_events(phi.sum(1))
        # truth = block id of the attack window the alert falls in (block includes the recovery tail)
        tr = []
        for j, (a, b) in enumerate(ev):
            ids = bid[a:b]; ids = ids[ids >= 0]
            blk = np.bincount(ids).argmax() if len(ids) else -1
            if blk >= 0 and L[a:b, 0].any() or (blk >= 0 and True):
                tr.append(blk)
            else:
                tr.append(-(j + 1))
        tr = np.array(tr)
        # alerts inside a block but before the attack starts are false alerts
        for j, (a, b) in enumerate(ev):
            if tr[j] >= 0:
                ids_rows = np.where(bid == tr[j])[0]
                a0 = ids_rows[0]; atk = np.where(L[ids_rows, 0] == 1)[0]
                if len(atk) == 0 or b <= a0 + atk[0]:
                    tr[j] = -(j + 1)
        combos = [(md, 0.6, dt) for md in ('temporal', 'graph', 'graph_phase') for dt in (30, 60, 120, 300, 600)]
        combos += [('graph_phase', g, 300) for g in (0.0, 0.2, 0.4, 0.8)] + [('graph', g, 300) for g in (0.0, 0.2, 0.4, 0.8)]
        for md, g, dt in combos:
            k = f'{md}_g{g}_dt{dt}'
            inc = correlate(ev, phi, A, mode=md, gamma=g, dt=dt, dt_small=0, ph=PH)
            res.setdefault(k, np.zeros(3)); bc.setdefault(k, [[], []])
            res[k] += pair_counts(inc, tr); p_, r_ = bcubed(inc, tr); bc[k][0] += p_; bc[k][1] += r_
        inc = np.arange(len(ev)); res['none'] += pair_counts(inc, tr); p_, r_ = bcubed(inc, tr); bc['none'][0] += p_; bc['none'][1] += r_
    out = {}
    for k, pc in res.items():
        P = pc[0] / max(pc[0] + pc[1], 1); R = pc[0] / max(pc[0] + pc[2], 1)
        BP, BR = np.mean(bc[k][0]), np.mean(bc[k][1])
        out[k] = dict(pair_p=float(P), pair_r=float(R), pair_f1=float(2 * P * R / max(P + R, 1e-12)),
                      bcubed_p=float(BP), bcubed_r=float(BR), bcubed_f1=float(2 * BP * BR / (BP + BR)))
    return out


if __name__ == '__main__':
    allres = {}
    try:
        allres = loadres(f'sa_{TAG}_{VER}')
    except Exception:
        pass
    for seed in SEEDS_:
        t0 = time.time()
        DET, files = load_run(TAG, seed, VER)
        AG = build_graph(DET)
        S = [situation(DET, f, AG, 'graph_phase') for f in files]
        tsa = time.time(); _ = [situation(DET, f, AG, 'graph_phase') for f in files[:1]]; tsa = time.time() - tsa
        r = dict(detection=detection_metrics(S), correlation=correlation_metrics(DET, files, AG),
                 state=state_metrics(S), impact=impact_metrics(S), projection=projection_metrics(S),
                 triage=triage_metrics(S), splice=splice_metrics(S, AG, seed=seed),
                 sa_update_ms_per_s=1000 * tsa / len(files[0]['phi']))
        allres[str(seed)] = r
        print(TAG, VER, 'seed', seed, round(time.time() - t0, 1), 's', flush=True)
    save(f'sa_{TAG}_{VER}', allres)
