"""RQ2: explanation fidelity, stability, consistency, localisation and situation-grounded correctness."""
import numpy as np, sys, os, time
from core import *
from detect import LAG
from sa import proc_agg, ref_states, segments
from xai import (score_from_aug, _cond_impute_aug, shapley_sampling, topk, jacc, spearman)

TAG = sys.argv[1] if len(sys.argv) > 1 else 'central'
SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else SEEDS


def window(X, t):
    lo = max(0, t - LAG)
    return X[lo:t + 1]


def raw_attr(det, Xw):
    rc = det.raw_contrib(Xw)
    return det._phi(rc, smooth=False)[-1] / det.theta_f


def comp_attr(det, Xw, comp):
    old = det.disabled; det.disabled = {'gcr', 'ae', 'rule'} - {comp}
    a = raw_attr(det, Xw); det.disabled = old
    return a


def magnitude_attr(det, Xw):
    z = np.zeros(len(det.feats))
    z[det.idx_keep] = np.abs(det.norm.z(Xw[-1:])[0])
    z[det.idx_const] = 10.0 * det.norm.rule_viol(Xw[-1:])[0]
    return z


def delete_score(det, Xw, attr, k, mode):
    Aw = det.norm.aug(Xw)[-1:].copy(); Xr = Xw[-1:].copy()
    keep_pos = -np.ones(len(det.feats), int); keep_pos[det.idx_keep] = np.arange(det.p)
    top = np.argsort(-attr)[:k]
    for t in top:
        if det.norm.const[t]:
            Xr[0, t] = det.norm.mu[t]
    kt = np.array([keep_pos[t] for t in top if keep_pos[t] >= 0], int)
    if len(kt):
        if mode == 'cond':
            Aw = _cond_impute_aug(det, Aw, kt)
        else:
            Aw[:, kt] = 0; Aw[:, kt + det.p] = 0
    return score_from_aug(det, Xr, Aw)[0]


def run(seed):
    D = dataset()
    det, files = load_run(TAG, seed)
    A = build_graph(det)
    rng = np.random.default_rng(seed)
    S = [situation(det, f, A, 'graph_phase') for f in files]
    nf = len(det.feats)
    # ---------------- sample explanation points: first alert, +30 s, +60 s of each detected episode
    pts = []
    for k, f in enumerate(S):
        for a, b in segments(f['L'][:, 0]):
            hit = np.where(f['s'][a:b] > 1)[0]
            if len(hit):
                for d in (0, 30, 60):
                    t = a + hit[0] + d
                    if t < len(f['s']) and f['s'][t] > 1:
                        pts.append((k, t))
    res = dict(n_points=len(pts))
    methods = ['residual', 'shapley', 'magnitude', 'random']
    fid = {m: {f'{mode}_k{k}': [] for mode in ('cond', 'mean') for k in (1, 3, 5, 10)} for m in methods}
    stab = {m: [] for m in ('residual', 'shapley', 'magnitude')}
    cons = dict(res_vs_shap_spearman=[], res_vs_shap_top5=[], gcr_vs_ae_top5=[], gcr_vs_ae_spearman=[])
    t0 = time.time(); shap_time = 0
    for (k, t) in pts:
        X = D['tests'][k][0]; Xw = window(X, t)
        base = score_from_aug(det, Xw[-1:], det.norm.aug(Xw)[-1:])[0]
        ts = time.time()
        sh = shapley_sampling(det, Xw, [len(Xw) - 1], M=16, seed=seed)[0]; shap_time += time.time() - ts
        at = dict(residual=raw_attr(det, Xw), shapley=sh, magnitude=magnitude_attr(det, Xw), random=rng.random(nf))
        for m in methods:
            for mode in ('cond', 'mean'):
                for kk in (1, 3, 5, 10):
                    fid[m][f'{mode}_k{kk}'].append((base - delete_score(det, Xw, at[m], kk, mode)) / (base + 1e-9))
        # perturbation stability (Gaussian noise of 5 % of each tag's normal std on the current row)
        Xp = Xw.copy(); Xp[-1] = Xp[-1] + rng.normal(0, 0.05, nf) * np.where(det.norm.keep, det.norm.sd, 0)
        pt = dict(residual=raw_attr(det, Xp), shapley=shapley_sampling(det, Xp, [len(Xp) - 1], M=16, seed=seed)[0],
                  magnitude=magnitude_attr(det, Xp))
        for m in stab:
            stab[m].append(jacc(topk(at[m]), topk(pt[m])))
        cons['res_vs_shap_spearman'].append(spearman(at['residual'], sh)); cons['res_vs_shap_top5'].append(jacc(topk(at['residual']), topk(sh)))
        ga, aa = comp_attr(det, Xw, 'gcr'), comp_attr(det, Xw, 'ae')
        cons['gcr_vs_ae_top5'].append(jacc(topk(ga), topk(aa))); cons['gcr_vs_ae_spearman'].append(spearman(ga, aa))
    res['fidelity'] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in fid.items()}
    res['perturbation_stability_top5'] = {m: float(np.mean(v)) for m, v in stab.items()}
    res['consistency'] = {k: float(np.mean(v)) for k, v in cons.items()}
    res['shapley_ms_per_explanation'] = 1000 * shap_time / max(len(pts), 1)
    # ---------------- temporal stability within incidents (instant vs smoothed vs grounded)
    tstab = dict(instant=[], smoothed=[], grounded=[])
    from xai import grounded_attr
    for k, f in enumerate(S):
        X = D['tests'][k][0]
        al = np.where((f['s'] > 1) & (f['open_id'] >= 0))[0]
        al = al[(al + 10 < len(f['s']))]
        al = al[f['open_id'][al] == f['open_id'][al + 10]]
        if len(al) == 0:
            continue
        sel = rng.choice(al, min(200, len(al)), replace=False)
        g = grounded_attr(f['phi'], f['open_id'], A)
        for t in sel:
            r1, r2 = raw_attr(det, window(X, t)), raw_attr(det, window(X, t + 10))
            tstab['instant'].append(jacc(topk(r1), topk(r2)))
            tstab['smoothed'].append(jacc(topk(f['phi'][t]), topk(f['phi'][t + 10])))
            tstab['grounded'].append(jacc(topk(g[t]), topk(g[t + 10])))
    res['temporal_stability_top5'] = {k: float(np.mean(v)) for k, v in tstab.items()}
    # ---------------- process localisation per detected episode
    loc = {m: [] for m in ('instant', 'smoothed', 'grounded', 'classifier', 'consensus')}
    for k, f in enumerate(S):
        X = D['tests'][k][0]
        for a, b in segments(f['L'][:, 0]):
            true = set(p for p in range(3) if f['L'][a:b, p + 1].any())
            hit = np.where(f['s'][a:b] > 1)[0]
            if not len(hit):
                continue
            t = min(a + hit[0] + 30, len(f['s']) - 1)
            ri = proc_agg(raw_attr(det, window(X, t))[None], det.proc_of)[0, :3]
            sm = proc_agg(f['phi'][t:t + 1], det.proc_of)[0, :3]
            est = dict(instant=int(np.argmax(ri)), smoothed=int(np.argmax(sm)), grounded=int(np.argmax(f['gs'][t])),
                       classifier=int(np.argmax(f['tg'][t])), consensus=int(np.argmax(0.5 * f['tg'][t] + 0.5 * f['gs'][t])))
            for m, e in est.items():
                loc[m].append(e in true)
    res['localisation_at_30s'] = {m: float(np.mean(v)) for m, v in loc.items()}
    res['n_episodes_localised'] = len(loc['instant'])
    # ---------------- situation-grounded explanation correctness at decision points
    corr = dict(feature_only=[], grounded=[])
    for f in S:
        ref = ref_states(f['L'])
        T3 = f['L'][:, 1:4].astype(bool)
        pts2 = np.arange(0, len(f['s']), 10); pts2 = pts2[f['open_id'][pts2] >= 0]
        pp = proc_agg(f['phi'], det.proc_of)[:, :3]
        for t in pts2:
            # feature-only explanation: "attack on argmax-attribution process" whenever the score alerts
            st_f = 1 if f['s'][t] > 1 else 0; tg_f = int(np.argmax(pp[t]))
            st_g = int(np.argmax(f['ph'][t])); tg_g = int(np.argmax(0.5 * f['tg'][t] + 0.5 * f['gs'][t]))
            for name, stt, tgt in (('feature_only', st_f, tg_f), ('grounded', st_g, tg_g)):
                ok = stt == ref[t] and (ref[t] != 1 or T3[t, tgt])
                corr[name].append(ok)
    res['explanation_situation_correctness'] = {k: float(np.mean(v)) for k, v in corr.items()}
    res['time_s'] = time.time() - t0
    return res


if __name__ == '__main__':
    out = {}
    try:
        out = loadres(f'xai_{TAG}')
    except Exception:
        pass
    for s in SEEDS_:
        out[str(s)] = run(s); print('xai seed', s, round(out[str(s)]['time_s']), 's', flush=True)
        save(f'xai_{TAG}', out)
