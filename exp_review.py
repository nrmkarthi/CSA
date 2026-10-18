"""Analyses requested in the review:
  fp      - where the hybrid detector's false-positive reduction comes from (suppression ablation)
  etapr   - eTaPR (HAI-standard metric) for the XAI-CSA detectors
  hmm     - causal HMM filtering of the phase posterior + incident-level target accumulation
  cshap   - Shapley values with a conditional (on-manifold) value function, matched deletion protocols
  fusion  - GCR/AE fusion-weight sweep under identical calibration, clean and under evasion
"""
import numpy as np, sys, os, time, joblib
from sklearn.metrics import f1_score
from common import *
from core import load_run, situation, build_graph
from sa import ref_states, segments, proc_agg
from detect import Q, ALPHA, LAG
from etapr_util import etapr_scores

WHAT = sys.argv[1].split(',')
TAG = os.environ.get('TAG', 'fedavg')
SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else SEEDS


def res_name(part):
    return f'review_{part}_{TAG}'


# ------------------------------------------------------------------ (a) false-positive decomposition
def fp_analysis(files):
    y = np.concatenate([f['L'][:, 0] for f in files]).astype(bool)
    st = np.concatenate([ref_states(f['L']) for f in files])
    s = np.concatenate([f['phi'].sum(1) for f in files]); ph = np.vstack([f['ph'] for f in files])
    a = s > 1
    variants = {'anomaly_only': a,
                'suppress_recovery (XAI-CSA)': a & ~(ph[:, 2] > 0.5),
                'suppress_normal': a & ~(ph[:, 0] > 0.5),
                'suppress_recovery_or_normal': a & (ph[:, 1] >= np.maximum(ph[:, 0], ph[:, 2]))}
    out = {}
    for k, p in variants.items():
        m = point_metrics(y, p, None)
        fp = p & ~y
        m['fp_seconds'] = int(fp.sum()); m['fp_in_recovery_window'] = float((fp & (st == 2)).sum() / max(fp.sum(), 1))
        dets = []; off = 0
        for f in files:
            T_ = len(f['L']); d, _ = event_metrics(f['L'][:, 0], p[off:off + T_]); dets += list(d); off += T_
        m['event_recall'] = float(np.mean(dets))
        off = 0; ys, ps = [], []
        for f in files:
            T_ = len(f['L']); ys.append(f['L'][:, 0]); ps.append(p[off:off + T_]); off += T_
        m.update(etapr_scores(ys, ps))
        out[k] = m
    fpa = a & ~y; rem = fpa & (ph[:, 2] > 0.5)
    out['removed_fp_share_in_recovery'] = float((rem & (st == 2)).sum() / max(rem.sum(), 1))
    out['removed_fp_fraction'] = float(rem.sum() / max(fpa.sum(), 1))
    lost = a & y & (ph[:, 2] > 0.5)
    out['lost_tp_fraction'] = float(lost.sum() / max((a & y).sum(), 1))
    return out


# ------------------------------------------------------------------ (b) HMM phase filtering
def trans_matrix(Ls, eps=1.0):
    C = np.full((3, 3), eps)
    for L in Ls:
        s = ref_states(L); np.add.at(C, (s[:-1], s[1:]), 1)
    return C / C.sum(1, keepdims=True)


def forward_filter(P, op, A, prior, kappa):
    """Causal HMM filter with scaled likelihoods (P / prior)^kappa, restarted at every incident opening;
    outside incidents the state is 'normal'."""
    T_ = len(P); out = np.zeros(T_, int); alpha = np.array([1.0, 0, 0]); prev = -1
    E = (P / prior[None, :] + 1e-12) ** kappa
    for t in range(T_):
        if op[t] < 0:
            out[t] = 0; prev = -1; continue
        if op[t] != prev:
            alpha = np.array([0.5, 0.5, 0.0]); prev = op[t]
        alpha = (alpha @ A) * E[t]; alpha /= alpha.sum()
        out[t] = int(np.argmax(alpha))
    return out


def accumulated_target(f):
    """Incident-level running mean of the fused target estimate (affected-asset estimation)."""
    c = 0.5 * f['tg'] + 0.5 * f['gs']; op = f['open_id']
    out = np.zeros_like(c); acc = np.zeros(3); n = 0; prev = -1
    for t in range(len(c)):
        if op[t] < 0:
            prev = -1; continue
        if op[t] != prev:
            acc[:] = 0; n = 0; prev = op[t]
        acc += c[t]; n += 1; out[t] = acc / n
    return out


def state_eval(S, est_list, tset_list):
    ref = np.concatenate([ref_states(f['L']) for f in S]); est = np.concatenate(est_list)
    T3 = np.vstack([f['L'][:, 1:4] for f in S]).astype(bool); ts = np.vstack(tset_list)
    act = ref == 1
    return dict(acc=float((est == ref).mean()), macro_f1=float(f1_score(ref, est, average='macro')),
                active_recall=float((est[act] == 1).mean()),
                fullstate_active=float(((est[act] == 1) & (ts[act] == T3[act]).all(1)).mean()))


def hmm_analysis(det, files):
    A_g = build_graph(det)
    S = [situation(det, f, A_g, 'graph_phase') for f in files]
    K = len(S); out = {}
    base_est = [np.where(f['open_id'] >= 0, np.argmax(f['ph'], 1), 0) for f in S]
    inst_t = [(0.5 * f['tg'] + 0.5 * f['gs']) > 0.5 for f in S]
    acc_t = [accumulated_target(f) > 0.5 for f in S]
    out['argmax_instant_target'] = state_eval(S, base_est, inst_t)
    out['argmax_accumulated_target'] = state_eval(S, base_est, acc_t)
    # HMM with LOFO-estimated transitions, kappa chosen on the training folds (never on the evaluated file)
    kappas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    est_h, chosen = [], []
    for k in range(K):
        tr = [j for j in range(K) if j != k]
        A = trans_matrix([S[j]['L'] for j in tr])
        prior = np.bincount(np.concatenate([ref_states(S[j]['L']) for j in tr]), minlength=3) + 1.0; prior /= prior.sum()
        best = None
        for kp in kappas:
            e = [forward_filter(S[j]['ph'], S[j]['open_id'], A, prior, kp) for j in tr]
            sc = state_eval([S[j] for j in tr], e, [acc_t[j] for j in tr])['fullstate_active'] + \
                 state_eval([S[j] for j in tr], e, [acc_t[j] for j in tr])['macro_f1']
            if best is None or sc > best[0]:
                best = (sc, kp)
        chosen.append(best[1])
        est_h.append(forward_filter(S[k]['ph'], S[k]['open_id'], A, prior, best[1]))
    out['hmm_accumulated_target'] = state_eval(S, est_h, acc_t)
    out['hmm_instant_target'] = state_eval(S, est_h, inst_t)
    out['hmm_kappa_chosen'] = float(np.mean(chosen))
    out['detection_only'] = state_eval(S, [(f['s'] > 1).astype(int) for f in S],
                                       [np.eye(3, dtype=bool)[np.argmax(proc_agg(f['phi'], det.proc_of)[:, :3], 1)] for f in S])
    return out


# ------------------------------------------------------------------ (c) conditional Shapley
def cond_value_chain(det, Xr, Ar, perm, keep_pos):
    """Values of the coalition chain (empty -> all tags) for one permutation, with absent tags imputed
    from their Gaussian conditional expectation given the present tags (on-manifold value function)."""
    from xai import score_from_aug
    p = det.p; P = det.gcr.P; nf = len(perm)
    rows_A = np.zeros((nf + 1, 2 * p), np.float32); rows_X = np.repeat(det.norm.mu[None, :].astype(np.float32), nf + 1, 0)
    present = np.zeros(nf, bool)
    for i in range(nf + 1):
        if i > 0:
            present[perm[i - 1]] = True
        pres_k = np.array([keep_pos[t] for t in np.where(present)[0] if keep_pos[t] >= 0], int)
        dims_p = np.r_[pres_k, pres_k + p] if len(pres_k) else np.array([], int)
        dims_a = np.setdiff1d(np.arange(2 * p), dims_p)
        a = Ar[0].astype(np.float64).copy()
        if len(dims_a):
            if len(dims_p):
                a[dims_a] = -np.linalg.solve(P[np.ix_(dims_a, dims_a)], P[np.ix_(dims_a, dims_p)] @ a[dims_p])
            else:
                a[dims_a] = 0.0
        rows_A[i] = a
        for t in np.where(present)[0]:
            if keep_pos[t] < 0:
                rows_X[i, t] = Xr[0, t]
    return score_from_aug(det, rows_X, rows_A)


def cshap_analysis(det, files, seed, M=8):
    from exp_xai import window, raw_attr, delete_score
    from xai import shapley_sampling, topk, jacc, spearman
    D = dataset()
    rng = np.random.default_rng(seed)
    pts = []
    for k, f in enumerate(files):
        s = f['phi'].sum(1)
        for a, b in segments(f['L'][:, 0]):
            hit = np.where(s[a:b] > 1)[0]
            if len(hit):
                pts.append((k, a + hit[0] + 30 if a + hit[0] + 30 < b else a + hit[0]))
    nf = len(det.feats); keep_pos = -np.ones(nf, int); keep_pos[det.idx_keep] = np.arange(det.p)
    res = {m: {f'{mode}_k{kk}': [] for mode in ('cond', 'mean') for kk in (1, 5)} for m in ('residual', 'shap_mean', 'shap_cond')}
    rho = dict(res_vs_shapmean=[], res_vs_shapcond=[], shapmean_vs_shapcond=[]); stab = []
    t0 = time.time()
    for (k, t) in pts:
        X = D['tests'][k][0]; Xw = window(X, t)
        Ar = det.norm.aug(Xw)[-1:]; Xr = Xw[-1:]
        from xai import score_from_aug
        base = score_from_aug(det, Xr, Ar)[0]
        sc = np.zeros(nf)
        for m in range(M):
            perm = rng.permutation(nf); v = cond_value_chain(det, Xr, Ar, perm, keep_pos); sc[perm] += np.diff(v)
        sc /= M
        at = dict(residual=raw_attr(det, Xw), shap_mean=shapley_sampling(det, Xw, [len(Xw) - 1], M=16, seed=seed)[0], shap_cond=sc)
        for m in at:
            for mode in ('cond', 'mean'):
                for kk in (1, 5):
                    res[m][f'{mode}_k{kk}'].append((base - delete_score(det, Xw, at[m], kk, mode)) / (base + 1e-9))
        rho['res_vs_shapmean'].append(spearman(at['residual'], at['shap_mean'])); rho['res_vs_shapcond'].append(spearman(at['residual'], sc))
        rho['shapmean_vs_shapcond'].append(spearman(at['shap_mean'], sc))
        # efficiency/completeness check of the conditional Shapley values: sum phi = v(N) - v(empty)
        stab.append(abs(sc.sum() - (base - cond_value_chain(det, Xr, Ar, np.arange(nf), keep_pos)[0])) / (abs(base) + 1e-9))
    out = {m: {k2: float(np.mean(v)) for k2, v in d.items()} for m, d in res.items()}
    out['spearman'] = {k2: float(np.mean(v)) for k2, v in rho.items()}
    out['completeness_rel_error'] = float(np.mean(stab)); out['n_points'] = len(pts); out['sec_per_point'] = (time.time() - t0) / len(pts)
    return out


# ------------------------------------------------------------------ (d) fusion-weight sweep
def fusion_analysis(det, seed, eps=3.0):
    from exp_robust import pgd_perturb
    D = dataset()
    rc_cal = det.raw_contrib(D['cal'])
    rc_t = [det.raw_contrib(X) for X, L in D['tests']]
    rc_p = [det.raw_contrib(pgd_perturb(det, X, L, eps)) for X, L in D['tests']]
    ys = [L[:, 0] for X, L in D['tests']]
    def fused(rc, w):
        c = (w * rc['gcr'] / det.theta['gcr'] + (1 - w) * rc['ae'] / det.theta['ae']).sum(1) / det.p
        c = c + 5.0 * rc['rule'].sum(1)
        return ewma(c, ALPHA)
    out = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        thr = np.quantile(fused(rc_cal, w), Q)
        s = [fused(r, w) / thr for r in rc_t]; sp = [fused(r, w) / thr for r in rc_p]
        y = np.concatenate(ys); sc = np.concatenate(s); scp = np.concatenate(sp)
        m = point_metrics(y, sc > 1, sc)
        dets = []
        for yy, ss in zip(ys, s):
            d, _ = event_metrics(yy, ss > 1); dets += list(d)
        m['event_recall'] = float(np.mean(dets)); m.update(etapr_scores(ys, [x > 1 for x in s]))
        dp = []
        for yy, ss in zip(ys, sp):
            d, _ = event_metrics(yy, ss > 1); dp += list(d)
        m['evasion_recall'] = float(((scp > 1) & (y == 1)).sum() / (y == 1).sum()); m['evasion_event_recall'] = float(np.mean(dp))
        out[str(w)] = m
    return out


if __name__ == '__main__':
    for part in WHAT:
        out = {}
        try:
            out = loadres(res_name(part))
        except Exception:
            pass
        for seed in SEEDS_:
            t0 = time.time()
            det, files = load_run(TAG, seed)
            if part == 'fp':
                out[str(seed)] = fp_analysis(files)
            elif part == 'hmm':
                out[str(seed)] = hmm_analysis(det, files)
            elif part == 'cshap':
                out[str(seed)] = cshap_analysis(det, files, seed)
            elif part == 'fusion':
                out[str(seed)] = fusion_analysis(det, seed)
            save(res_name(part), out)
            print(part, TAG, seed, round(time.time() - t0), 's', flush=True)
