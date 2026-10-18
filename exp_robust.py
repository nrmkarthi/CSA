"""RQ4/RQ5 robustness: zero-day, concept drift, adversarial evasion, component failures, efficiency."""
import numpy as np, sys, os, time, pickle, joblib
from core import *
from detect import LAG, ALPHA, Q
from sa import frame_features, SituationClassifier, segments, REC_WIN
from common import ewma

TAG = sys.argv[1] if len(sys.argv) > 1 else 'central'
WHAT = sys.argv[2].split(',') if len(sys.argv) > 2 else ['zeroday', 'adv', 'drift', 'fail', 'eff']
SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else SEEDS


def det_metrics(D, scores, preds=None):
    y = np.concatenate([L[:, 0] for X, L in D['tests']])
    s = np.concatenate(scores)
    pred = s > 1 if preds is None else np.concatenate(preds)
    m = point_metrics(y, pred, s)
    dets = []; off = 0
    for X, L in D['tests']:
        T = len(L); d, _ = event_metrics(L[:, 0], pred[off:off + T]); dets += list(d); off += T
    m['event_recall'] = float(np.mean(dets))
    return m


# ------------------------------------------------------------------ zero-day (leave-one-process-out)
def zeroday(D, det, files, seed):
    out = {}
    Fs = [frame_features(f['phi'], det.proc_of) for f in files]; Ls = [f['L'] for f in files]
    nfeat = len(det.feats); K = len(Fs)
    for p in range(3):
        rec = dict(anomaly_only=[], classifier_only=[], hybrid=[]); pts = dict(anomaly_only=[0, 0], classifier_only=[0, 0], hybrid=[0, 0])
        for k in range(K):
            clf = SituationClassifier(seed).fit([Fs[j] for j in range(K) if j != k], [Ls[j] for j in range(K) if j != k], nfeat, exclude_proc=p)
            ph, _ = clf.predict(Fs[k], nfeat)
            s = files[k]['phi'].sum(1); L = Ls[k]
            preds = dict(anomaly_only=s > 1, classifier_only=ph[:, 1] > 0.5, hybrid=(s > 1) & ~(ph[:, 2] > 0.5))
            for a, b in segments(L[:, 0]):
                if not L[a:b, p + 1].any():
                    continue
                for m, pr in preds.items():
                    rec[m].append(pr[a:b].any()); pts[m][0] += pr[a:b].sum(); pts[m][1] += b - a
        out[f'P{p + 1}'] = {m: dict(event_recall=float(np.mean(v)) if v else float('nan'), point_recall=float(pts[m][0] / max(pts[m][1], 1)), n=len(v))
                            for m, v in rec.items()}
    return out


# ------------------------------------------------------------------ adversarial evasion (FGSM on GCR, transfer to AE)
def pgd_perturb(det, X, L, eps, iters=100):
    """White-box evasion against the GCR component, transferred to the full detector.
    During attack windows the adversary adds a perturbation delta to the readings of the tags of the
    attacked sub-process(es), |delta_i| <= eps * sigma_i, chosen by projected gradient descent to minimise
    the GCR anomaly score sum_i r_i^2 (gradient through both the value and the lagged-difference features)."""
    p = det.p; ik = det.idx_keep; sd = det.norm.sd[ik]
    att = L[:, 0] == 1
    procs = np.zeros((len(X), 4), bool); procs[:, :3] = L[:, 1:4] == 1
    ctrl = procs[:, det.proc_of[ik]] & att[:, None]              # (T, p) attacker-controlled entries
    if not ctrl.any() or eps == 0:
        return X
    rows = np.where(ctrl.any(1))[0]
    lo = max(0, rows.min() - 20); hi = rows.max() + 1
    Xw = X[lo:hi].astype(np.float64); cw = ctrl[lo:hi]
    d = np.zeros((hi - lo, p))                                   # perturbation in z units
    Pc = det.gcr.Pc; sc = det.gcr.scale
    from detect import LAG
    Wm = Pc * sc[None, :]                                        # a -> standardized residuals
    Hz = (Wm[:p] + Wm[p:]) @ (Wm[:p] + Wm[p:]).T                 # curvature w.r.t. z (value + difference)
    eta = 1.0 / (2 * np.linalg.eigvalsh(Hz)[-1])                  # 1/L step for the quadratic objective
    for it in range(iters):
        Xp = Xw.copy(); Xp[:, ik] += d * sd
        A = det.norm.aug(Xp)
        r = A @ Wm
        g = 2 * r @ Wm.T                                         # d score / d a
        gz = g[:, :p] + g[:, p:]
        gz[:-LAG] -= g[LAG:, p:]                                 # effect through the lagged difference at t+LAG
        d = np.clip(d - eta * gz, -eps, eps) * cw                # projected gradient step
    Xo = X.copy(); Xo[lo:hi, ik] += (d * sd).astype(np.float32)
    return Xo


def adversarial(D, det, files, seed):
    out = {}
    Fs_clean = [frame_features(f['phi'], det.proc_of) for f in files]; Ls = [f['L'] for f in files]
    nfeat = len(det.feats); K = len(Fs_clean)
    clfs = [SituationClassifier(seed).fit([Fs_clean[j] for j in range(K) if j != k], [Ls[j] for j in range(K) if j != k], nfeat) for k in range(K)]
    det.disabled = {'ae', 'rule'}; th_g = np.quantile(det.score(D['cal']).sum(1), Q); det.disabled = set()
    for eps in (0.0, 1.0, 3.0, 10.0, 30.0):
        sc, hy, gc = [], [], []
        for k, (X, L) in enumerate(D['tests']):
            Xp = pgd_perturb(det, X, L, eps) if eps > 0 else X
            phi = det.score(Xp); s = phi.sum(1)
            ph, _ = clfs[k].predict(frame_features(phi, det.proc_of), nfeat)
            sc.append(s); hy.append((s > 1) & ~(ph[:, 2] > 0.5))
            det.disabled = {'ae', 'rule'}; gc.append(det.score(Xp).sum(1) / th_g); det.disabled = set()
        out[str(eps)] = dict(anomaly_only=det_metrics(D, sc), hybrid=det_metrics(D, sc, hy), gcr_target=det_metrics(D, gc))
    return out


# ------------------------------------------------------------------ concept drift
def inject_drift(X, det, rng, n_tags=6, mag=3.0):
    Xd = X.copy()
    cand = det.idx_keep
    tags = rng.choice(cand, n_tags, replace=False)
    ramp = np.linspace(0, 1, len(X))[:, None]
    Xd[:, tags] += (mag * det.norm.sd[tags][None, :] * ramp).astype(np.float32)
    return Xd, tags


def page_hinkley(x, delta=0.05, lam=50.0):
    """Two-sided Page-Hinkley test on a 1-D stream; returns first alarm index or -1."""
    m = 0.0; mt_p = mt_n = 0.0; Mp = Mn = 0.0; n = 0
    for i, v in enumerate(x):
        n += 1; m += (v - m) / n
        mt_p += v - m - delta; Mp = min(Mp, mt_p)
        mt_n += m - v - delta; Mn = min(Mn, mt_n)
        if mt_p - Mp > lam or mt_n - Mn > lam:
            return i
    return -1


def adaptive_score(det, X, chunk=60, persist=60, elev=1.0, rate=0.1):
    """Streaming scoring with drift monitoring and controlled per-tag bias compensation.
    For every tag the chunk-mean residual is computed over seconds in which the REST of the plant is
    quiet (leave-tag-out score < 0.5).  A tag is declared drifting only after its residual has been
    elevated (|mean| > elev sigma) for `persist` consecutive chunks (CUSUM-style persistence test, 1 h by
    default, longer than any labelled attack); only then is a bias tracked and subtracted.  Every
    declaration is written to an audit log."""
    T = len(X); p = det.p
    A = det.norm.aug(X)
    r_g = det.gcr.resid(A)
    r_a = (A[:, :p] - det.ae.predict(A)) / det.ae_sd
    rule = det.norm.rule_viol(X) * det.rule_on[None, :]
    bg = np.zeros(p); ba = np.zeros(p)
    count = np.zeros(p, int); flagged = np.zeros(p, bool)
    audit = []
    phi_raw = np.zeros((T, len(det.feats)))
    ik = det.idx_keep
    for c0 in range(0, T, chunk):
        c1 = min(T, c0 + chunk)
        rc = dict(gcr=(r_g[c0:c1] - bg) ** 2, ae=(r_a[c0:c1] - ba) ** 2, rule=rule[c0:c1])
        ph = det._phi(rc, smooth=False) / det.theta_f
        phi_raw[c0:c1] = ph
        s = ph.sum(1)
        ok = np.full(p, (c1 - c0) >= chunk // 2)
        mg = np.median(r_g[c0:c1], 0); ma = np.median(r_a[c0:c1], 0)
        el = ok & (np.abs(mg - bg) > elev)
        count = np.where(el, count + 1, np.where(ok, 0, count))
        newf = (count >= persist) & ~flagged
        for i in np.where(newf)[0]:
            audit.append((c0, int(ik[i])))
        flagged |= newf
        # controlled update: only while the plant, excluding already-flagged tags, is not alerting
        s_excl = s - ph[:, ik[flagged]].sum(1) if flagged.any() else s
        upd = flagged & ok & ((s_excl > 1).mean() < 0.1)
        bg = np.where(upd, (1 - rate) * bg + rate * mg, bg)
        ba = np.where(upd, (1 - rate) * ba + rate * ma, ba)
    phi = ewma(phi_raw, ALPHA)
    return phi, audit


def drift(D, det, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for mag in (1.0, 3.0, 5.0):
        sc_clean, sc_d, sc_a, tags_all, audits = [], [], [], [], []
        for X, L in D['tests']:
            Xd, tags = inject_drift(X, det, rng, mag=mag)
            sc_d.append(det.score(Xd).sum(1))
            phi, audit = adaptive_score(det, Xd); sc_a.append(phi.sum(1))
            tags_all.append(set(tags.tolist())); audits.append(audit)
        flagged = [set(t for _, t in a) for a in audits]
        tp = sum(len(f & t) for f, t in zip(flagged, tags_all)); fl = sum(len(f) for f in flagged)
        out[str(mag)] = dict(no_adaptation=det_metrics(D, sc_d), drift_aware=det_metrics(D, sc_a),
                             drift_tag_precision=tp / max(fl, 1), drift_tag_recall=tp / sum(len(t) for t in tags_all),
                             n_updates=int(fl))
    # no injected drift: does the adaptive mechanism hurt on clean (or naturally drifting) data?
    sc_a = [adaptive_score(det, X)[0].sum(1) for X, L in D['tests']]
    out['clean'] = dict(no_adaptation=det_metrics(D, [det.score(X).sum(1) for X, L in D['tests']]), drift_aware=det_metrics(D, sc_a))
    return out


# ------------------------------------------------------------------ component failures
def failures(D, det, seed):
    rng = np.random.default_rng(seed)
    out = {}
    out['nominal'] = det_metrics(D, [det.score(X).sum(1) for X, L in D['tests']])
    for comp in ('ae', 'gcr'):
        det.disabled = {comp}
        # graceful degradation: fallback threshold pre-calibrated on held-out normal data for the
        # remaining component(s)
        th = np.quantile(det.score(D['cal']).sum(1), Q)
        out[f'{comp}_node_down'] = det_metrics(D, [det.score(X).sum(1) / th for X, L in D['tests']])
        out[f'{comp}_node_down_nofallback'] = det_metrics(D, [det.score(X).sum(1) for X, L in D['tests']])
        det.disabled = set()
    # sensor dropout: a random fraction of tags stops reporting for the whole file
    from xai import _cond_impute_aug
    for frac in (0.1, 0.2, 0.3):
        st_s, im_s = [], []
        for X, L in D['tests']:
            kp = rng.choice(det.p, int(frac * det.p), replace=False)
            tags = det.idx_keep[kp]
            Xs = X.copy(); Xs[:, tags] = X[0, tags]            # stuck-at (last value before failure)
            st_s.append(det.score(Xs).sum(1))
            # graceful degradation: missing tags imputed from the Gaussian conditional model; their own
            # residual terms are excluded and the remaining terms rescaled
            A = det.norm.aug(X); A2 = _cond_impute_aug(det, A, kp)
            rc = det.raw_contrib(X, A2)
            for k in ('gcr', 'ae'):
                rc[k][:, kp] = 0; rc[k] *= det.p / (det.p - len(kp))
            im_s.append((det._phi(rc) / det.theta_f).sum(1))
        out[f'dropout{frac}_stuck'] = det_metrics(D, st_s)
        out[f'dropout{frac}_imputed'] = det_metrics(D, im_s)
    return out


# ------------------------------------------------------------------ efficiency
def efficiency(D, det, files, seed):
    import tracemalloc, resource
    X, L = D['tests'][0]
    out = {}
    # batch=1 edge latency of each component (median over 500 calls, 11-row window for the lag feature)
    rows = np.random.default_rng(seed).integers(LAG + 1, len(X), 500)
    def timeit(fn):
        ts = []
        for r in rows:
            t = time.perf_counter(); fn(X[r - LAG:r + 1]); ts.append(time.perf_counter() - t)
        return float(np.median(ts) * 1000)
    A0 = det.norm.aug(X[:LAG + 1])
    out['lat_ms_normalise'] = timeit(lambda w: det.norm.aug(w))
    out['lat_ms_gcr'] = timeit(lambda w: det.gcr.resid(det.norm.aug(w)[-1:]))
    out['lat_ms_ae'] = timeit(lambda w: det.ae.predict(det.norm.aug(w)[-1:]))
    out['lat_ms_full_detector'] = timeit(lambda w: det.raw_contrib(w))
    # classifier latency
    F = frame_features(files[0]['phi'][:2000], det.proc_of)
    clf = SituationClassifier(seed).fit([frame_features(f['phi'], det.proc_of) for f in files[1:]], [f['L'] for f in files[1:]], len(det.feats))
    ts = []
    for i in range(200):
        t = time.perf_counter(); clf.predict(F[i:i + 1], len(det.feats)); ts.append(time.perf_counter() - t)
    out['lat_ms_classifier'] = float(np.median(ts) * 1000)
    # throughput (batch processing of a full file) and CPU utilisation
    c0 = time.process_time(); w0 = time.perf_counter(); det.score(X); c1 = time.process_time(); w1 = time.perf_counter()
    out['batch_throughput_samples_per_s'] = len(X) / (w1 - w0)
    out['cpu_util_batch'] = (c1 - c0) / (w1 - w0)
    # streaming at 1 Hz real time: CPU fraction = per-sample cost / 1 s
    out['cpu_fraction_at_1Hz'] = (out['lat_ms_full_detector'] + out['lat_ms_classifier']) / 1000
    # memory footprint of models
    out['model_bytes_detector'] = len(pickle.dumps(det))
    out['model_bytes_ae'] = len(pickle.dumps(det.ae))
    out['model_bytes_gcr'] = int(det.gcr.P.nbytes)
    out['model_bytes_classifier'] = len(pickle.dumps(clf))
    out['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return out


if __name__ == '__main__':
    VER = os.environ.get('VER', 'hai-21.03')
    D = dataset(VER)
    SUF = ('' if VER == 'hai-21.03' else '_' + VER) + os.environ.get('OUTSUF', '')
    out = {}
    try:
        out = loadres(f'robust_{TAG}{SUF}')
    except Exception:
        pass
    for seed in SEEDS_:
        det, files = load_run(TAG, seed, VER)
        r = out.get(str(seed), {})
        t0 = time.time()
        if 'zeroday' in WHAT: r['zeroday'] = zeroday(D, det, files, seed)
        if 'adv' in WHAT: r['adversarial'] = adversarial(D, det, files, seed)
        if 'drift' in WHAT: r['drift'] = drift(D, det, seed)
        if 'fail' in WHAT: r['failures'] = failures(D, det, seed)
        if 'eff' in WHAT and seed == SEEDS_[0]: r['efficiency'] = efficiency(D, det, files, seed)
        out[str(seed)] = r
        print('robust', TAG, seed, WHAT, round(time.time() - t0), 's', flush=True)
        save(f'robust_{TAG}{SUF}', out)
