"""Essential ablation study (7 configurations) with identical splits, seeds and model capacity."""
import numpy as np, sys, os, time
from sklearn.metrics import f1_score
from core import *
from sa import ref_states, proc_agg, segments
from exp_response import evaluate, BASE

SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else SEEDS
MAIN = os.environ.get('MAIN_TAG', 'fedavg')

CONFIGS = {
    '1 Detection only':               dict(tag=MAIN, mode='alert', use_phase=False, target='clf', policy='static', verify=False, safe=False, det='anomaly', loc='instant'),
    '2 + Correlation & SA':           dict(tag=MAIN, mode='graph_phase', use_phase=True, target='clf', policy='static', verify=False, safe=False, det='hybrid', loc='classifier'),
    '3 SA + XAI':                     dict(tag=MAIN, mode='graph_phase', use_phase=True, target='consensus', policy='static', verify=False, safe=False, det='hybrid', loc='consensus'),
    '4 SA + adaptive response':       dict(tag=MAIN, mode='graph_phase', use_phase=True, target='clf', policy='safe_rl', verify=True, safe=True, det='hybrid', loc='classifier'),
    '5 Full w/o federated learning':  dict(tag='local', mode='graph_phase', use_phase=True, target='consensus', policy='safe_rl', verify=True, safe=True, det='hybrid', loc='consensus'),
    '6 Full w/o response verification': dict(tag=MAIN, mode='graph_phase', use_phase=True, target='consensus', policy='safe_rl', verify=False, safe=True, det='hybrid', loc='consensus'),
    '7 Full XAI-CSA':                 dict(tag=MAIN, mode='graph_phase', use_phase=True, target='consensus', policy='safe_rl', verify=True, safe=True, det='hybrid', loc='consensus'),
}


def metrics_for(S, det, cfg):
    y = np.concatenate([f['L'][:, 0] for f in S]).astype(bool)
    s = np.concatenate([f['s'] for f in S]); ph = np.vstack([f['ph'] for f in S])
    pred = s > 1 if cfg['det'] == 'anomaly' else (s > 1) & ~(ph[:, 2] > 0.5)
    sc = s if cfg['det'] == 'anomaly' else s * (1 - ph[:, 2])
    m = point_metrics(y, pred, sc)
    dets = []; off = 0
    for f in S:
        T = len(f['s']); d, _ = event_metrics(f['L'][:, 0], pred[off:off + T]); dets += list(d); off += T
    m['event_recall'] = float(np.mean(dets))
    ref = np.concatenate([ref_states(f['L']) for f in S]); op = np.concatenate([f['open_id'] for f in S])
    est = (s > 1).astype(int) if cfg['mode'] == 'alert' else np.where(op >= 0, np.argmax(ph, 1), 0)
    m['state_acc'] = float((est == ref).mean()); m['state_macro_f1'] = float(f1_score(ref, est, average='macro'))
    loc = []
    for f in S:
        for a, b in segments(f['L'][:, 0]):
            true = set(p for p in range(3) if f['L'][a:b, p + 1].any())
            hit = np.where(f['s'][a:b] > 1)[0]
            if not len(hit):
                continue
            t = min(a + hit[0] + 30, len(f['s']) - 1)
            if cfg['loc'] == 'instant':
                e = int(np.argmax(proc_agg(f['phi'][t:t + 1], det.proc_of)[0, :3]))
            elif cfg['loc'] == 'classifier':
                e = int(np.argmax(f['tg'][t]))
            else:
                e = int(np.argmax(0.5 * f['tg'][t] + 0.5 * f['gs'][t]))
            loc.append(e in true)
    m['localisation'] = float(np.mean(loc))
    m['incidents_per_day'] = float(sum(len(set(f['inc'].tolist())) for f in S) / (len(s) / 3600) * 24)
    return m


if __name__ == '__main__':
    out = {}
    try:
        out = loadres('ablation')
    except Exception:
        pass
    runs = [(s, s) for s in SEEDS_]
    for seed in SEEDS_:
        t0 = time.time()
        r = out.get(str(seed), {})
        for name, c in CONFIGS.items():
            tag = c['tag'] if c['tag'] != 'local' else f'local{seed}'
            det, files = load_run(tag, seed)
            A = build_graph(det)
            S = [situation(det, f, A, c['mode']) for f in files]
            cfg = dict(BASE); cfg.update({k: c[k] for k in ('use_phase', 'target', 'policy', 'verify', 'safe')})
            m = metrics_for(S, det, c)
            m.update(evaluate(S, cfg, seed))
            r[name] = m
            print(seed, name, {k: round(v, 3) for k, v in m.items() if k in ('f1', 'state_macro_f1', 'localisation', 'msr', 'uar', 'disruption_h_per_day')}, flush=True)
        out[str(seed)] = r
        save('ablation', out)
        print('ablation seed', seed, round(time.time() - t0), 's', flush=True)
