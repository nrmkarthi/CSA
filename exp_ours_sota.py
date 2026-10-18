"""Oracle-threshold F1 and eTaPR for the XAI-CSA detectors (from cached out-of-fold outputs)."""
import numpy as np
from common import *
from core import load_run
from etapr_util import etapr_scores
out = {}
for tag in ('central', 'fedavg'):
    for seed in SEEDS:
        det, files = load_run(tag, seed)
        y = np.concatenate([f['L'][:, 0] for f in files])
        r = {}
        for k in ('anomaly', 'hybrid'):
            sc = [f['phi'].sum(1) * (1 if k == 'anomaly' else (1 - f['ph'][:, 2])) for f in files]
            pr = [f['phi'].sum(1) > 1 if k == 'anomaly' else (f['phi'].sum(1) > 1) & ~(f['ph'][:, 2] > 0.5) for f in files]
            m = point_metrics(y, np.concatenate(pr), np.concatenate(sc))
            m.update(etapr_scores([f['L'][:, 0] for f in files], pr))
            dets = []
            for f, p_ in zip(files, pr):
                d, _ = event_metrics(f['L'][:, 0], p_); dets += list(d)
            m['event_recall'] = float(np.mean(dets)); r[k] = m
        out.setdefault(tag, {})[str(seed)] = r
for tag in out:
    save(f'ours_sota_{tag}', out[tag])
print('done')
