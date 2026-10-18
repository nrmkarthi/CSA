"""Glue: load cached out-of-fold outputs and build the per-file situation picture."""
import numpy as np, joblib, glob
from common import *
from sa import alert_events, correlate, incidents_from, build_graph, proc_agg, event_truth, QUIET_CLOSE
from xai import grounded_attr


def load_run(tag, seed, ver='hai-21.03'):
    d = f'{CACHE}/run_{ver}'
    det = joblib.load(f'{d}/det_{tag}_s{seed}.pkl')
    files = []
    for k in range(len(glob.glob(f'{d}/oof_{tag}_s{seed}_f*.npz'))):
        z = np.load(f'{d}/oof_{tag}_s{seed}_f{k}.npz')
        files.append(dict(phi=z['phi'].astype(np.float64), ph=z['ph'], tg=z['tg'], L=z['L']))
    return det, files


def situation(det, f, A, mode='graph', close=QUIET_CLOSE):
    """Adds score, alert events, incidents, open_id and grounded process shares to file dict f."""
    phi = f['phi']; s = phi.sum(1)
    ev = alert_events(s)
    if mode == 'alert':                         # detection-only: every alert event is its own incident
        inc = np.arange(len(ev)); T = len(s)
        open_id = np.full(T, -1)
        for k, (a, b) in enumerate(ev):
            open_id[a:min(T, b + 60)] = k
    else:
        inc = correlate(ev, phi, A, mode=mode, ph=f['ph'])
        open_id, _ = incidents_from(ev, inc, len(s))
    g = grounded_attr(phi, open_id, A)
    gp = proc_agg(g, det.proc_of)[:, :3]
    gs = gp / (gp.sum(1, keepdims=True) + 1e-9)
    out = dict(f); out.update(s=s, events=ev, inc=inc, open_id=open_id, gs=gs)
    return out
