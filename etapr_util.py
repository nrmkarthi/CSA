"""eTaPR / TaPR via the reference implementation (github.com/saurf4ng/eTaPR)."""
import sys, os, numpy as np
# path to a clone of https://github.com/saurf4ng/eTaPR (default: ext/eTaPR inside this repository)
sys.path.insert(0, os.environ.get('ETAPR_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext', 'eTaPR')))
import types
# the package imports an optional plotting module (opencv); stub it if unavailable
try:
    from eTaPR_pkg import etapr
except Exception:
    sys.modules['eTaPR_pkg.DataManage.Time_Plot'] = types.ModuleType('Time_Plot')
    from eTaPR_pkg import etapr
from eTaPR_pkg.DataManage import Range as rng


def _ranges(mask):
    m = np.asarray(mask).astype(np.int8); d = np.diff(np.r_[0, m, 0])
    st = np.where(d == 1)[0]; en = np.where(d == -1)[0] - 1
    return [rng.Range(int(a), int(b), str(i)) for i, (a, b) in enumerate(zip(st, en))]


def etapr_scores(y_list, pred_list, theta_p=0.5, theta_r=0.1, delta=0.0):
    """Concatenates files (with a 1-s gap marker) and returns eTaP, eTaR, eTaF1."""
    y = np.concatenate([np.r_[np.asarray(a), 0] for a in y_list]); p = np.concatenate([np.r_[np.asarray(b), 0] for b in pred_list])
    r = etapr.evaluate_w_ranges(_ranges(y), _ranges(p), theta_p, theta_r, delta)
    P, R = r['eTaP'], r['eTaR']
    return dict(etap=float(P), etar=float(R), etaf1=float(2 * P * R / (P + R)) if P + R > 0 else 0.0)
