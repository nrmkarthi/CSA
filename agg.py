"""Aggregate per-seed nested result dicts into mean and 95 % CI half-width."""
import numpy as np
from common import ci95, loadres


def _leaves(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, prefix + (k,))
        elif isinstance(v, (int, float)):
            yield prefix + (k,), float(v)


def aggregate(res):
    """res: {seed: nested dict}. Returns {path_tuple: (mean, ci, n)}."""
    vals = {}
    for seed, d in res.items():
        for path, v in _leaves(d):
            vals.setdefault(path, []).append(v)
    return {p: ci95(v) + (len(v),) for p, v in vals.items()}


def get(agg, *path):
    return agg[tuple(path)]


def fmt(agg, *path, d=3, pct=False, ci=True):
    m, h, n = agg[tuple(path)]
    if pct:
        m, h = 100 * m, 100 * h
        d = 1 if d == 3 else d
    if np.isnan(m):
        return '--'
    return f'{m:.{d}f}' + (f'$\\pm${h:.{d}f}' if ci and n > 1 else '')


def load_agg(name):
    return aggregate(loadres(name))
