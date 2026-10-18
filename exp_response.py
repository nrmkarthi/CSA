"""RQ3: adaptive, safety-constrained response with verification (trace-driven simulation, LOFO)."""
import numpy as np, sys, os, time
from core import *
from response import simulate, summarize, QAgent

SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else SEEDS

BASE = dict(mode='graph_phase', use_phase=True, target='consensus', verify=True, safe=False, p_fail=0.0)
POLICIES = {
    'no_response': dict(policy='none'),
    'static_isolate': dict(policy='static'),
    'threshold_playbook': dict(policy='playbook'),
    'escalate_all': dict(policy='escalate'),
    'static_isolate+controller': dict(policy='static', safe=True),
    'q_learning_unconstrained': dict(policy='rl'),
    'q_learning_safe (XAI-CSA)': dict(policy='safe_rl', safe=True),
}


def train_agent(train_files, cfg, seed, epochs=25):
    ag = QAgent(seed)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        ag.eps = max(0.02, 0.3 * (1 - ep / epochs))
        for j in rng.permutation(len(train_files)):
            simulate(train_files[j], cfg, ag, train=True, seed=seed * 7919 + ep * 31 + j)
    return ag


def evaluate(S, cfg, seed, n_sim=3):
    """LOFO evaluation; RL agents are trained on the other files. Returns summary dict."""
    runs = []
    K = len(S)
    for k in range(K):
        ag = None
        if cfg['policy'] in ('rl', 'safe_rl'):
            ag = train_agent([S[j] for j in range(K) if j != k], cfg, seed * 100 + k)
        for r in range(n_sim):
            runs.append(simulate(S[k], cfg, ag, train=False, seed=seed * 1000 + k * 10 + r))
    return summarize(runs)


def build_S(tag, seed, mode, ver='hai-21.03'):
    det, files = load_run(tag, seed, ver)
    A = build_graph(det)
    return [situation(det, f, A, mode) for f in files]


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'central'
    out = {}
    try:
        out = loadres(f'response_{tag}')
    except Exception:
        pass
    for seed in SEEDS_:
        t0 = time.time()
        S = build_S(tag, seed, 'graph_phase')
        out[str(seed)] = {}
        for name, over in POLICIES.items():
            cfg = dict(BASE); cfg.update(over)
            out[str(seed)][name] = evaluate(S, cfg, seed)
        # reliability: action-execution failures with / without verification
        for pf in (0.2, 0.5):
            for ver in (True, False):
                cfg = dict(BASE); cfg.update(policy='safe_rl', safe=True, verify=ver, p_fail=pf)
                out[str(seed)][f'pfail{pf}_verify{int(ver)}'] = evaluate(S, cfg, seed)
        print('response seed', seed, round(time.time() - t0), 's', flush=True)
        save(f'response_{tag}', out)
