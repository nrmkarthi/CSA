"""Layers 5-6 of XAI-CSA: multi-agent adaptive response, safety/policy controller, verification.

The response is evaluated in a trace-driven simulator built on the HAI test files: the recorded
detector outputs drive the agents; containment, execution failure, analyst escalation and
post-action telemetry are simulated (counterfactual: once every attacked process of an episode is
contained, post-action telemetry is treated as normal, apart from spurious persistence w.p. EPS_V).
"""
import numpy as np
from common import segments

DT = 5                     # decision period (s)
CRIT = np.array([1.0, 1.0, 0.6])  # criticality of P1 boiler, P2 turbine, P3 water treatment
P_ISO, P_RES = 0.95, 0.15  # containment prob. (isolation: once; restriction: per decision period)
C_ISO, C_RES = 1.0, 0.3    # service-disruption cost per second of isolation / restriction
V_WIN, EPS_V = 30, 0.05    # verification window (s), spurious-persistence probability
V_REL = 60                 # release measures after this many verified-normal seconds
D_MIN, D_MAX = 60, 180     # analyst response delay (s)
A_MON, A_RES, A_ISO, A_ESC = 0, 1, 2, 3
ACTIONS = ['monitor', 'restrict', 'isolate', 'escalate']


class SafetyController:
    """Checks authorisation, confidence, uncertainty, explanation consistency and service-impact budget."""
    def __init__(self, iso_conf=0.7, iso_margin=0.2, res_conf=0.4, need_agree=True, max_iso=1):
        self.iso_conf, self.iso_margin, self.res_conf = iso_conf, iso_margin, res_conf
        self.need_agree, self.max_iso = need_agree, max_iso
        self.log = []

    def allowed(self, a, obs, n_iso):
        if a == A_ISO:
            ok = obs['pa'] >= self.iso_conf and obs['margin'] >= self.iso_margin and n_iso < self.max_iso
            if self.need_agree and obs.get('agree') is not None:
                ok = ok and obs['agree']
            return ok
        if a == A_RES:
            return obs['pa'] >= self.res_conf
        return True

    def filter(self, a, obs, n_iso, t):
        if self.allowed(a, obs, n_iso):
            return a
        b = A_RES if (a == A_ISO and self.allowed(A_RES, obs, n_iso)) else (A_ESC if obs['pa'] >= self.res_conf else A_MON)
        self.log.append((t, ACTIONS[a], ACTIONS[b]))
        return b


def state_key(obs, ver, meas, esc):
    pa = 0 if obs['pa'] < 0.3 else (1 if obs['pa'] < 0.7 else 2)
    mg = 0 if obs['margin'] < 0.2 else (1 if obs['margin'] < 0.5 else 2)
    sv = 0 if obs['s'] < 1 else (1 if obs['s'] < 3 else 2)
    return (pa, mg, sv, ver, meas, int(esc))


class QAgent:
    def __init__(self, seed, alpha=0.1, gamma=0.95, eps=0.2):
        self.Q = {}; self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma, self.eps = alpha, gamma, eps

    def q(self, s):
        if s not in self.Q:
            self.Q[s] = np.zeros(4)
        return self.Q[s]

    def act(self, s, allowed, greedy):
        qs = self.q(s).copy()
        mask = np.array([allowed(a) for a in range(4)])
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(np.where(mask)[0]))
        qs[~mask] = -1e9
        return int(np.argmax(qs))

    def update(self, s, a, r, s2, done):
        tgt = r if done else r + self.gamma * self.q(s2).max()
        self.q(s)[a] += self.alpha * (tgt - self.q(s)[a])


def episodes_of(L):
    eps = []
    for s, e in segments(L[:, 0]):
        tg = [k for k in range(3) if L[s:e, k + 1].any()]
        if not tg:
            tg = [0]
        eps.append(dict(s=s, e=e, targets=tg))
    return eps


def observe(t, s, ph, tg, gs, cfg):
    if cfg['use_phase']:
        pa = float(ph[t, 1])
    else:
        pa = float(np.clip(0.25 * s[t] + 0.25, 0, 1)) if s[t] > 1 else 0.0
    if cfg['target'] == 'consensus':
        c = 0.5 * tg[t] + 0.5 * gs[t]
        agree = int(np.argmax(tg[t])) == int(np.argmax(gs[t]))
    else:
        c = tg[t].copy(); agree = None
    order = list(np.argsort(-c))
    margin = float(c[order[0]] - c[order[1]])
    return dict(pa=pa, s=float(s[t]), rank=order, margin=margin, agree=agree)


def simulate(file, cfg, agent=None, train=False, seed=0):
    """file: dict(s, open_id, ph, tg, gs, L).  cfg keys: use_phase, target, policy, verify, p_fail, safe."""
    rng = np.random.default_rng(seed)
    s, open_id, ph, tg, gs, L = file['s'], file['open_id'], file['ph'], file['tg'], file['gs'], file['L']
    T = len(s)
    eps = episodes_of(L)
    active = np.zeros((T, 3), bool)            # true attack activity per process
    ep_of = -np.ones(T, int)
    for k, e in enumerate(eps):
        for p in e['targets']:
            active[e['s']:e['e'], p] = True
        ep_of[e['s']:e['e']] = k
    contained = [dict() for _ in eps]          # process -> containment time
    measure = np.zeros(3, int)                 # 0 none, 1 restricted, 2 isolated
    meas_t = np.zeros(3)
    meas_ok = np.zeros(3, bool)                # measure actually in effect (execution succeeded)
    disruption = 0.0; disruption_unneeded = 0.0
    actions = []                               # (t, action, proc, needed, failed)
    escalations = 0; analyst_due = None
    first_resp = {}
    ctrl = SafetyController(need_agree=(cfg['target'] == 'consensus')) if cfg.get('safe') else None
    verdicts = []                              # (action idx, verdict, truly_effective)
    pending_verif = []
    acted = set()
    tried = set(); handled = False; cur_inc = -2; last_ok = None; ver_state = 0; escalated = False
    prev = None
    reward_acc = 0.0

    def is_contained(k, p):
        return p in contained[k]

    def cf_alert(t):
        """Counterfactual post-action telemetry at time t."""
        k = ep_of[t]
        if k >= 0 and all(is_contained(k, p) for p in eps[k]['targets']):
            return rng.random() < EPS_V
        if cfg['use_phase']:
            return bool(s[t] > 1 and not ph[t, 2] > 0.5)      # hybrid (recovery-suppressed) alert stream
        return s[t] > 1

    def apply_action(t, a, p):
        nonlocal escalations, analyst_due
        if a in (A_RES, A_ISO):
            if measure[p] >= (2 if a == A_ISO else 1) and meas_ok[p]:
                return                                   # measure already in force: no-op
            failed = rng.random() < cfg.get('p_fail', 0.0)
            needed = bool(active[t, p])
            actions.append(dict(t=t, a=a, p=p, needed=needed, failed=failed))
            if measure[p] < (2 if a == A_ISO else 1):
                measure[p] = 2 if a == A_ISO else 1; meas_t[p] = t; meas_ok[p] = False
            if not failed:
                meas_ok[p] = True
                k = ep_of[t]
                if k >= 0 and p in eps[k]['targets'] and not is_contained(k, p):
                    if a == A_ISO and rng.random() < P_ISO:
                        contained[k][p] = t + DT
                    actions[-1]['res_active'] = True
            pending_verif.append(len(actions) - 1)
            return
        if a == A_ESC and analyst_due is None:
            escalations += 1
            analyst_due = t + int(rng.integers(D_MIN, D_MAX + 1))
            actions.append(dict(t=t, a=a, p=-1, needed=bool(active[t].any()), failed=False))

    # next second at which an incident is open (for skipping idle periods)
    nxt = np.full(T + 1, T)
    for u in range(T - 1, -1, -1):
        nxt[u] = u if open_id[u] >= 0 else nxt[u + 1]
    t = 0
    while t < T:
        inc = open_id[t]
        if inc < 0 and not measure.any() and analyst_due is None and not pending_verif:
            tn = nxt[t]
            if tn > t + DT:
                tn2 = t + ((tn - t) // DT) * DT
                for k, e in enumerate(eps):                      # damage reward of skipped seconds
                    lo, hi = max(t, e['s']), min(tn2, e['e'])
                    if hi > lo:
                        for p in e['targets']:
                            c = contained[k].get(p)
                            n_unc = hi - lo if c is None else max(0, hi - max(lo, c))
                            reward_acc -= CRIT[p] * n_unc / DT
                t = tn2
                if t >= T:
                    break
                inc = open_id[t]
        # ---- background dynamics every second inside the decision period
        for u in range(t, min(T, t + DT)):
            k = ep_of[u]
            for p in range(3):
                if measure[p]:
                    c = C_ISO if measure[p] == 2 else C_RES
                    disruption += c
                    if not active[u, p]:
                        disruption_unneeded += c
            if k >= 0:
                for p in eps[k]['targets']:
                    if not is_contained(k, p):
                        reward_acc -= CRIT[p] / DT
            # restriction: stochastic containment
            if k >= 0 and (u - t) == 0:
                for p in eps[k]['targets']:
                    if measure[p] == 1 and meas_ok[p] and not is_contained(k, p):
                        if rng.random() < P_RES:
                            contained[k][p] = u
        for p in range(3):
            if measure[p]:
                reward_acc -= (0.1 if measure[p] == 2 else 0.03) + ((C_ISO if measure[p] == 2 else C_RES) * 0.5 if not active[t, p] else 0)
        # ---- analyst
        if analyst_due is not None and t >= analyst_due:
            k = ep_of[t]
            if k >= 0:
                for p in eps[k]['targets']:
                    if not is_contained(k, p):
                        contained[k][p] = t
                        measure[p] = 2; meas_t[p] = t; meas_ok[p] = True
                last_ok = t
            # analyst lifts unnecessary measures
            for p in range(3):
                if measure[p] and not active[t, p]:
                    measure[p] = 0
            analyst_due = None
            if k < 0:
                handled = True
        # ---- verification of pending actions
        if cfg['verify'] and pending_verif:
            j = pending_verif[0]
            if t >= actions[j]['t'] + V_WIN:
                obs_alert = np.mean([cf_alert(u) for u in range(actions[j]['t'] + 1, min(T, actions[j]['t'] + V_WIN + 1))])
                verdict = 'persist' if obs_alert >= 0.5 else 'mitigated'
                k = ep_of[actions[j]['t']]
                effective = k >= 0 and actions[j]['p'] in contained[k] and not actions[j]['failed']
                verdicts.append(dict(j=j, verdict=verdict, effective=effective,
                                     problem=(k >= 0 and (actions[j]['failed'] or actions[j]['p'] not in eps[k]['targets']))))
                pending_verif.pop(0)
                if verdict == 'persist':
                    tried.add(actions[j]['p']); ver_state = 1
                    if actions[j]['failed']:
                        tried.discard(actions[j]['p'])     # re-issue on the same target after a failed execution
                        actions[j]['failed_detected'] = True
                        measure[actions[j]['p']] = 0; meas_ok[actions[j]['p']] = False
                        acted.discard((actions[j]['p'], actions[j]['a']))
                else:
                    ver_state = 2; last_ok = t
        # release measures after verified-normal period, or when no incident is open (without verification:
        # held until the incident closes)
        for p in range(3):
            if measure[p]:
                if cfg['verify'] and last_ok is not None and t - last_ok >= V_REL and not any(cf_alert(u) for u in range(max(0, t - V_REL), t, 10)):
                    measure[p] = 0
                elif inc < 0:
                    measure[p] = 0
        # ---- decision
        if inc >= 0:
            if inc != cur_inc:
                cur_inc = inc; tried = set(); handled = False; ver_state = 0; last_ok = None; escalated = False
                acted = set()
            obs = observe(t, s, ph, tg, gs, cfg)
            cand = [p for p in obs['rank'] if p not in tried] or obs['rank']
            p_star = cand[0]
            # C3/C4 are evaluated for the sub-process that will actually be acted on (it differs from the top-ranked one
            # after a verified failure): margin of p_star over the best alternative, and agreement of both sources on p_star
            c_now = 0.5 * tg[t] + 0.5 * gs[t] if cfg['target'] == 'consensus' else tg[t]
            obs = dict(obs, margin=float(c_now[p_star] - np.max(np.delete(c_now, p_star))),
                       agree=(None if obs['agree'] is None else bool(p_star == int(np.argmax(tg[t])) == int(np.argmax(gs[t])))))
            n_iso = int((measure == 2).sum())
            a = A_MON
            pol = cfg['policy']
            if handled and not cfg['verify']:
                a = A_MON
            elif pol == 'static':
                if obs['pa'] > 0.5 and (p_star, A_ISO) not in acted:
                    a = A_ISO
            elif pol == 'playbook':
                if obs['pa'] >= 0.9 and (p_star, A_ISO) not in acted:
                    a = A_ISO
                elif obs['pa'] >= 0.6 and (p_star, A_RES) not in acted and (p_star, A_ISO) not in acted:
                    a = A_RES
                    if not escalated:
                        apply_action(t, A_ESC, -1); escalated = True
            elif pol == 'escalate':
                if obs['pa'] > 0.5 and not escalated:
                    a = A_ESC
            elif pol in ('rl', 'safe_rl'):
                key = state_key(obs, ver_state, int(measure[p_star]), escalated)
                if prev is not None:
                    agent.update(prev[0], prev[1], reward_acc, key, False) if train else None
                reward_acc = 0.0
                allowed = (lambda x: ctrl.allowed(x, obs, n_iso)) if ctrl is not None else (lambda x: True)
                a = agent.act(key, allowed, greedy=not train)
                prev = (key, a)
            if ctrl is not None and pol not in ('rl', 'safe_rl'):
                a = ctrl.filter(a, obs, n_iso, t)
            if a == A_ESC and escalated:
                a = A_MON
            if a != A_MON:
                acted.add((p_star, a))
                apply_action(t, a, p_star)
                if a == A_ESC:
                    escalated = True
                if not cfg['verify'] and a in (A_ISO, A_RES):
                    handled = True
                k = ep_of[t]
                if k >= 0 and k not in first_resp:
                    first_resp[k] = t - eps[k]['s']
        t += DT
    if train and prev is not None:
        agent.update(prev[0], prev[1], reward_acc, prev[0], True)
    # ---- metrics
    res = []
    for k, e in enumerate(eps):
        ct = [contained[k][p] for p in e['targets'] if p in contained[k]]
        full = len(ct) == len(e['targets']) and max(ct) < e['e']
        ttc = (max(ct) - e['s']) if full else (e['e'] - e['s'])
        res.append(dict(contained=full, ttc=float(ttc), dur=float(e['e'] - e['s']), resp_lat=first_resp.get(k, np.nan),
                        damage=float(sum(CRIT[p] * ((contained[k].get(p, e['e']) if contained[k].get(p, e['e']) < e['e'] else e['e']) - e['s']) for p in e['targets']))))
    prot = [x for x in actions if x['a'] in (A_RES, A_ISO)]
    problems = [v for v in verdicts if v['problem']]
    return dict(episodes=res, n_actions=len(prot), unnecessary=sum(not x['needed'] for x in prot),
                disruption=disruption, disruption_unneeded=disruption_unneeded, escalations=escalations,
                hours=T / 3600, n_failed=sum(x['failed'] for x in prot),
                failed_detected=sum(bool(x.get('failed_detected')) for x in prot),
                problem_actions=len(problems), problem_detected=sum(v['verdict'] == 'persist' for v in problems),
                ctrl_overrides=len(ctrl.log) if ctrl else 0)


def summarize(runs):
    eps = [e for r in runs for e in r['episodes']]
    n_act = sum(r['n_actions'] for r in runs)
    hours = sum(r['hours'] for r in runs)
    pa = sum(r['problem_actions'] for r in runs)
    return dict(msr=float(np.mean([e['contained'] for e in eps])),
                ttc=float(np.mean([e['ttc'] for e in eps])),
                ttc_contained=float(np.nanmean([e['ttc'] if e['contained'] else np.nan for e in eps])) if any(e['contained'] for e in eps) else float('nan'),
                resp_lat=float(np.nanmedian([e['resp_lat'] for e in eps])) if any(np.isfinite(e['resp_lat']) for e in eps) else float('nan'),
                damage=float(np.mean([e['damage'] for e in eps])),
                uar=float(sum(r['unnecessary'] for r in runs) / max(n_act, 1)),
                unnecessary_per_day=float(sum(r['unnecessary'] for r in runs) / hours * 24),
                disruption_h_per_day=float(sum(r['disruption'] for r in runs) / 3600 / hours * 24),
                disruption_unneeded_h_per_day=float(sum(r['disruption_unneeded'] for r in runs) / 3600 / hours * 24),
                escalations_per_day=float(sum(r['escalations'] for r in runs) / hours * 24),
                failed_detect_rate=float(sum(r['failed_detected'] for r in runs) / max(sum(r['n_failed'] for r in runs), 1)),
                problem_detect_rate=float(sum(r['problem_detected'] for r in runs) / max(pa, 1)) if pa else float('nan'),
                actions_per_day=float(n_act / hours * 24))
