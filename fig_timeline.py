"""Qualitative figure: one coordinated multi-process incident as seen by XAI-CSA."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *
from sa import segments
from figstyle import *

TAG = 'fedavg'
det, files = load_run(TAG, 0)
A = build_graph(det)
k = 2                                            # test3 contains P1+P2 coordinated attacks
f = situation(det, files[k], A, 'graph_phase')
L = f['L']
eps = [(a, b) for a, b in segments(L[:, 0]) if L[a:b, 1].any() and L[a:b, 2].any()]
a, b = eps[0]
lo, hi = a - 300, b + 700
t = np.arange(lo, hi) - a
fig, ax = plt.subplots(3, 1, figsize=(3.5, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
ax[0].plot(t, f['s'][lo:hi], color=C[0], lw=1.2, label='fused score $s_t$')
ax[0].axhline(1, color=GREY, lw=0.8, ls='--')
ax[0].set_ylabel('score'); ax[0].set_yscale('log')
for x in ax:
    x.axvspan(0, b - a, color=C[1], alpha=0.12, lw=0)
    x.axvspan(b - a, b - a + 600, color=GREY, alpha=0.10, lw=0)
ax[0].set_ylim(0.05, 2e3)
ax[0].text(5, 1.5e3, 'attack\n(P1+P2)', fontsize=6, va='top')
ax[0].text(b - a + 15, 1.5e3, 'reference recovery window', fontsize=6, va='top')
ph = f['ph'][lo:hi]; op = f['open_id'][lo:hi] >= 0
ax[1].plot(t, ph[:, 1] * op, color=C[4], lw=1.2, label='$h^A$ (active)')
ax[1].plot(t, ph[:, 2] * op, color=C[5], lw=1.2, label='$h^R$ (recovery)')
ax[1].set_ylabel('phase'); ax[1].set_ylim(-0.05, 1.05); ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), fontsize=6, frameon=False, ncol=2, borderaxespad=0.1)
c = 0.5 * f['tg'][lo:hi] + 0.5 * f['gs'][lo:hi]
for p in range(3):
    ax[2].plot(t, c[:, p] * op, color=C[p], lw=1.2, label=f'P{p + 1}')
ax[2].set_ylabel('target $c_{t,k}$'); ax[2].set_ylim(-0.05, 1.05); ax[2].legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), fontsize=6, frameon=False, ncol=3, borderaxespad=0.1)
ax[2].set_xlabel('time since attack onset (s)')
fig.align_ylabels(ax)
fig.tight_layout(h_pad=0.9)
fig.savefig(f'{FIG}/timeline.pdf'); fig.savefig(f'{FIG}/timeline.png', dpi=200)
print('episode', a, b, 'targets', [p for p in range(3) if L[a:b, p + 1].any()])
