"""Figures for the paper (all values are means over seeds, error bars = 95 % CI)."""
import numpy as np, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agg import load_agg
from figstyle import *

MAIN = sys.argv[1] if len(sys.argv) > 1 else 'fedavg'
which = sys.argv[2].split(',') if len(sys.argv) > 2 else ['corr', 'xai', 'resp', 'robust']


def fig_corr():
    a = load_agg(f'sa_{MAIN}_hai-21.03')
    dts = [30, 60, 120, 300, 600]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6))
    for j, (md, lab) in enumerate([('temporal', 'temporal only'), ('graph', 'graph-gated'), ('graph_phase', 'phase-aware (ours)')]):
        m = [a[('splice', f'{md}_g0.6_dt{d}', 'bcubed_f1')][0] for d in dts]
        h = [a[('splice', f'{md}_g0.6_dt{d}', 'bcubed_f1')][1] for d in dts]
        ax[0].errorbar(range(len(dts)), m, yerr=h, color=C[j], marker='o', ms=3, lw=1.2, capsize=1.5, label=lab)
    nm = a[('splice', 'none', 'bcubed_f1')][0]
    ax[0].axhline(nm, color=GREY, ls='--', lw=0.8, label='no correlation')
    ax[0].set_xticks(range(len(dts))); ax[0].set_xticklabels(dts)
    ax[0].set_xlabel(r'window $\delta_t$ (s)  [$\gamma$=0.6]'); ax[0].set_ylabel('B-cubed F1'); ax[0].set_ylim(0, 1.05)
    gs = [0.0, 0.2, 0.4, 0.6, 0.8]
    for j, md in [(1, 'graph'), (2, 'graph_phase')]:
        m = [a[('splice', f'{md}_g{g}_dt300', 'bcubed_f1')][0] for g in gs]
        h = [a[('splice', f'{md}_g{g}_dt300', 'bcubed_f1')][1] for g in gs]
        ax[1].errorbar(gs, m, yerr=h, color=C[j], marker='o', ms=3, lw=1.2, capsize=1.5)
    ax[1].axhline(nm, color=GREY, ls='--', lw=0.8)
    ax[1].set_xlabel(r'similarity $\gamma$  [$\delta_t$=300 s]'); ax[1].set_ylim(0, 1.05)
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=6, columnspacing=0.8, handlelength=1.5)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(f'{FIG}/corr_sweep.pdf'); fig.savefig(f'{FIG}/corr_sweep.png', dpi=200)


def fig_xai():
    a = load_agg(f'xai_{MAIN}')
    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6), sharey=False)
    names = [('residual', 'residual attribution (ours)'), ('shapley', 'Shapley sampling'), ('magnitude', r'deviation $|z|$'), ('random', 'random')]
    for i, mode in enumerate(['cond', 'mean']):
        for j, (m, lab) in enumerate(names):
            v = [a[('fidelity', m, f'{mode}_k{k}')] for k in ks]
            ax[i].errorbar(ks, [x[0] for x in v], yerr=[x[1] for x in v], color=C[j] if m != 'random' else GREY, marker='o', ms=3, lw=1.2, capsize=1.5, label=lab)
        ax[i].set_xticks(ks); ax[i].set_xlabel('tags removed $k$'); ax[i].set_title('conditional imputation' if mode == 'cond' else 'mean replacement', fontsize=6.5)
    ax[0].set_ylabel('relative score drop')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(f'{FIG}/xai_fidelity.pdf'); fig.savefig(f'{FIG}/xai_fidelity.png', dpi=200)


def fig_resp():
    a = load_agg(f'response_{MAIN}')
    pols = ['escalate_all', 'static_isolate', 'threshold_playbook', 'static_isolate+controller', 'q_learning_unconstrained', 'q_learning_safe (XAI-CSA)']
    labs = ['Escalate all', 'Static isolation', 'Threshold playbook', 'Static + controller', 'Q-learning', 'Safe Q-learning (XAI-CSA)']
    mets = [('msr', 'mitigation success rate', 1), ('uar', 'unnecessary-action rate', 1), ('disruption_unneeded_h_per_day', 'unneeded disruption (h/day)', 1)]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.55), sharey=True)
    y = np.arange(len(pols))
    for j, (m, lab, _) in enumerate(mets):
        v = [a[(p, m)] for p in pols]
        cols = [C[0] if 'XAI-CSA' in p else GREY for p in pols]
        ax[j].barh(y, [x[0] for x in v], xerr=[x[1] for x in v], color=cols, height=0.6, error_kw=dict(lw=0.8, capsize=1.5))
        for yi, x in zip(y, v):
            ax[j].text(x[0] + x[1] + 0.03 * max(z[0] + z[1] for z in v), yi, f'{x[0]:.2f}', va='center', fontsize=5.8)
        ax[j].set_xlabel(lab); ax[j].grid(axis='y', visible=False)
        ax[j].set_xlim(0, 1.18 * max(z[0] + z[1] for z in v))
    ax[0].set_yticks(y); ax[0].set_yticklabels(labs)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(f'{FIG}/resp_tradeoff.pdf'); fig.savefig(f'{FIG}/resp_tradeoff.png', dpi=200)


def fig_robust():
    a = load_agg(f'robust_{MAIN}'); av = load_agg(f'robust_{MAIN}_adv')
    eps = ['0.0', '1.0', '3.0', '10.0', '30.0']
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.65))
    for j, (k, lab) in enumerate([('gcr_target', 'GCR (attacked)'), ('anomaly_only', 'fused detector'), ('hybrid', 'hybrid (ours)')]):
        v = [av[('adversarial', e, k, 'recall')] for e in eps]
        ax[0].errorbar(range(len(eps)), [x[0] for x in v], yerr=[x[1] for x in v], color=[GREY, C[0], C[1]][j], marker='o', ms=3, lw=1.2, capsize=1.5, label=lab)
    ax[0].set_xticks(range(len(eps))); ax[0].set_xticklabels([e.rstrip('0').rstrip('.') for e in eps])
    ax[0].set_xlabel(r'evasion budget $\epsilon$ ($\sigma$)'); ax[0].set_ylabel('point recall'); ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=5.5, frameon=False, loc='lower left')
    mags = ['1.0', '3.0', '5.0']
    for j, (k, lab) in enumerate([('no_adaptation', 'static'), ('drift_aware', 'drift-aware (ours)')]):
        v = [a[('drift', m, k, 'f1')] for m in mags]
        ax[1].errorbar([1, 3, 5], [x[0] for x in v], yerr=[x[1] for x in v], color=[GREY, C[2]][j], marker='o', ms=3, lw=1.2, capsize=1.5, label=lab)
    ax[1].set_xlabel(r'injected drift ($\sigma$)'); ax[1].set_ylabel('F1'); ax[1].set_xticks([1, 3, 5]); ax[1].set_ylim(0, 0.8)
    ax[1].legend(fontsize=5.5, frameon=False)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(f'{FIG}/robust.pdf'); fig.savefig(f'{FIG}/robust.png', dpi=200)


if __name__ == '__main__':
    for w in which:
        globals()[f'fig_{w}']()
        print('fig', w)
