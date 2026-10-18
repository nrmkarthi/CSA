"""Generate LaTeX tables (paper/tables/*.tex) directly from the aggregated result files."""
import os, numpy as np
from agg import load_agg, fmt
from common import RES, ROOT

OUT = os.environ.get('XAICSA_TABLES', os.path.join(ROOT, 'paper', 'tables'))
os.makedirs(OUT, exist_ok=True)
MAIN = 'fedavg'


def w(name, s):
    import re
    if s.lstrip().startswith('\\begin{table}'):   # single-column table: never exceed the column width
        s = s.replace('\\begin{tabular}', '\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}', 1)
        s = s.replace('\\end{tabular}', '\\end{tabular}}', 1)
    else:
        s = s.replace('\\begin{tabular}', '\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', 1)
        s = s.replace('\\end{tabular}', '\\end{tabular}\n\\end{adjustbox}', 1)
    open(f'{OUT}/{name}.tex', 'w').write(s)


def have(name):
    return os.path.exists(f'{RES}/{name}.json')


def P(a, *p, d=3):
    return fmt(a, *p, d=d)


def pct(a, *p, d=2):
    m, h, n = a[tuple(p)]
    return f'{100 * m:.{d}f}$\\pm${100 * h:.{d}f}'


def t_detection():
    b = load_agg('baselines_central_hai-21.03'); s = load_agg(f'sa_{MAIN}_hai-21.03'); sc = load_agg('sa_central_hai-21.03')
    rows = []
    def row(name, a, *pre, lat=None):
        r = f'{name} & ' + ' & '.join([P(a, *pre, 'precision'), P(a, *pre, 'recall'), P(a, *pre, 'f1'), P(a, *pre, 'pr_auc'),
                                     pct(a, *pre, 'fpr'), P(a, *pre, 'event_recall', d=2)])
        r += ' & ' + (P(a, *pre, lat, d=1) if lat else '--') + r' \\'
        rows.append(r)
    rows.append(r'\multicolumn{8}{l}{\textit{Unsupervised baselines and single components (centralized training)}}\\')
    row('Isolation Forest', b, 'isolation_forest'); row('PCA residual', b, 'pca'); row('GCR only', b, 'gcr_only'); row('AE only', b, 'ae_only')
    rows.append(r'\midrule\multicolumn{8}{l}{\textit{Centralized training, Layer-3 outputs}}\\')
    row('Anomaly detector', sc, 'detection', 'anomaly_only', lat='latency_median')
    row('Attack classifier only$^\\dagger$', sc, 'detection', 'classifier_only', lat='latency_median')
    row('Hybrid', sc, 'detection', 'hybrid', lat='latency_median')
    rows.append(r'\midrule\multicolumn{8}{l}{\textit{Federated training (XAI-CSA deployment)}}\\')
    row('Anomaly detector', s, 'detection', 'anomaly_only', lat='latency_median')
    row('Attack classifier only$^\\dagger$', s, 'detection', 'classifier_only', lat='latency_median')
    row('\\textbf{Hybrid (XAI-CSA)}', s, 'detection', 'hybrid', lat='latency_median')
    w('detection', r'''\begin{table*}[!t]
\centering
\caption{Point-wise and event-level detection on HAI 21.03 (mean$\pm$95\% CI over five seeds; FPR in \%; latency is the median delay
from attack onset to first alert, in seconds). $^\dagger$Supervised, leave-one-file-out.}
\label{tab:detection}
\footnotesize
\begin{tabular}{lccccccc}
\toprule
Method & Precision & Recall & F1 & PR-AUC & FPR (\%) & Event recall & Latency (s)\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table*}
''')


def t_sa():
    s = load_agg(f'sa_{MAIN}_hai-21.03')
    L = []
    L.append(r'\multicolumn{4}{l}{\textit{Perception / comprehension}}\\')
    L.append(f"Incident-state accuracy & {P(s,'state','detection_only','acc')} & {P(s,'state','sa_engine','acc')} & \\\\")
    L.append(f"Incident-state macro-F1 & {P(s,'state','detection_only','macro_f1')} & {P(s,'state','sa_engine','macro_f1')} & \\\\")
    L.append(f"Full-state acc. (active, target set) & {P(s,'state','detection_only_fullstate_acc_active')} & {P(s,'state','sa_consensus_fullstate_acc_active')} & \\\\")
    L.append(f"Asset-impact Jaccard & {P(s,'impact','detection_only','jaccard')} & {P(s,'impact','sa_consensus','jaccard')} & \\\\")
    L.append(f"Asset-impact exact match & {P(s,'impact','detection_only','exact')} & {P(s,'impact','sa_consensus','exact')} & \\\\")
    L.append(f"Attacked-process macro-F1 & -- & {P(s,'detection','target_macro_f1')} & \\\\")
    L.append(r'\midrule\multicolumn{4}{l}{\textit{Event correlation (real timeline)}}\\')
    L.append(f"Pairwise F1 & 0 (none) & {P(s,'correlation','graph_phase','pair_f1')} & {P(s,'correlation','temporal','pair_f1')}\\\\")
    L.append(f"B-cubed F1 & {P(s,'correlation','none','bcubed_f1')} & {P(s,'correlation','graph_phase','bcubed_f1')} & {P(s,'correlation','temporal','bcubed_f1')}\\\\")
    L.append(f"Incidents per day & {P(s,'correlation','none','incidents_per_day',d=1)} & {P(s,'correlation','graph_phase','incidents_per_day',d=1)} & {P(s,'correlation','temporal','incidents_per_day',d=1)}\\\\")
    L.append(f"Incident precision & {P(s,'correlation','none','incident_precision')} & {P(s,'correlation','graph_phase','incident_precision')} & {P(s,'correlation','temporal','incident_precision')}\\\\")
    L.append(r'\midrule\multicolumn{4}{l}{\textit{Projection ($H=60$\,s)}}\\')
    L.append(f"Continuation F1 & {P(s,'projection','continuation','persistence','f1')} & {P(s,'projection','continuation','sa_hazard','f1')} & {P(s,'projection','continuation','phase','f1')}\\\\")
    L.append(f"Continuation ROC-AUC & {P(s,'projection','continuation','persistence','roc_auc')} & {P(s,'projection','continuation','sa_hazard','roc_auc')} & {P(s,'projection','continuation','phase','roc_auc')}\\\\")
    L.append(f"Continuation FPR & {P(s,'projection','continuation','persistence','fpr')} & {P(s,'projection','continuation','sa_hazard','fpr')} & {P(s,'projection','continuation','phase','fpr')}\\\\")
    L.append(f"Propagation AP & {P(s,'projection','propagation','prior','ap')} & {P(s,'projection','propagation','transition','ap')} & {P(s,'projection','propagation','graph','ap')}\\\\")
    L.append(f"Propagation ROC-AUC & {P(s,'projection','propagation','prior','auc')} & {P(s,'projection','propagation','transition','auc')} & {P(s,'projection','propagation','graph','auc')}\\\\")
    L.append(f"Propagation recall@5 & {P(s,'projection','propagation','prior','recall_at5')} & {P(s,'projection','propagation','transition','recall_at5')} & {P(s,'projection','propagation','graph','recall_at5')}\\\\")
    L.append(r'\midrule\multicolumn{4}{l}{\textit{Triage of alert events}}\\')
    L.append(f"ROC-AUC (true vs.\\ false alert) & {P(s,'triage','peak_score','auc')} & {P(s,'triage','sa_risk','auc')} & \\\\")
    L = [l for l in L if l]
    w('sa', r'''\begin{table}[!t]
\centering
\caption{Situation-awareness quality on HAI 21.03 (federated detector). Columns: \emph{Baseline} = detection-only estimate
(perception/comprehension), no correlation, persistence (continuation) or frequency prior (propagation), peak score (triage);
\emph{XAI-CSA} = SA engine, phase-aware correlation, hazard model, learned transition model, SA risk; \emph{Alt.} = temporal-only
correlation, phase posterior, structural graph diffusion.}
\label{tab:sa}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccc}
\toprule
Metric & Baseline & XAI-CSA & Alt.\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


def t_xai():
    x = load_agg(f'xai_{MAIN}')
    meth = [('residual', 'Residual attribution (ours)'), ('shapley', 'Shapley sampling'), ('magnitude', 'Deviation $|z|$'), ('random', 'Random')]
    L = []
    for m, lab in meth:
        st = P(x, 'perturbation_stability_top5', m) if m != 'random' else '--'
        L.append(f"{lab} & {P(x,'fidelity',m,'cond_k1')} & {P(x,'fidelity',m,'cond_k5')} & {P(x,'fidelity',m,'mean_k5')} & {st}\\\\")
    w('xai_fid', r'''\begin{table}[!t]
\centering
\caption{Explanation fidelity (relative score drop after removing the top-$k$ tags; cond.\ = conditional-Gaussian imputation,
mean = normal-mean replacement) and perturbation stability (top-5 Jaccard) on HAI 21.03.}
\label{tab:xaifid}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Method & cond.\ $k{=}1$ & cond.\ $k{=}5$ & mean $k{=}5$ & Stability\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table}
''')
    L = []
    L.append(f"Temporal stability (top-5, $t$ vs.\\ $t{{+}}10$\\,s) & {P(x,'temporal_stability_top5','instant')} & {P(x,'temporal_stability_top5','smoothed')} & {P(x,'temporal_stability_top5','grounded')}\\\\")
    L.append(f"Localization at $+30$\\,s & {P(x,'localisation_at_30s','instant')} & {P(x,'localisation_at_30s','classifier')} & {P(x,'localisation_at_30s','consensus')}\\\\")
    L.append(f"Label-based expl.--situation agreement & {P(x,'explanation_situation_correctness','feature_only')} & -- & {P(x,'explanation_situation_correctness','grounded')}\\\\")
    w('xai_ctx', r'''\begin{table}[!t]
\centering
\caption{Situation-grounded vs.\ feature-level explanations (HAI 21.03). Columns: instantaneous feature attribution;
smoothed attribution (stability) or attack classifier (localization); situation-grounded explanation.}
\label{tab:xaictx}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{p{3.3cm}ccc}
\toprule
Metric & Feature & Smoothed/Clf. & Grounded\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


def t_response():
    a = load_agg(f'response_{MAIN}')
    pols = [('no_response', 'No response'), ('escalate_all', 'Escalate all (human)'), ('static_isolate', 'Static isolation (SOAR)'),
            ('threshold_playbook', 'Threshold playbook'), ('static_isolate+controller', 'Static + safety controller'),
            ('q_learning_unconstrained', 'Q-learning (unconstrained)'), ('q_learning_safe (XAI-CSA)', '\\textbf{Safe Q-learning (XAI-CSA)}')]
    L = []
    for k, lab in pols:
        L.append(f"{lab} & {P(a,k,'msr',d=2)} & {P(a,k,'ttc',d=1)} & {P(a,k,'damage',d=0)} & {P(a,k,'uar',d=2)} & {P(a,k,'unnecessary_per_day',d=1)} & {P(a,k,'disruption_h_per_day',d=2)} & {P(a,k,'disruption_unneeded_h_per_day',d=2)} & {P(a,k,'escalations_per_day',d=1)}\\\\")
    w('response', r'''\begin{table*}[!t]
\centering
\caption{Response quality in the trace-driven simulator (HAI 21.03, federated detector, leave-one-file-out RL training, three
simulator repetitions per fold, mean$\pm$95\% CI over five seeds). MSR: mitigation success rate; TTC: mean time to contain (s,
censored at attack end); damage: criticality-weighted uncontained attack seconds per episode; UAR: unnecessary-action rate;
disruption: isolation-equivalent hours per day (total / on non-attacked sub-processes).}
\label{tab:response}
\footnotesize
\begin{tabular}{lcccccccc}
\toprule
Policy & MSR & TTC (s) & Damage & UAR & Unnec./day & Disrupt.\ h/day & Unneeded h/day & Escal./day\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table*}
''')
    L = []
    for pf in ('0.2', '0.5'):
        L.append(f"{pf} & {P(a,f'pfail{pf}_verify1','msr',d=2)} & {P(a,f'pfail{pf}_verify0','msr',d=2)} & {P(a,f'pfail{pf}_verify1','ttc',d=1)} & {P(a,f'pfail{pf}_verify0','ttc',d=1)} & {P(a,f'pfail{pf}_verify1','failed_detect_rate',d=2)}\\\\")
    w('reliability', r'''\begin{table}[!t]
\centering
\caption{Reliability under action-execution failures (safe Q-learning, with/without verification).}
\label{tab:reliability}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{cccccc}
\toprule
& \multicolumn{2}{c}{MSR} & \multicolumn{2}{c}{TTC (s)} & Failed-action\\
$p_{\mathrm{fail}}$ & verif. & no verif. & verif. & no verif. & detection\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


def t_fl():
    f = load_agg('fl')
    rows = []
    def row(lab, key, comm=''):
        rows.append(f"{lab} & {P(f,key,'f1')} & {P(f,key,'pr_auc')} & {pct(f,key,'fpr')} & {P(f,key,'event_recall',d=2)} & {comm}\\\\")
    mb = lambda v: f'{v / 1e6:.1f}'
    cu = f[('comm', 'bytes_up')][0]; cd = f[('comm', 'bytes_down')][0]; raw = f[('comm', 'raw_bytes_central')][0]
    row('Centralized (raw data pooled)', 'central', mb(raw))
    row('Local only (single site)', 'local', '0')
    row('FedAvg (XAI-CSA)', 'fedavg', mb(cu + cd))
    if ('dp_z1.0', 'f1') in f:
        row('FedAvg + Gaussian noise ($z{=}1$)', 'dp_z1.0', mb(cu + cd))
    if ('poison_fedavg', 'f1') in f:
        row('FedAvg, 1 poisoned site', 'poison_fedavg', mb(cu + cd))
        row('Trimmed mean, 1 poisoned site', 'poison_trimmed', mb(cu + cd))
        row('Trimmed mean, clean', 'clean_trimmed', mb(cu + cd))
    w('fl', r'''\begin{table}[!t]
\centering
\caption{Privacy-preserving learning on HAI 21.03 (anomaly detector only). Comm.: total MB exchanged (up+down) for training,
compared with uploading all raw normal data (centralized). Local only: one site's model (five sites, one per seed).
Rows with noise, poisoning and trimmed mean use three seeds; all other rows use five.}
\label{tab:fl}
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lccccc}
\toprule
Training & F1 & PR-AUC & FPR (\%) & Ev.\ rec. & Comm.\ (MB)\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


def t_robust():
    r = load_agg(f'robust_{MAIN}')
    L = []
    for p in ('P1', 'P2', 'P3'):
        L.append(f"{p} ({int(r[('zeroday',p,'anomaly_only','n')][0])}) & {P(r,'zeroday',p,'classifier_only','event_recall',d=2)} & {P(r,'zeroday',p,'anomaly_only','event_recall',d=2)} & {P(r,'zeroday',p,'hybrid','event_recall',d=2)}\\\\")
    w('zeroday', r'''\begin{table}[!t]
\centering
\caption{Zero-day evaluation: event recall on attacks touching a sub-process whose attacks were removed from all classifier
training folds (number of episodes in parentheses).}
\label{tab:zeroday}
\footnotesize
\begin{tabular}{lccc}
\toprule
Held-out & Classifier only & Anomaly only & Hybrid\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table}
''')
    d20 = load_agg('robust_central_hai-20.07')
    L = []
    for lab, a, key in [('HAI 21.03, no injected drift', r, 'clean'), ('HAI 21.03, injected drift $1\\sigma$', r, '1.0'),
                        ('HAI 21.03, injected drift $3\\sigma$', r, '3.0'), ('HAI 21.03, injected drift $5\\sigma$', r, '5.0'),
                        ('HAI 20.07, natural drift$^\\ast$', d20, 'clean')]:
        L.append(f"{lab} & {P(a,'drift',key,'no_adaptation','f1')} & {P(a,'drift',key,'drift_aware','f1')} & {pct(a,'drift',key,'no_adaptation','fpr')} & {pct(a,'drift',key,'drift_aware','fpr')} & {P(a,'drift',key,'no_adaptation','event_recall',d=2)} & {P(a,'drift',key,'drift_aware','event_recall',d=2)}\\\\")
    w('drift', r'''\begin{table*}[!t]
\centering
\caption{Concept drift: anomaly detection without and with persistence-gated drift compensation. Injected drift: six random
tags ramp linearly to the given offset over each test file. $^\ast$Centralized detector trained on HAI 20.07 normal data.}
\label{tab:drift}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{F1} & \multicolumn{2}{c}{FPR (\%)} & \multicolumn{2}{c}{Event recall}\\
Condition & static & drift-aware & static & drift-aware & static & drift-aware\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table*}
''')
    L = []
    for k, lab in [('nominal', 'Nominal'), ('ae_node_down', 'AE node failure (GCR + rules)'), ('gcr_node_down', 'GCR node failure (AE + rules)'),
                   ('dropout0.1_stuck', '10\\% tags stuck'), ('dropout0.1_imputed', '10\\% tags lost, imputed'),
                   ('dropout0.2_stuck', '20\\% tags stuck'), ('dropout0.2_imputed', '20\\% tags lost, imputed'),
                   ('dropout0.3_stuck', '30\\% tags stuck'), ('dropout0.3_imputed', '30\\% tags lost, imputed')]:
        L.append(f"{lab} & {P(r,'failures',k,'f1')} & {pct(r,'failures',k,'fpr')} & {P(r,'failures',k,'event_recall',d=2)}\\\\")
    w('failures', r'''\begin{table}[!t]
\centering
\caption{Degradation under component failures (anomaly detector, HAI 21.03).}
\label{tab:failures}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
Condition & F1 & FPR (\%) & Event recall\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table}
''')
    e = {k[-1]: v[0] for k, v in r.items() if k[0] == 'efficiency'}
    w('efficiency', r'''\begin{table}[!t]
\centering
\caption{Computational cost on one cloud vCPU (batch size 1, median over 500 calls).}
\label{tab:eff}
\footnotesize
\begin{tabular}{lc}
\toprule
Quantity & Value\\
\midrule
''' + f"""Normalization + lag features & {e['lat_ms_normalise']:.3f} ms\\\\
GCR residuals & {e['lat_ms_gcr']:.3f} ms\\\\
Auto-encoder & {e['lat_ms_ae']:.3f} ms\\\\
Full detector (GCR+AE+rules) & {e['lat_ms_full_detector']:.3f} ms\\\\
Phase/target classifier & {e['lat_ms_classifier']:.2f} ms\\\\
SA engine update (per second of data) & {SAUPD:.4f} ms\\\\
CPU share at 1\\,Hz (detector+classifier) & {100 * e['cpu_fraction_at_1Hz']:.2f}\\%\\\\
Batch throughput & {e['batch_throughput_samples_per_s'] / 1000:.1f}k samples/s\\\\
Detector model size & {e['model_bytes_detector'] / 1024:.0f} KiB\\\\
\\quad of which AE / GCR precision & {e['model_bytes_ae'] / 1024:.0f} / {e['model_bytes_gcr'] / 1024:.0f} KiB\\\\
Classifier model size & {e['model_bytes_classifier'] / 1024:.0f} KiB\\\\
""" + r'''\bottomrule
\end{tabular}
\end{table}
''')


def t_ablation():
    a = load_agg('ablation')
    names = ['1 Detection only', '2 + Correlation & SA', '3 SA + XAI', '4 SA + adaptive response', '5 Full w/o federated learning',
             '6 Full w/o response verification', '7 Full XAI-CSA']
    L = []
    for n in names:
        lab = n.replace('&', '\\&')
        if n.startswith('7'):
            lab = '\\textbf{' + lab + '}'
        L.append(f"{lab} & {P(a,n,'f1')} & {pct(a,n,'fpr')} & {P(a,n,'state_macro_f1')} & {P(a,n,'localisation',d=2)} & {P(a,n,'msr',d=2)} & {P(a,n,'ttc',d=1)} & {P(a,n,'uar',d=2)} & {P(a,n,'disruption_h_per_day',d=2)} & {P(a,n,'escalations_per_day',d=1)}\\\\")
    w('ablation', r'''\begin{table*}[!t]
\centering
\caption{Ablation study on HAI 21.03 (mean$\pm$95\% CI over five seeds; identical splits and model capacity). For configuration 5 each
seed uses the model of a different site, so its CI reflects site-to-site variation.}
\label{tab:ablation}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccccccc}
\toprule
Configuration & F1 & FPR (\%) & State macro-F1 & Localization & MSR & TTC (s) & UAR & Disrupt.\ h/day & Escal./day\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table*}
''')


def t_cross():
    s = load_agg('sa_central_hai-20.07'); b = load_agg('baselines_central_hai-20.07'); d = load_agg('robust_central_hai-20.07')
    L = []
    L.append(f"Isolation Forest & {P(b,'isolation_forest','f1')} & {P(b,'isolation_forest','pr_auc')} & {pct(b,'isolation_forest','fpr')} & {P(b,'isolation_forest','event_recall',d=2)}\\\\")
    L.append(f"Anomaly detector (static) & {P(s,'detection','anomaly_only','f1')} & {P(s,'detection','anomaly_only','pr_auc')} & {pct(s,'detection','anomaly_only','fpr')} & {P(s,'detection','anomaly_only','event_recall',d=2)}\\\\")
    L.append(f"Anomaly detector, drift-aware & {P(d,'drift','clean','drift_aware','f1')} & {P(d,'drift','clean','drift_aware','pr_auc')} & {pct(d,'drift','clean','drift_aware','fpr')} & {P(d,'drift','clean','drift_aware','event_recall',d=2)}\\\\")
    L.append(f"Attack classifier only & {P(s,'detection','classifier_only','f1')} & {P(s,'detection','classifier_only','pr_auc')} & {pct(s,'detection','classifier_only','fpr')} & {P(s,'detection','classifier_only','event_recall',d=2)}\\\\")
    L.append(f"Hybrid (static) & {P(s,'detection','hybrid','f1')} & {P(s,'detection','hybrid','pr_auc')} & {pct(s,'detection','hybrid','fpr')} & {P(s,'detection','hybrid','event_recall',d=2)}\\\\")
    L.append(r'\midrule')
    L.append(f"\\multicolumn{{5}}{{l}}{{State macro-F1: detection-only {P(s,'state','detection_only','macro_f1')}, SA engine {P(s,'state','sa_engine','macro_f1')}}}\\\\")
    L.append(f"\\multicolumn{{5}}{{l}}{{Asset-impact Jaccard: detection-only {P(s,'impact','detection_only','jaccard')}, SA {P(s,'impact','sa_consensus','jaccard')}}}\\\\")
    w('cross', r'''\begin{table}[!t]
\centering
\caption{Second dataset: HAI 20.07 with unchanged hyperparameters (centralized training, two-fold leave-one-file-out).}
\label{tab:cross}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Method & F1 & PR-AUC & FPR (\%) & Ev.\ rec.\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


if __name__ == '__main__':
    import sys
    SAUPD = 0.0
    if have(f'sa_{MAIN}_hai-21.03'):
        SAUPD = load_agg(f'sa_{MAIN}_hai-21.03')[('sa_update_ms_per_s',)][0]
    for fn, need in [(t_detection, f'sa_{MAIN}_hai-21.03'), (t_sa, f'sa_{MAIN}_hai-21.03'), (t_xai, f'xai_{MAIN}'),
                     (t_response, f'response_{MAIN}'), (t_fl, 'fl'), (t_robust, f'robust_{MAIN}'), (t_ablation, 'ablation'),
                     (t_cross, 'sa_central_hai-20.07')]:
        if have(need):
            try:
                fn(); print('ok', fn.__name__)
            except Exception as ex:
                print('FAIL', fn.__name__, repr(ex))
        else:
            print('missing', need)


# ---------------------------------------------------------------- revision tables
def t_suppression():
    a = load_agg('review_fp_fedavg'); c = load_agg('review_fp_central')
    rows = []
    for lab, k in [('Anomaly detector (no suppression)', 'anomaly_only'), ('Suppress alerts classified \\emph{normal}', 'suppress_normal'),
                   ('Suppress alerts classified \\emph{recovery} (XAI-CSA)', 'suppress_recovery (XAI-CSA)'),
                   ('Suppress unless classified \\emph{active}', 'suppress_recovery_or_normal')]:
        rows.append(f"{lab} & {P(a,k,'f1')} & {pct(a,k,'fpr')} & {P(a,k,'event_recall',d=2)} & {P(a,k,'etaf1')} & {P(c,k,'f1')} & {pct(c,k,'fpr')}\\\\")
    w('suppression', r'''\begin{table*}[!t]
\centering
\caption{Ablation of the hybrid detector's suppression mechanism (HAI 21.03, five seeds). eTaF1: eTaPR F1 with
$\theta_p=0.5$, $\theta_r=0.1$, $\delta=0$~\cite{hwang2022etapr}.}
\label{tab:suppression}
\footnotesize
\begin{tabular}{lcccc|cc}
\toprule
& \multicolumn{4}{c|}{Federated detector} & \multicolumn{2}{c}{Centralized detector}\\
Alert rule & F1 & FPR (\%) & Ev.\ rec. & eTaF1 & F1 & FPR (\%)\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table*}
''')


def t_hmm():
    h = load_agg('review_hmm_fedavg')
    rows = []
    for lab, k in [('Detection only (alert $\\Rightarrow$ active, top process)', 'detection_only'),
                   ('SA, per-second $\\arg\\max\\mathbf h_t$, instantaneous target', 'argmax_instant_target'),
                   ('SA, per-second $\\arg\\max\\mathbf h_t$, incident-accumulated target', 'argmax_accumulated_target'),
                   ('SA + HMM filter \\eqref{eq:hmm}, instantaneous target', 'hmm_instant_target'),
                   ('SA + HMM filter \\eqref{eq:hmm}, incident-accumulated target', 'hmm_accumulated_target')]:
        rows.append(f"{lab} & {P(h,k,'acc')} & {P(h,k,'macro_f1')} & {P(h,k,'active_recall')} & {P(h,k,'fullstate_active')}\\\\")
    w('hmm', r'''\begin{table}[!t]
\centering
\caption{Active-phase and affected-asset estimation (federated detector, five seeds). The HMM tempering exponent was selected on the
training folds (mean $\kappa=''' + f"{h[('hmm_kappa_chosen',)][0]:.3f}" + r'''$).}
\label{tab:hmm}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{p{4.3cm}cccc}
\toprule
State estimator & Acc. & Macro-F1 & Active rec. & Full-state\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


def t_cshap():
    s = load_agg('review_cshap_fedavg')
    rows = []
    for lab, k in [('Residual attribution (ours)', 'residual'), ('Shapley, mean baseline', 'shap_mean'), ('Shapley, conditional value function', 'shap_cond')]:
        rows.append(f"{lab} & {P(s,k,'cond_k1')} & {P(s,k,'cond_k5')} & {P(s,k,'mean_k1')} & {P(s,k,'mean_k5')}\\\\")
    w('cshap', r'''\begin{table}[!t]
\centering
\caption{Deletion fidelity under matched and mismatched protocols (one point $30$\,s after the first alert of each detected
episode, three seeds; conditional Shapley with $8$ permutations, mean-baseline Shapley with $16$). A Shapley value function and a deletion protocol are \emph{matched} when both use the same
replacement of absent tags.}
\label{tab:cshap}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Cond.\ deletion} & \multicolumn{2}{c}{Mean deletion}\\
Attribution & $k{=}1$ & $k{=}5$ & $k{=}1$ & $k{=}5$\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


def t_fusion():
    f = load_agg('review_fusion_fedavg'); c = load_agg('review_fusion_central')
    rows = []
    for wgt in ('0.0', '0.25', '0.5', '0.75', '1.0'):
        rows.append(f"{wgt} & {P(f,wgt,'f1')} & {pct(f,wgt,'fpr')} & {P(f,wgt,'evasion_recall',d=2)} & {P(c,wgt,'f1')} & {pct(c,wgt,'fpr')} & {P(c,wgt,'evasion_recall',d=2)}\\\\")
    w('fusion', r'''\begin{table}[!t]
\centering
\caption{Fusion weight $\omega$ on the GCR component ($1-\omega$ on the AE), each recalibrated to the same $0.999$ quantile on held-out
normal data (all columns: three seeds). Evasion: point recall under the PGD attack on the GCR with $\epsilon=3$.}
\label{tab:fusion}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{c|ccc|ccc}
\toprule
& \multicolumn{3}{c|}{Federated} & \multicolumn{3}{c}{Centralized}\\
$\omega$ & F1 & FPR (\%) & Evasion & F1 & FPR (\%) & Evasion\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')


if __name__ == '__main__':
    for fn, need in [(t_suppression, 'review_fp_central'), (t_hmm, 'review_hmm_fedavg'), (t_cshap, 'review_cshap_fedavg'),
                     (t_fusion, 'review_fusion_central')]:
        if have(need):
            try:
                fn(); print('ok', fn.__name__)
            except Exception as ex:
                print('FAIL', fn.__name__, repr(ex))
        else:
            print('missing', need)


def t_sota():
    s = load_agg('sota'); b = load_agg('baselines_central_hai-21.03'); oc = load_agg('ours_sota_central'); of = load_agg('ours_sota_fedavg')
    def row(lab, a, *pre, n=''):
        g = lambda m, d=3: P(a, *pre, m, d=d) if tuple(pre) + (m,) in a else '--'
        return f"{lab} & {g('f1')} & {g('f1_oracle')} & {g('pr_auc')} & {(pct(a,*pre,'fpr') if tuple(pre)+('fpr',) in a else '--')} & {g('event_recall',2)} & {g('etaf1')}\\\\"
    L = [r'\multicolumn{7}{l}{\textit{(a) Re-implemented under the identical protocol (normal-data calibration, same splits)}}\\']
    L.append(row('Isolation Forest~\\cite{liu2008isolation}', b, 'isolation_forest'))
    L.append(row('PCA residual', b, 'pca'))
    L.append(row('Deep SVDD~\\cite{ruff2018deep}$^\\ddagger$', s, 'deep_svdd'))
    L.append(row('USAD~\\cite{audibert2020usad}$^\\ddagger$', s, 'usad'))
    L.append(row('GDN~\\cite{deng2021graph}$^\\ddagger$', s, 'gdn'))
    L.append(row('XAI-CSA anomaly detector (centralized)', oc, 'anomaly'))
    L.append(row('XAI-CSA anomaly detector (federated)', of, 'anomaly'))
    L.append(row('\\textbf{XAI-CSA hybrid (centralized)}$^\\S$', oc, 'hybrid'))
    L.append(row('\\textbf{XAI-CSA hybrid (federated)}$^\\S$', of, 'hybrid'))
    L.append(r'\midrule\multicolumn{7}{l}{\textit{(b) Published results on HAI 21.03 (reported values; protocols differ, see text)}}\\')
    pub = [('RANSynCoder~\\cite{kim2023comparative}', '--', '0.829', '--', '--', '--', '--'),
           ('InterFusion~\\cite{kim2023comparative}', '--', '0.789', '--', '--', '--', '--'),
           ('LSTM-ED~\\cite{kim2023comparative}', '--', '0.717', '--', '--', '--', '--'),
           ('GDN~\\cite{kim2023comparative}', '--', '0.597', '--', '--', '--', '--'),
           ('USAD~\\cite{kim2023comparative}', '--', '0.588', '--', '--', '--', '--'),
           ('Stacked-GRU ensemble~\\cite{kim2021ensemble}', '--', '0.751', '--', '--', '0.96', '0.938$^\\ast$'),
           ('SIMPLE~\\cite{wolsing2022simple,wolsing2025gecos}', '0.547', '--', '--', '--', '--', '0.717$^\\ast$'),
           ('GeCo~\\cite{wolsing2025gecos}', '0.639', '--', '--', '--', '--', '0.694$^\\ast$'),
           ('Physics features + IF~\\cite{chahine2026features}', '0.581', '--', '--', '--', '--', '0.635')]
    for p_ in pub:
        L.append(' & '.join(p_) + '\\\\')
    w('sota', r'''\begin{table*}[!t]
\centering
\caption{Comparison with state-of-the-art detectors on HAI 21.03. F1: threshold calibrated on normal data only; F1$^\star$:
best F1 over all thresholds (oracle, uses test labels); eTaF1: eTaPR F1 ($\theta_p{=}0.5$, $\theta_r{=}0.1$, $\delta{=}0$).
Block (a): mean$\pm$95\% CI, five seeds (three for $^\ddagger$); all rows are unsupervised except the hybrid detector,
whose recovery classifier is trained on attack labels of the other test files ($^\S$). $^\ddagger$Our re-implementation following the original
architecture (no deep-learning framework was available; see Section~\ref{sec:sota}). $^\ast$eTaPR parameters not reported by
the source.}
\label{tab:sota}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Method & F1 & F1$^\star$ & PR-AUC & FPR (\%) & Ev.\ rec. & eTaF1\\
\midrule
''' + '\n'.join(L) + r'''
\bottomrule
\end{tabular}
\end{table*}
''')


if __name__ == '__main__':
    if have('sota') and have('ours_sota_central'):
        try:
            t_sota(); print('ok t_sota')
        except Exception as ex:
            print('FAIL t_sota', repr(ex))
