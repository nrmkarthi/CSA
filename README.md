# XAI-CSA: Explainable Self-Adaptive Cyber Situation Awareness for IoT

Source code and per-seed results for the paper

> M. M. Aborokbah, M. Shanmuganathan, Karthikeyan P, A. Paul, H. Aldawood, and A. Aljuhani,
> "Explainable Self-Adaptive Cyber Awareness for IoT Threat Detection and Safe Response,"
> submitted to *IEEE Internet of Things Journal*, 2026.

XAI-CSA is a six-layer framework for industrial IoT (IIoT) security. It does more than raise alerts: it builds a
situation picture (what is happening, which assets are affected and what is likely to happen next), explains that
picture in terms of the incident and the action taken, and chooses defensive actions that a safety controller has
checked and that are verified after execution. The repository reproduces every number, table and figure in the paper on
the public HAI 21.03 and HAI 20.07 industrial control system datasets.

---

## Framework at a glance

| Layer | What it does | Code |
|---|---|---|
| 1 | Event acquisition (sensor, actuator and controller tags) | `prep.py`, `common.py` |
| 2–3 | Normalization and a hybrid detector: Gaussian conditional residual (GCR) model + MLP auto-encoder + invariant rules, whose fused score decomposes **exactly** into per-tag attributions; gradient-boosted attack classifier (phase and target sub-process) | `detect.py`, `sa.py` |
| 4 | Situation awareness: perception (alert events), comprehension (incident correlation, phase, affected assets, optional HMM phase filter) and projection (attack continuation, asset propagation) | `sa.py`, `core.py` |
| 4 | Situation-grounded explanations: incident-level attribution diffused over the asset-dependency graph | `xai.py` |
| 5–6 | Multi-agent response: triage, tabular Q-learning agent masked by a safety and policy controller, post-action verification, trace-driven response simulator | `response.py` |
| Governance | Federated learning (exact aggregation of sufficient statistics for the GCR model, FedAvg / trimmed-mean for the auto-encoder, optional DP-style noise and a poisoning client), drift monitoring | `fl.py` |

Everything runs on CPU (the paper's experiments used a 2-vCPU cloud VM without a GPU). No deep-learning framework is
needed. The state-of-the-art baselines (USAD, GDN, Deep SVDD) are NumPy re-implementations built on a small autodiff
engine in `nn.py`.

---

## Repository layout

```
CSA/
├── README.md
├── requirements.txt
├── common.py           shared utilities: paths, data loading, metrics, 95 % confidence intervals
├── prep.py             HAI CSV files -> cache/*.npz
├── detect.py           Layers 2-3: normalizer, GCR + AE + invariant rules, exact attributions
├── sa.py               Layer 3 classifier; Layer 4 perception / comprehension / projection
├── xai.py              attribution, Shapley sampling, situation-grounded attribution, explanation metrics
├── response.py         Layers 5-6: Q-learning agent, safety controller, verification, simulator
├── fl.py               federated learning, differential-privacy noise, poisoning
├── core.py             builds the per-file situation picture from cached detector outputs
├── nn.py               minimal NumPy autodiff (used only by the SOTA baselines)
├── sota.py             re-implemented USAD, GDN and Deep SVDD under the same protocol
├── etapr_util.py       wrapper around the reference eTaPR implementation
├── pipeline.py         runs Layers 2-4 per seed and caches out-of-fold outputs
├── exp_baselines.py    detection baselines (Isolation Forest, PCA, GCR-only, AE-only)
├── exp_sa.py           RQ1: detection, event correlation, situation awareness
├── exp_xai.py          RQ2: explanation fidelity, stability, localization
├── exp_response.py     RQ3: safety-constrained adaptive response
├── exp_fl.py           RQ4: federated learning, DP noise, poisoning
├── exp_robust.py       RQ4/RQ5: zero-day, drift, adversarial evasion, component failure, efficiency
├── exp_ablation.py     RQ5: seven ablation configurations
├── exp_review.py       additional analyses: FP suppression, HMM filter, conditional Shapley, fusion weights
├── exp_ours_sota.py    oracle-threshold F1 and eTaPR of the XAI-CSA detectors
├── agg.py              aggregates per-seed results into mean ± 95 % CI
├── gen_tables.py       writes the LaTeX tables of the paper
├── make_figs.py        writes the result figures of the paper
├── fig_timeline.py     writes the qualitative incident-timeline figure
├── figstyle.py         shared plotting style
└── results/            per-seed JSON results used in the paper (see below)
```

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/nrmkarthi/CSA.git
cd CSA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# eTaPR reference implementation (range-based detection metric)
git clone https://github.com/saurf4ng/eTaPR ext/eTaPR
```

`etapr_util.py` looks for eTaPR in `ext/eTaPR`. To use a clone stored elsewhere, set `ETAPR_PATH`.

---

## Quick start: regenerate the paper's tables and figures from the included results

No dataset is needed for this step. The `results/` folder already holds the per-seed outputs used in the paper.

```bash
python3 gen_tables.py          # -> paper/tables/*.tex
python3 make_figs.py fedavg    # -> paper/figs/{corr_sweep,xai_fidelity,resp_tradeoff,robust}.{pdf,png}
```

`fig_timeline.py` (the qualitative incident timeline) needs the cached detector outputs. Run it after the full
pipeline below.

---

## Full reproduction

### 1. Get the data

Download the HAI security datasets from <https://github.com/icsdataset/hai> and place the two versions under
`data/hai/`:

```
data/hai/hai-21.03/train1.csv.gz ... train3.csv.gz, test1.csv.gz ... test5.csv.gz
data/hai/hai-20.07/train1.csv.gz, train2.csv.gz, test1.csv.gz, test2.csv.gz
```

Then convert them once:

```bash
python3 prep.py                # -> cache/<version>_<file>.npz
```

To use another dataset (e.g., SWaT or WADI), convert it to the same layout (`X` = tag matrix, `L` = `[attack,
attack_P1, attack_P2, attack_P3]`) and add a branch to `common.dataset()`.

### 2. Train detectors and cache out-of-fold outputs (5 seeds)

```bash
python3 pipeline.py hai-21.03 central                         # centralized training
FL_ROUNDS=50 python3 pipeline.py hai-21.03 fedavg             # federated training
for i in 0 1 2 3 4; do python3 pipeline.py hai-21.03 local$i $i; done   # single-site models
python3 pipeline.py hai-20.07 central                         # second dataset
```

### 3. Run the experiments

| Research question / analysis | Command |
|---|---|
| Detection baselines | `python3 exp_baselines.py hai-21.03 central` <br> `python3 exp_baselines.py hai-20.07 central` |
| **RQ1** detection, correlation, situation awareness | `python3 exp_sa.py central` <br> `python3 exp_sa.py fedavg` <br> `python3 exp_sa.py central hai-20.07` |
| **RQ2** situation-grounded explainability | `python3 exp_xai.py fedavg` |
| **RQ3** safety-constrained adaptive response | `python3 exp_response.py fedavg` |
| **RQ4** federated learning, DP noise, poisoning | `python3 exp_fl.py base` <br> `SEEDS_ONLY=0,1,2 python3 exp_fl.py dp,poison` |
| **RQ4/RQ5** robustness and efficiency | `python3 exp_robust.py fedavg zeroday,drift,fail,eff` <br> `OUTSUF=_adv SEEDS_ONLY=0,1,2 python3 exp_robust.py fedavg adv` <br> `VER=hai-20.07 python3 exp_robust.py central drift` |
| **RQ5** ablation (7 configurations) | `python3 exp_ablation.py` |
| FP suppression, HMM phase filter, conditional Shapley, fusion weights | `TAG=fedavg python3 exp_review.py fp,hmm,cshap,fusion` <br> `TAG=central python3 exp_review.py fp,fusion` |
| Comparison with state-of-the-art detectors | `python3 sota.py` (USAD, GDN, Deep SVDD; 3 seeds) <br> `python3 exp_ours_sota.py` |

### 4. Build the paper assets

```bash
python3 gen_tables.py
python3 make_figs.py fedavg
python3 fig_timeline.py
```

When several jobs run in parallel, set `OMP_NUM_THREADS=1` for each one.

---

## Included results

`results/` contains one JSON file per experiment, keyed by seed. `agg.py` turns them into mean ± 95 % confidence
intervals.

| File | Produced by |
|---|---|
| `baselines_central_hai-21.03.json`, `baselines_central_hai-20.07.json` | `exp_baselines.py` |
| `sa_central_hai-21.03.json`, `sa_fedavg_hai-21.03.json`, `sa_central_hai-20.07.json` | `exp_sa.py` |
| `xai_fedavg.json` | `exp_xai.py` |
| `response_fedavg.json` | `exp_response.py` |
| `fl.json` | `exp_fl.py` |
| `robust_fedavg.json`, `robust_fedavg_adv.json`, `robust_central_hai-20.07.json` | `exp_robust.py` |
| `ablation.json` | `exp_ablation.py` |
| `review_fp_*.json`, `review_hmm_fedavg.json`, `review_cshap_fedavg.json`, `review_fusion_*.json` | `exp_review.py` |
| `sota.json` | `sota.py` |
| `ours_sota_central.json`, `ours_sota_fedavg.json` | `exp_ours_sota.py` |

Running an experiment again overwrites (or extends, seed by seed) the matching file.

---

## Configuration

All paths default to folders inside the repository. Override them with environment variables if needed:

| Variable | Default | Purpose |
|---|---|---|
| `HAI_DATA` | `data/hai` | folder containing `hai-21.03/` and `hai-20.07/` |
| `XAICSA_ROOT` | repository folder | base folder for the defaults below |
| `XAICSA_CACHE` | `cache/` | preprocessed data and cached detector outputs |
| `XAICSA_RESULTS` | `results/` | per-seed JSON results |
| `XAICSA_TABLES` | `paper/tables/` | generated LaTeX tables |
| `XAICSA_FIGS` | `paper/figs/` | generated figures |
| `ETAPR_PATH` | `ext/eTaPR` | clone of the eTaPR reference implementation |

Experiment-level switches: `SEEDS_ONLY` (comma-separated seeds, e.g. `0,1,2`), `FL_ROUNDS` (federated rounds),
`TAG` (`central` or `fedavg` for `exp_review.py`), `MAIN_TAG` (detector used by `exp_ablation.py`), `VER`
(dataset version for `exp_robust.py`) and `OUTSUF` (suffix of the output file).

---

## Data

The HAI datasets are not redistributed here. They are publicly available from the HAI security dataset repository
(<https://github.com/icsdataset/hai>) under that repository's terms.

## Citation

Code: <https://github.com/nrmkarthi/CSA>

If you use this code, please cite:

```bibtex
@article{aborokbah2026xaicsa,
  author  = {Aborokbah, Majed M. and Shanmuganathan, Manimurugan and P, Karthikeyan and
             Paul, Anand and Aldawood, Hussain and Aljuhani, Ahmed},
  title   = {Explainable Self-Adaptive Cyber Awareness for {IoT} Threat Detection and Safe Response},
  journal = {IEEE Internet of Things Journal},
  note    = {Submitted},
  year    = {2026}
}
```

## Contact

Majed M. Aborokbah (corresponding author), Faculty of Computers and Information Technology, University of Tabuk,
Saudi Arabia: m.aborokbah@ut.edu.sa
