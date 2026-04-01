# SDL Causal Analysis

**Causal inference for self-driving laboratory optimisation strategies**

---

## What this project is

Self-driving laboratories (SDLs) use optimisation algorithms — Bayesian optimisation, random search, evolutionary strategies — to autonomously navigate experimental spaces. A persistent open question is: *which strategy should be used, when, and why?*

This project applies causal inference (DoWhy) to answer that question rigorously. Rather than asking "which strategy correlated with better outcomes?", we ask "what is the causal effect of strategy choice on campaign performance, and what landscape and experimental conditions moderate that effect?" The goal is to derive actionable design rules that an adaptive campaign planner can use to select strategies automatically.

The work is domain-general (any SDL) but is grounded in pharmaceutical formulation — specifically the discovery of high-solubility liquid formulations for injectable medicines — as the primary translation target.

---

## The ten use cases

Each use case is a distinct causal question. They are ordered from conceptually simple to complex, and each builds on the reasoning established in the previous ones.

| # | Question | DoWhy feature |
|---|---|---|
| 1 | Does strategy choice causally improve campaign outcomes, after adjusting for landscape and experimental confounders? | Backdoor adjustment |
| 2 | Why did a strategy fail — which upstream factor was the root cause? | Do-calculus, counterfactuals |
| 3 | Does BO converge faster than random search, and when does the advantage emerge? | Time-varying ATE (OLS interaction + cluster-robust SEs) |
| 4 | Are the confounders in our simulation design correctly identified? | DAG identification |
| 5 | What would have happened if we had used a different strategy from iteration N? | Counterfactual estimation |
| 6 | Which landscape features causally determine whether BO outperforms random search? | Causal discovery + DoWhy |
| 7 | Are our derived design rules robust to hidden confounders? | Refutation tests |
| 8 | How does a strategy achieve its effect — through model quality or acquisition behaviour? | Mediation analysis |
| 9 | When does strategy X beat strategy Y, as a function of campaign conditions? | CATE / EconML |
| 10 | Does physical constraint severity causally degrade strategy performance? | Frontdoor adjustment |

---

## Dataset strategy

**Primary benchmark**: [Olympus](https://github.com/aspuru-guzik-group/olympus) — 33 experimentally-derived benchmark surfaces from chemistry and materials science, emulated via Bayesian neural networks. Provides realistic noise, known ground truth, and 23 optimisation planners. Used to generate hundreds of simulated campaigns across varied strategies, landscapes, budgets, and noise levels.

**Pharmaceutical anchor**: The HPLC dataset within Olympus (6 continuous parameters, peak area outcome, pharmaceutical instrument noise) is the closest available analogue to the medicine formulation target. When real solubility data becomes available, the emulator is swapped for a GP surrogate trained on that data — the causal analysis code is unchanged.

**Curcumin formulation data**: The 256-point dataset from [Ros et al. (2025)](https://doi.org/10.1039/D5DD00171D) is used as a realistic pharmaceutical landscape for the simulator (use cases 6 and 8), where the unit of analysis is the formulation rather than the campaign.

**What needs to be tailored for medicine formulation**: When real solubility data is available, (1) retrain the landscape emulator on that data, (2) verify the noise model matches experimental noise, (3) check whether design rules derived from HPLC generalise to the formulation space.

---

## Implemented use cases

| UC | Status | Notebook | Reasoning doc | Design rules |
|---|---|---|---|---|
| UC1 | Complete | `notebooks/uc1_backdoor_adjustment.ipynb` | `uc1_dag_reasoning.md` | DR-A: BO ATE = +0.10 after backdoor adjustment; 4 refutation tests pass (CI breach at hidden confounder eff ≥ 0.05) |
| UC2 | Complete | `notebooks/uc2_noise_moderation.ipynb` | `uc2_dag_reasoning.md` | DR-B/C: BO advantage collapses below SNR τ=2.0 |
| UC3 | Complete | `notebooks/uc3_convergence_curves.ipynb` | `uc3_dag_reasoning.md` | DR-D: BO warmup ~10% of budget; dim drives delay |
| UC4 | Complete | `notebooks/uc4_confounder_sensitivity.ipynb` | `uc4_dag_reasoning.md` | DR-E/F/G: omission +135%, collider −41% (Berkson) |
| UC5 | Complete | `notebooks/uc5_att_strategy_switching.ipynb` | `uc5_dag_reasoning.md` | DR-H: ATT = +0.073 > ATE = +0.069; switching loss concentrated in high-SNR campaigns |
| UC6 | Complete | `notebooks/uc6_distribution_shift.ipynb` | `uc6_dag_reasoning.md` | DR-I/J/K/L: BO 1.35× more sensitive to noise shifts; static adjustment set overestimates BO advantage; monitor GP residual variance mid-campaign |

---

## UC6: Mid-Campaign Distribution Shift

**Scientific question**: Does a mid-campaign noise increase causally degrade campaign outcome? Does the degradation depend on when the shift occurs and how large it is? Is BO more sensitive than random search?

**New causal variables**:
- `shift_time` — step at which noise increases, drawn from Uniform(0.2 × budget, 0.8 × budget)
- `shift_magnitude` — noise multiplier, drawn from LogNormal with E[·] = 3.0×
- `post_shift` — binary indicator: 1 if step > shift_time

**Identification challenge**: The static backdoor adjustment set from UC2/UC3 `{budget, dim, ls, nl, snr}` is insufficient. Practitioners avoid BO when large shifts are expected (`shift_magnitude → strategy`, weight = −0.15), so BO campaigns face systematically smaller shifts. The static estimator therefore **overestimates** the BO advantage (bias = +0.009 in simulation). The correct adjustment set adds `shift_time` and `shift_magnitude`.

**Estimator**: Interrupted Time Series (ITS) with DiD interpretation and cluster-robust standard errors (clustered by campaign). The outcome variable is `latent_outcome_t` (instantaneous step quality), which directly captures the structural shift effect. The key coefficient is `strategy:log_step:post_shift` (three-way interaction) — a negative value (−0.014, p = 0.047) confirms BO's convergence rate advantage shrinks post-shift. The `post_shift` main effect (level drop) is −0.064, p < 0.001, confirming an immediate quality drop at the shift point. The rate change is the robust signal for the BO-specific degradation.

**Mechanism**: Post-shift, the effective SNR drops (median pre-shift SNR ≈ 8.24 → post-shift ≈ 2.93 with 3× noise increase), pushing many campaigns below the τ = 2.0 threshold and collapsing the BO advantage. BO's GP surrogate becomes miscalibrated; RS is unaffected because it does not use a surrogate model.

**Design rules**:
- **DR-UC6-A**: Earlier shifts and larger shifts cause more outcome degradation (both confirmed in simulation)
- **DR-UC6-B**: BO is 1.35× more sensitive to noise shifts than RS (improvement rate drop: BO = 0.0123, RS = 0.0092)
- **DR-UC6-C**: Static adjustment set is insufficient when noise is time-varying; always include shift parameters in the adjustment set
- **DR-UC6-D**: Monitor GP residual variance mid-campaign to detect shifts; if post-shift SNR < τ = 2.0, consider switching to RS

---

## Repository structure

```
sdl-causal/
├── README.md
├── environment.yml              # verified compatible dependencies
├── notebooks/
│   ├── uc1_backdoor_adjustment.ipynb
│   ├── uc2_noise_moderation.ipynb
│   ├── uc3_convergence_curves.ipynb
│   ├── uc4_confounder_sensitivity.ipynb
│   ├── uc5_att_strategy_switching.ipynb
│   └── uc6_distribution_shift.ipynb
├── src/
│   ├── simulator.py             # UC1–UC6 campaign simulator
│   └── graphs.py                # DAG definitions (UC1–UC6)
├── tests/
│   ├── test_uc2.py
│   ├── test_uc3.py
│   ├── test_uc4.py
│   ├── test_uc5.py
│   └── test_uc6.py              # 55 tests (51 pass, 4 skipped — DoWhy not installed)
└── uc*_dag_reasoning.md         # scientific reasoning per use case
```

---

## How to reproduce

**1. Install system dependencies (Linux/Debian)**
```bash
apt-get install -y graphviz graphviz-dev
```

**On macOS**
```bash
brew install graphviz
```

**2. Create the environment**
```bash
conda env create -f environment.yml
conda activate sdl-causal
```

**3. Open the notebook**
```bash
jupyter notebook notebooks/causal_analysis.ipynb
```

---

## Commit conventions

Each commit answers one of:
- `reasoning: [UC-N] <what was decided and why>` — scientific modelling decisions
- `dag: [UC-N] <what was added>` — DAG definition in code
- `sim: [UC-N] <what was built>` — simulator or data generation
- `analysis: [UC-N] <what was run>` — DoWhy estimation and refutation
- `rule: [UC-N] <what was extracted>` — design rule derived from results

This makes the commit history a readable scientific log, not just a code log.

---

## Key references

- Pearl, J. & Mackenzie, D. (2018). *The Book of Why*. Basic Books.
- Neal, B. (2020). *Introduction to Causal Inference*. https://www.bradyneal.com/causal-inference-course
- Häse, F. et al. (2021). Olympus: a benchmarking framework for noisy optimization. *Mach. Learn.: Sci. Technol.* https://doi.org/10.1088/2632-2153/abedc8
- Ros, H. et al. (2025). Efficient discovery of new medicine formulations using a semi-self-driven robotic formulator. *Digital Discovery*. https://doi.org/10.1039/D5DD00171D
- DoWhy documentation: https://py-why.github.io/dowhy/
