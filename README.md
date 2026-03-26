# SDL Causal Analysis

**Causal inference for self driving laboratory optimisation strategies**

---

## What this project is

Self driving laboratories (SDLs) use optimisation algorithms - Bayesian optimisation, random search, evolutionary strategies - to autonomously navigate experimental spaces. A persistent open question is: *which strategy should be used, when, and why?*

This project applies causal inference (DoWhy) to answer that question rigorously. Rather than asking "which strategy correlated with better outcomes?", we ask "what is the causal effect of strategy choice on campaign performance, and what landscape and experimental conditions moderate that effect?" The goal is to derive actionable design rules that an adaptive campaign planner can use to select strategies automatically.

The work is domain general (any SDL) but is grounded in pharmaceutical formulation - specifically the discovery of high solubility liquid formulations for injectable medicines - as the primary translation target.

---

## The ten use cases

Each use case is a distinct causal question. They are ordered from conceptually simple to complex, and each builds on the reasoning established in the previous ones.

| # | Question | DoWhy feature |
|---|---|---|
| 1 | Does strategy choice causally improve campaign outcomes, after adjusting for landscape and experimental confounders? | Backdoor adjustment |
| 2 | Why did a strategy fail - which upstream factor was the root cause? | Do-calculus, counterfactuals |
| 3 | Did a mid-campaign strategy switch causally improve subsequent performance? | Instrumental variables |
| 4 | Are the confounders in our simulation design correctly identified? | DAG identification |
| 5 | What would have happened if we had used a different strategy from iteration N? | Counterfactual estimation |
| 6 | Which landscape features causally determine whether BO outperforms random search? | Causal discovery + DoWhy |
| 7 | Are our derived design rules robust to hidden confounders? | Refutation tests |
| 8 | How does a strategy achieve its effect - through model quality or acquisition behaviour? | Mediation analysis |
| 9 | When does strategy X beat strategy Y, as a function of campaign conditions? | CATE / EconML |
| 10 | Does physical constraint severity causally degrade strategy performance? | Frontdoor adjustment |

---

## Dataset strategy

**Primary benchmark**: [Olympus](https://github.com/aspuru-guzik-group/olympus) - 33 experimentally derived benchmark surfaces from chemistry and materials science, emulated via Bayesian neural networks. Provides realistic noise, known ground truth, and 23 optimisation planners. Used to generate hundreds of simulated campaigns across varied strategies, landscapes, budgets, and noise levels.

**Pharmaceutical anchor**: The HPLC dataset within Olympus (6 continuous parameters, peak area outcome, pharmaceutical instrument noise) is the closest available analogue to the medicine formulation target. When real solubility data becomes available, the emulator is swapped for a GP surrogate trained on that data - the causal analysis code is unchanged.

**Curcumin formulation data**: The 256 point dataset from [Ros et al. (2025)](https://doi.org/10.1039/D5DD00171D) is used as a realistic pharmaceutical landscape for the simulator (use cases 6 and 8), where the unit of analysis is the formulation rather than the campaign.

**What needs to be tailored for medicine formulation**: When real solubility data is available, (1) retrain the landscape emulator on that data, (2) verify the noise model matches experimental noise, (3) check whether design rules derived from HPLC generalise to the formulation space.

---

## Repository structure

```
sdl-causal/
├── README.md
├── environment.yml              # verified compatible dependencies
├── notebooks/
│   └── causal_analysis.ipynb   # single living notebook, one section per use case
├── src/
│   ├── simulator.py             # Olympus campaign simulator wrapper
│   └── graphs.py                # DAG definitions, one per use case
├── results/
│   └── design_rules.md          # extracted design rules, updated as each UC completes
└── docs/
    ├── uc1_dag_reasoning.md     # scientific reasoning: UC1
    ├── uc2_dag_reasoning.md     # scientific reasoning: UC2
    └── ...
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
- `reasoning: [UC-N] <what was decided and why>` - scientific modelling decisions
- `dag: [UC-N] <what was added>` - DAG definition in code
- `sim: [UC-N] <what was built>` - simulator or data generation
- `analysis: [UC-N] <what was run>` - DoWhy estimation and refutation
- `rule: [UC-N] <what was extracted>` - design rule derived from results

This makes the commit history a readable scientific log.

---

## Key references

- Pearl, J. & Mackenzie, D. (2018). *The Book of Why*. Basic Books.
- Neal, B. (2020). *Introduction to Causal Inference*. https://www.bradyneal.com/causal-inference-course
- Häse, F. et al. (2021). Olympus: a benchmarking framework for noisy optimization. *Mach. Learn.: Sci. Technol.* https://doi.org/10.1088/2632-2153/abedc8
- Ros, H. et al. (2025). Efficient discovery of new medicine formulations using a semi-self-driven robotic formulator. *Digital Discovery*. https://doi.org/10.1039/D5DD00171D
- DoWhy documentation: https://py-why.github.io/dowhy/
