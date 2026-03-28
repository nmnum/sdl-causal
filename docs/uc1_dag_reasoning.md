# Use Case 1: Causal Effect of Strategy Choice on Campaign Outcome

**Status**: Complete — reasoning, implementation, and refutation done  
**DoWhy feature**: Backdoor adjustment (linear regression)  
**Unit of analysis**: Campaign (one row = one complete experimental campaign)

---

## 1. The scientific question

> Does the choice of optimisation strategy (Bayesian optimisation vs. random search) causally improve the best outcome found in a campaign, after adjusting for differences in landscape properties, experimental noise, and budget?

The word *causally* is load-bearing. We are not asking whether BO correlates with better outcomes — it does, trivially, because practitioners tend to apply BO to problems where they expect it to work. We are asking: if we took two campaigns that were identical in every other respect and changed only the strategy, how much would the outcome differ?

This distinction matters for the adaptive planner. A design rule derived from naive correlation ("BO campaigns achieve higher solubility") is not actionable — it may simply reflect that BO was used on easier problems. A design rule derived from a causal estimate ("BO causally improves normalised best outcome by X, after adjusting for landscape smoothness and budget") is actionable.

---

## 2. Variable classification

| Variable | Description | Role | Justification |
|---|---|---|---|
| `strategy` | BO vs. random search | **Treatment** | The intervention we are studying |
| `norm_best_outcome` | Best outcome found / true optimum | **Outcome** | Campaign-level, scale-free, comparable across landscapes |
| `landscape_smoothness` | Estimated smoothness of the objective surface (e.g. length-scale from GP fit) | **Confounder** | Practitioners choose BO on smoother landscapes; smoother landscapes also yield higher outcomes |
| `budget` | Number of experiments allowed | **Confounder** | Small budgets favour random search (BO overhead not worth it); larger budgets also improve outcomes |
| `noise_level` | Measurement noise standard deviation | **Confounder** | High noise discourages BO (surrogate model degrades); high noise also reduces outcome quality |
| `dimensionality` | Number of input variables | **Confounder** | High dimensionality discourages standard BO; also makes optimisation harder |
| `initial_design_quality` | Space-filling score of seed points | **Non-confounder** | Affects outcome but not strategy choice in our simulation (assigned independently) |
| `batch_id` | Experimental batch identifier | **Noise variable** | Affects outcome via reagent variability but does not affect strategy choice |

### Why `initial_design_quality` is not a confounder

In a real SDL, a practitioner might choose BO precisely because they have a good initial design. In our simulation, initial design quality is assigned independently of strategy — it is a controlled variable, not a confounder. This is a deliberate simulation design choice that we must document and revisit when translating to real data.

### Why `batch_id` is not a confounder

Batch effects (reagent lot, instrument day) affect the measured outcome but are not known to the practitioner at the time of strategy selection. They are therefore not on the causal path from strategy to outcome, and not a confounder. They contribute to outcome variance (noise) but do not bias the causal estimate.

---

## 3. The causal DAG

```
landscape_smoothness ──► strategy ──────────────────► norm_best_outcome
budget               ──► strategy                          ▲
noise_level          ──► strategy                          │
dimensionality       ──► strategy                          │
                                                           │
landscape_smoothness ──────────────────────────────────────┤
budget               ──────────────────────────────────────┤
noise_level          ──────────────────────────────────────┤
dimensionality       ──────────────────────────────────────┤
initial_design_quality ────────────────────────────────────┤
batch_id (noise) ──────────────────────────────────────────┘
```

Formally, the confounders are `{landscape_smoothness, budget, noise_level, dimensionality}`. These are the variables with arrows into both `strategy` and `norm_best_outcome`.

---

## 4. Backdoor paths and identification

A backdoor path is any path from `strategy` to `norm_best_outcome` that enters `strategy` through an incoming arrow (i.e. goes through a confounder). The backdoor paths here are:

1. `strategy ← landscape_smoothness → norm_best_outcome`
2. `strategy ← budget → norm_best_outcome`
3. `strategy ← noise_level → norm_best_outcome`
4. `strategy ← dimensionality → norm_best_outcome`

**Backdoor criterion**: A set Z satisfies the backdoor criterion if (i) no node in Z is a descendant of `strategy`, and (ii) Z blocks all backdoor paths. The set `{landscape_smoothness, budget, noise_level, dimensionality}` satisfies both conditions. DoWhy's `identify_effect()` will confirm this automatically.

**Identification**: The causal effect is identifiable by adjusting for `{landscape_smoothness, budget, noise_level, dimensionality}`. No instrumental variables or frontdoor adjustment are needed.

---

## 5. Outcome variable choice and justification

**Chosen outcome**: `norm_best_outcome = best_found / true_optimum`

This is the normalised best outcome — the fraction of the true optimum recovered by the end of the campaign. Equivalently, `simple_regret = 1 - norm_best_outcome`.

**Why not raw outcome value**: Raw values are not comparable across landscapes with different optima. A campaign on a landscape with optimum 10 mg/mL and a campaign on a landscape with optimum 100 mg/mL cannot be compared on the raw scale.

**Why not log(concentration)**: Log-concentration is the correct outcome for real medicine formulation data (where the true optimum is unknown). In simulation, we know the true optimum, so normalised best outcome is strictly more informative.

**Why not binary (found optimum yes/no)**: Too sparse and loses information about how close the campaign got.

**Translation to real data**: When applying this to real medicine formulation campaigns, substitute `norm_best_outcome` with `log(best_dissolved_concentration_found)`. The causal model structure is unchanged; only the outcome scale changes.

---

## 6. Refutation results (n=2000, seed=42)

Four tests were run against the backdoor-adjusted estimate (ATE = +0.096, 95% CI [+0.088, +0.104]).
The true ATE (+0.10) lies inside the CI. All binary tests pass.

| Test | New effect | p-value | Pass | What it checks |
|---|---|---|---|---|
| Random common cause | +0.096 | 0.96 | Yes | Estimate stable when a random variable is added as a confounder |
| Placebo treatment | +0.000 | 0.88 | Yes | Effect collapses to ~0 when treatment is permuted |
| Data subset (80%) | +0.096 | 0.94 | Yes | Estimate stable on a random 80% subsample |
| Unobserved confounder | See curve | — | — | How strong would a hidden confounder need to be to invalidate DR-A? |

**Unobserved confounder sensitivity** (binary flip on treatment, linear on outcome, effect on outcome = 2× effect on treatment):

| Effect on treatment | New ATE | Shift from adjusted |
|---|---|---|
| 0.01 | +0.094 | −0.003 |
| 0.02 | +0.091 | −0.005 |
| 0.05 | +0.084 | −0.012 |
| 0.10 | +0.074 | −0.023 |
| 0.15 | +0.064 | −0.032 |
| 0.20 | +0.059 | −0.038 |
| 0.25 | +0.052 | −0.045 |
| 0.30 | +0.042 | −0.054 |
| 0.35 | +0.032 | −0.064 |
| 0.40 | +0.021 | −0.075 |

**Sensitivity parameterisation**: `effect_strength_on_outcome = 2 × effect_strength_on_treatment` throughout the sweep. This means a confounder with eff = 0.05 on treatment has eff = 0.10 on outcome — an effect on outcome equal in magnitude to the true ATE itself. The functional form is binary flip on treatment, linear on outcome (DoWhy defaults).

**CI breach point**: A hidden confounder with effect strength ≥ 0.05 on treatment (and therefore ≥ 0.10 on outcome under the 2× parameterisation) is sufficient to push the mean estimate below the 95% CI lower bound (+0.089). Each strength value is averaged over **10 seeds** to suppress Monte Carlo noise from the DoWhy refuter (N increased from 5 to 10 to halve per-point std at high effect strengths). The grid is denser at the high end (0.20–0.40) where noise is largest. Stress-testing across 20 independent seed offsets confirms 0/20 non-monotone trials. This is a moderately strong confounder — it would need to flip ~5% of treatment assignments and shift outcomes by ~0.10 units.

**Interpretation**: DR-A is well-supported. The estimate is stable across all three binary tests, the placebo collapses correctly, and a hidden confounder would need to be of moderate strength to explain away the result. The residual caveat is that the sensitivity analysis uses a specific functional form for the hidden confounder; a different structure could have a different effect.

**Visualisation**: `uc1_refutation.png`

---

## 7. Ground truth specification for the simulator

### 6a. Empirical derivation of the BO advantage (updated 2026-03-26)

The `+0.15` placeholder has been replaced with an empirically grounded value derived from
direct simulation on four Olympus benchmark datasets [3, 5]. We ran GP-BO (Matérn-5/2 kernel,
Expected Improvement acquisition) and random search for 20 independent seeds each, budget=100,
n_init=5, on the raw experimental lookup tables downloaded from the Olympus GitHub repository [2].

**Results at budget = 50 evaluations (mean over 20 seeds):**

| Dataset | Difficulty | Dim | RS norm_best | BO norm_best | BO advantage (Δ) |
|---|---|---|---|---|---|
| HPLC | easy | 6 | 0.783 | 0.889 | **+0.106** |
| SnAr | easy | 4 | 1.000 | 1.000 | +0.000 |
| Suzuki | hard | 4 | 0.942 | 1.000 | **+0.058** |
| Fullerenes | hard | 3 | 0.992 | 1.000 | +0.008 |

**Key findings:**

1. **The BO advantage is strongly landscape-dependent.** On HPLC (6D, rugged, large dataset),
   BO outperforms random search by +0.11 at n=50. On SnAr (4D, smooth, only 66 points),
   both strategies saturate near the optimum within 20 evaluations — the landscape is too
   easy for BO to show an advantage.

2. **The "hard" classification in the Olympus paper refers to random search difficulty, not
   BO difficulty.** Suzuki is hard for random search (RS=0.94 at n=50) but trivial for BO
   (BO=1.00 at n=10), giving a large apparent gap (+0.24 at n=10). Fullerenes is easy for
   both (both near 1.0 throughout).

3. **The HPLC dataset is the most informative for UC1.** It is the primary pharmaceutical
   anchor dataset, has the highest dimensionality (6D), and shows a sustained, non-trivial
   BO advantage across the full budget range (+0.08 to +0.11).

4. **Across all datasets and budgets, the BO advantage ranges from 0.00 to +0.24.**
   The median advantage at n=50 across the four datasets is approximately **+0.04**.
   For the HPLC dataset specifically, the advantage at n=50 is **+0.11**.

**Revised ground truth for the simulator:**

The simulator will encode a strategy effect of **+0.10** (normalised best outcome units),
representing the BO advantage on a moderately difficult, 4–6 dimensional landscape at a
budget of ~50 evaluations. This is grounded in the HPLC result (+0.11) and is conservative
relative to the Suzuki result (+0.06–+0.24 depending on budget). The value is intentionally
set slightly below the HPLC empirical estimate to account for the fact that our simulator
uses a GP lookup-table emulator rather than the Olympus BNN emulator, which introduces
additional noise.

The simulator will also encode **landscape-dependent heterogeneity**: the strategy effect
will be modulated by `landscape_smoothness` (smooth landscapes → smaller BO advantage,
consistent with the SnAr result) and `dimensionality` (higher dimensions → larger BO
advantage, consistent with HPLC > SnAr/Suzuki/Fullerenes).

**Benchmark figure**: `uc1_bo_vs_random_benchmark.png` (saved to results/).

---

The simulator will encode the following known causal effects, which DoWhy must recover:

| Effect | Value | Empirical basis |
|---|---|---|
| `strategy → norm_best_outcome` | **+0.10** | HPLC: +0.11 at n=50; Suzuki: +0.06 at n=50; median across datasets: +0.04 |
| `landscape_smoothness → norm_best_outcome` | +0.20 | Smoother landscapes are easier to optimise (SnAr near-optimal throughout) |
| `budget → norm_best_outcome` | +0.10 per unit | More experiments always helps (monotone improvement in all traces) |
| `noise_level → norm_best_outcome` | -0.15 | Higher noise degrades surrogate model quality |
| `dimensionality → norm_best_outcome` | -0.08 per unit | Curse of dimensionality (HPLC 6D hardest) |
| `landscape_smoothness → P(strategy=BO)` | +0.25 | Practitioners prefer BO on smooth landscapes |
| `budget → P(strategy=BO)` | +0.15 | BO is worth the overhead for larger budgets |
| `noise_level → P(strategy=BO)` | -0.20 | Practitioners avoid BO in high-noise settings |
| `dimensionality → P(strategy=BO)` | -0.10 | Standard BO degrades in high dimensions |

The naive (unadjusted) estimate of the strategy effect will be inflated above +0.10 because
BO is preferentially used on easier problems (smoother, lower-noise, lower-dimensional).
DoWhy's backdoor adjustment should recover +0.10.

---

## 8. What the design rule will look like

After completing this use case, the design rule extracted will take the form:

> "The naive performance advantage of BO over random search across our campaign library is approximately +X (normalised best outcome). After adjusting for landscape smoothness, budget, noise level, and dimensionality, the true causal effect is +0.10. The gap between naive and adjusted estimates reflects practitioner selection bias — BO is preferentially applied to easier problems. **Design rule**: when benchmarking strategies across campaigns, always condition on landscape properties and budget before attributing performance differences to strategy choice. Failure to do so will overestimate the benefit of BO by approximately X − 0.10 units."
>
> A secondary design rule concerns when BO is worth deploying at all: empirically, the BO advantage is near-zero on low-dimensional (≤3D), smooth, or data-sparse landscapes (SnAr: +0.00; Fullerenes: +0.01), and largest on higher-dimensional (6D), rugged landscapes with dense lookup tables (HPLC: +0.11). The adaptive planner should therefore gate BO deployment on estimated landscape dimensionality and smoothness, not apply it uniformly.

---

## 9. Connections to other use cases

- **UC4** (confounders in simulation): This use case assumes the confounders are correctly identified. UC4 will formally verify this using DoWhy's identification machinery.
- **UC7** (robustness): The design rule derived here has been validated by four refutation tests (§6). UC7 will extend this with systematic sensitivity analysis across the full parameter space.
- **UC9** (CATE): The average treatment effect estimated here will be decomposed into conditional effects in UC9 — i.e., for which values of landscape smoothness, budget, and noise does BO help most?

---

## 10. Medicine formulation specifics

When translating to the curcumin/solubility setting:

- `landscape_smoothness` → estimated from the GP surrogate trained on the 256-point dataset (length-scale parameter)
- `noise_level` → estimated from replicate standard deviations in the Ros et al. dataset (~high, as noted in the paper)
- `budget` → number of formulations tested (256 total in the paper; 32 per BO loop)
- `dimensionality` → 5 (five excipients)
- `norm_best_outcome` → `log(best_concentration) / log(true_maximum_concentration)`

The true maximum concentration is unknown in the real experiment. The paper uses the GP surrogate prediction to estimate it (top 0.1% of predicted formulations). This introduces uncertainty into the outcome variable that must be propagated into the causal estimate.
