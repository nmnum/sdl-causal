# Use Case 1: Causal Effect of Strategy Choice on Campaign Outcome

**Status**: Reasoning complete, awaiting implementation  
**DoWhy feature**: Backdoor adjustment (linear regression, propensity score matching)  
**Unit of analysis**: Campaign (one row = one complete experimental campaign)

---

## 1. The scientific question

> Does the choice of optimisation strategy (Bayesian optimisation vs. random search) causally improve the best outcome found in a campaign, after adjusting for differences in landscape properties, experimental noise, and budget?

The word *causally* is load bearing. We are not asking whether BO correlates with better outcomes - it does, trivially, because practitioners tend to apply BO to problems where they expect it to work. We are asking: if we took two campaigns that were identical in every other respect and changed only the strategy, how much would the outcome differ?

This distinction matters for the adaptive planner. A design rule derived from naive correlation ("BO campaigns achieve higher solubility") is not actionable - it may simply reflect that BO was used on easier problems. A design rule derived from a causal estimate ("BO causally improves normalised best outcome by X, after adjusting for landscape smoothness and budget") is actionable.

---

## 2. Variable classification

| Variable | Description | Role | Justification |
|---|---|---|---|
| `strategy` | BO vs. random search | **Treatment** | The intervention we are studying |
| `norm_best_outcome` | Best outcome found / true optimum | **Outcome** | Campaign level, scale free, comparable across landscapes |
| `landscape_smoothness` | Estimated smoothness of the objective surface (e.g. length scale from GP fit) | **Confounder** | Practitioners choose BO on smoother landscapes; smoother landscapes also yield higher outcomes |
| `budget` | Number of experiments allowed | **Confounder** | Small budgets favour random search (BO overhead not worth it); larger budgets also improve outcomes |
| `noise_level` | Measurement noise standard deviation | **Confounder** | High noise discourages BO (surrogate model degrades); high noise also reduces outcome quality |
| `dimensionality` | Number of input variables | **Confounder** | High dimensionality discourages standard BO; also makes optimisation harder |
| `initial_design_quality` | Space-filling score of seed points | **Non-confounder** | Affects outcome but not strategy choice in our simulation (assigned independently) |
| `batch_id` | Experimental batch identifier | **Noise variable** | Affects outcome via reagent variability but does not affect strategy choice |

### Why `initial_design_quality` is not a confounder

In a real SDL, a practitioner might choose BO precisely because they have a good initial design. In our simulation, initial design quality is assigned independently of strategy - it is a controlled variable, not a confounder. This is a deliberate simulation design choice that we must document and revisit when translating to real data.

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

This is the normalised best outcome - the fraction of the true optimum recovered by the end of the campaign. Equivalently, `simple_regret = 1 - norm_best_outcome`.

**Why not raw outcome value**: Raw values are not comparable across landscapes with different optima. A campaign on a landscape with optimum 10 mg/mL and a campaign on a landscape with optimum 100 mg/mL cannot be compared on the raw scale.

**Why not log(concentration)**: Log concentration is the correct outcome for real medicine formulation data (where the true optimum is unknown). In simulation, we know the true optimum, so normalised best outcome is strictly more informative.

**Why not binary (found optimum yes/no)**: Too sparse and loses information about how close the campaign got.

**Translation to real data**: When applying this to real medicine formulation campaigns, substitute `norm_best_outcome` with `log(best_dissolved_concentration_found)`. The causal model structure is unchanged; only the outcome scale changes.

---

## 6. Ground truth specification for the simulator

The simulator will encode the following known causal effects, which DoWhy must recover:

| Effect | Value | Justification |
|---|---|---|
| `strategy → norm_best_outcome` | **+0.15** | BO recovers 15 percentage points more of the optimum than random search, all else equal |
| `landscape_smoothness → norm_best_outcome` | +0.20 | Smoother landscapes are easier to optimise |
| `budget → norm_best_outcome` | +0.10 per unit | More experiments always helps |
| `noise_level → norm_best_outcome` | -0.15 | Higher noise degrades surrogate model quality |
| `dimensionality → norm_best_outcome` | -0.08 per unit | Curse of dimensionality |
| `landscape_smoothness → P(strategy=BO)` | +0.25 | Practitioners prefer BO on smooth landscapes |
| `budget → P(strategy=BO)` | +0.15 | BO is worth the overhead for larger budgets |
| `noise_level → P(strategy=BO)` | -0.20 | Practitioners avoid BO in high-noise settings |
| `dimensionality → P(strategy=BO)` | -0.10 | Standard BO degrades in high dimensions |

The naive (unadjusted) estimate of the strategy effect will be inflated above +0.15 because BO is preferentially used on easier problems (smoother, lower noise, lower dimensional). DoWhy's backdoor adjustment should recover +0.15.

---

## 7. What the design rule will look like

After completing this use case, the design rule extracted will take the form:

> "The naive performance advantage of BO over random search across our campaign library is approximately +X (normalised best outcome). After adjusting for landscape smoothness, budget, noise level, and dimensionality, the true causal effect is +0.15. The gap between naive and adjusted estimates reflects practitioner selection bias - BO is preferentially applied to easier problems. **Design rule**: when benchmarking strategies across campaigns, always condition on landscape properties and budget before attributing performance differences to strategy choice. Failure to do so will overestimate the benefit of BO by approximately X - 0.15 units."

---

## 8. Connections to other use cases

- **UC4** (confounders in simulation): This use case assumes the confounders are correctly identified. UC4 will formally verify this using DoWhy's identification machinery.
- **UC7** (robustness): The design rule derived here will be subjected to refutation tests in UC7 to check sensitivity to hidden confounders.
- **UC9** (CATE): The average treatment effect estimated here will be decomposed into conditional effects in UC9 - i.e., for which values of landscape smoothness, budget, and noise does BO help most?

---

## 9. Medicine formulation specifics

When translating to the curcumin/solubility setting:

- `landscape_smoothness` → estimated from the GP surrogate trained on the 256 point dataset (length-scale parameter)
- `noise_level` → estimated from replicate standard deviations in the Ros et al. dataset (~high, as noted in the paper)
- `budget` → number of formulations tested (256 total in the paper; 32 per BO loop)
- `dimensionality` → 5 (five excipients)
- `norm_best_outcome` → `log(best_concentration) / log(true_maximum_concentration)`

The true maximum concentration is unknown in the real experiment. The paper uses the GP surrogate prediction to estimate it (top 0.1% of predicted formulations). This introduces uncertainty into the outcome variable that must be propagated into the causal estimate.
