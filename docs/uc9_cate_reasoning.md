# UC9: Heterogeneous Treatment Effects — Does BO Advantage Vary Across Campaigns?

**Status**: Complete — CausalForestDML fit, subgroup ATEs, variable importance, OLS sensitivity, partial dependence  
**Estimator**: CausalForestDML (econml 0.16)  
**Unit of analysis**: Campaign (one row = one complete experimental campaign, one strategy)

---

## 1. The Scientific Question

> Does the causal advantage of Bayesian optimisation (RBF-surrogate BO) over random search
> vary across campaigns, and if so, which campaign characteristics — SNR, budget, search
> space dimensionality — moderate it?

UC1 established a global average treatment effect (ATE) of BO over random search. But an
average can mask important heterogeneity: BO might be strongly beneficial in high-SNR,
large-budget campaigns and near-neutral in noisy, budget-constrained ones. If such
heterogeneity exists, the adaptive planner should condition its strategy recommendation on
campaign context rather than applying a single global rule.

UC9 answers this with a Conditional Average Treatment Effect (CATE) analysis: instead of
one number, we estimate a per-campaign treatment effect and ask what drives its variation.

---

## 2. Variable Classification

| Variable | Description | Role |
|---|---|---|
| `T` | Strategy indicator (1 = bo_rbf, 0 = random_search) | **Treatment** |
| `best_found` | Best objective value found in the campaign | **Outcome** |
| `budget` | Total number of evaluations allowed | **Effect moderator / covariate** |
| `snr` | Signal-to-noise ratio (1 / noise_std) | **Effect moderator / covariate** |
| `n_dims` | Search space dimensionality | **Effect moderator / covariate** (constant in this sim) |

**Treatment assignment**: `_resolve_treatment()` assigns the better-performing strategy as
treatment (bo_rbf in this simulation), ensuring the CATE is always the BO-over-RS effect.

---

## 3. Identification Strategy

### 3.1 Causal model

In simulation, strategy assignment is randomised per campaign (propensity = 0.5 by design).
The three key identification assumptions are:

1. **Unconfoundedness**: conditional on (budget, SNR, n_dims), strategy assignment is
   independent of potential outcomes. Holds by design in simulation.
2. **Overlap**: both strategies are observed across the full covariate space. Verified:
   all four SNR × budget subgroups are populated (n = 192–308 each).
3. **SUTVA**: no interference between campaigns. Holds by design.

### 3.2 Estimator: CausalForestDML

CausalForestDML (Double Machine Learning + Causal Forest) was chosen over a simple
interaction-OLS model because:

- It is non-parametric in the CATE surface — no linearity assumption on how moderators
  interact with treatment.
- It provides honest, asymptotically valid pointwise confidence intervals via
  bootstrap-of-little-bags (BLB).
- It produces a variable importance score that directly quantifies each moderator's
  contribution to CATE heterogeneity.

**Configuration:**

| Parameter | Value |
|---|---|
| Nuisance model (outcome) | GradientBoostingRegressor |
| Nuisance model (propensity) | LogisticRegression (max_iter=500) |
| Honest splitting | True |
| Inference | Bootstrap-of-little-bags (BLB) |
| N_estimators | 500 |
| random_state | 42 |

**Note on propensity model:** LogisticRegression is used rather than a non-parametric
classifier. In this simulation the propensity is exactly 0.5 by design, so the choice of
propensity model is inconsequential. In real data with observational strategy assignment,
a more flexible model (e.g., GradientBoostingClassifier) would be appropriate.

---

## 4. Simulation Design

| Parameter | Value |
|---|---|
| Campaigns per strategy | 500 |
| Total rows | 1000 (500 × 2 strategies) |
| Budget values | {30, 35, 40, 45, 50} (discrete, drawn uniformly at random) |
| noise_std range | [0.05, 0.60] (uniform random → SNR range ≈ [1.7, 20]) |
| Warmup | 10 steps (random) |
| Landscape | 2D RBF interpolant (GP sample, Matérn 5/2, ℓ=0.3) |
| BO surrogate | RBF interpolant + distance-based UCB |
| n_dims | 2 (constant) |

**Note on n_dims**: Dimensionality is constant in this simulation. The variable importance
score for n_dims is therefore exactly 0 — this is a design constraint, not an empirical
finding. To estimate the dimensionality moderator, a follow-up simulation with varied n_dims
is required.

**Note on budget**: Budget takes five discrete values. The break-even sweep interpolates
continuously between 30 and 50 for visualisation purposes, but the actual simulation only
samples these five levels.

---

## 5. Results

### 5.1 Global CATE

| Quantity | Value |
|---|---|
| Global CATE | **+0.2293** |
| 95% CI (BLB) | [0.1730, 0.2856] |
| Fraction positive | **100%** |
| CATE std | 0.0281 |

The BO advantage is uniformly positive across all 500 campaigns. The CATE distribution is
narrow (std = 0.028), indicating modest heterogeneity — the effect is consistent rather
than highly variable. No formal omnibus test of heterogeneity was run; the non-zero CATE
std and the forest's variable importance scores are used as descriptive evidence of
moderation.

### 5.2 Subgroup ATEs (2 × 2: SNR × Budget)

Split at median SNR = 3.01, median budget = 40.

| Subgroup | n | ATE | 95% CI |
|---|---|---|---|
| High SNR, Long budget | 296 | **+0.244** | [0.198, 0.290] |
| High SNR, Short budget | 204 | **+0.202** | [0.159, 0.244] |
| Low SNR, Long budget | 308 | **+0.238** | [0.173, 0.303] |
| Low SNR, Short budget | 192 | **+0.223** | [0.150, 0.296] |

All four subgroups show a positive BO advantage with 95% CIs that exclude zero. The spread
across subgroups is small (+0.20 to +0.24), consistent with the low CATE std. There is no
subgroup where random search is competitive. Note: subgroup ATEs are means of per-campaign
CATEs with averaged CI bounds — they are descriptive summaries, not the output of a formal
subgroup hypothesis test.

### 5.3 Variable Importance

| Moderator | Importance |
|---|---|
| **SNR** | **0.744** |
| Budget | 0.256 |
| n_dims | 0.000 (constant) |

SNR is the dominant moderator of CATE heterogeneity, accounting for ~74% of the variation
in individual treatment effects. Budget contributes ~26%. The n_dims importance is exactly
zero because dimensionality is constant in this simulation.

### 5.4 Break-even Thresholds

| Moderator | Break-even | Observed range |
|---|---|---|
| SNR | **None** | [1.67, 19.73] |
| Budget | **None** | {30, 35, 40, 45, 50} |

No break-even threshold was found in either dimension — the CATE remains positive
throughout the observed covariate range (min CATE ≈ +0.187 over SNR sweep, +0.206 over
budget sweep). This is a null result: there is no regime in this simulation where random
search is preferable to BO.

### 5.5 OLS Sensitivity Check

An OLS interaction model (`best_found ~ T * budget_z + T * snr_z`) was fit as a
parametric sensitivity check.

| Quantity | Value |
|---|---|
| OLS ATE (T coef) | +0.2090 [0.177, 0.241] |
| T × budget_z | +0.030 (p = 0.069) |
| T × snr_z | −0.019 (p = 0.261) |
| CausalForest CATE | +0.2293 |
| \|diff\| | 0.020 |

The OLS ATE (+0.209) is consistent with the CausalForest CATE (+0.229), with a difference
of 0.020. The T × SNR interaction is clearly non-significant (p = 0.261). The T × budget
interaction is borderline (p = 0.069): not significant at α = 0.05 but would be at α = 0.10,
and is consistent with the forest's budget importance score (0.256). These two terms should
not be treated equivalently — the budget interaction warrants monitoring in larger or more
varied simulations, while the SNR interaction does not show a linear component.

### 5.6 Partial Dependence

| Moderator | CATE range (partial dependence) |
|---|---|
| SNR (budget held at median = 40) | [+0.187, +0.260] |
| Budget (SNR held at median = 3.01) | [+0.206, +0.268] |

Both partial dependence curves are positive throughout their observed ranges. The SNR curve
shows a wider spread (+0.073) than the budget curve (+0.062), consistent with SNR being the
stronger moderator. The shape of the curves (monotone vs. non-monotone) is visible in the
figure (Panel C) but was not formally tested.

---

## 6. Design Rules

### DR-H-subgroup: BO advantage is robust and consistent across all campaign contexts

**Statement:** BO (RBF-surrogate) causally outperforms random search by +0.229 on average
(95% CI [0.173, 0.286]). The effect is positive in 100% of campaigns and consistent across
all SNR × budget subgroups (+0.20 to +0.24). No break-even threshold exists in the observed
SNR [1.7, 20] and budget {30–50} ranges.

**Implication for the adaptive planner:** In this landscape regime, the planner does not
need to condition BO deployment on SNR or budget — BO is the dominant strategy throughout.
The planner should default to BO and only switch away if there is strong evidence of a
regime outside the observed covariate range (e.g., very low budget < 30 or very high noise
std > 0.6).

**Qualification:** This applies to RBF-surrogate BO on 2D landscapes. Higher-dimensional
landscapes or different surrogate models may show stronger heterogeneity and a genuine
break-even threshold.

### DR-H-varimp: SNR is the dominant moderator of BO advantage; budget shows a marginal linear signal

**Statement:** SNR accounts for ~74% of CATE heterogeneity (variable importance = 0.744);
budget accounts for ~26% (importance = 0.256). The OLS T × SNR interaction is
non-significant (p = 0.261), indicating no strong linear SNR moderation. The OLS T × budget
interaction is borderline (p = 0.069), suggesting a weak positive linear budget effect that
is consistent with the forest importance score but not conclusive at α = 0.05.

**Implication for the adaptive planner:** SNR (or its proxy, estimated noise level) is the
primary campaign feature to monitor for strategy adaptation. Budget is a secondary signal
with a marginal linear component; a larger simulation with wider budget range would be
needed to confirm it. Dimensionality effects cannot be estimated from this simulation and
require a follow-up study with varied n_dims.

---

## 7. Limitations

1. **RBF-surrogate BO, not GP-UCB:** The surrogate is an RBF interpolant with a
   distance-based exploration bonus, not a Gaussian Process with calibrated predictive
   uncertainty. GP-UCB BO likely shows stronger heterogeneity and may have a genuine
   break-even threshold at low SNR.

2. **Constant n_dims = 2:** Dimensionality effects cannot be estimated. The n_dims
   importance score of 0.000 is a design constraint, not evidence that dimensionality
   does not moderate the BO advantage.

3. **Narrow budget range {30, 35, 40, 45, 50}:** A break-even threshold may exist at very
   small budgets (< 30 evaluations) where BO's warmup overhead dominates. This range was
   not tested.

4. **Propensity model is redundant in simulation:** In simulation, propensity is 0.5 by
   design. The LogisticRegression propensity model is harmless but adds no value. In real
   observational data, propensity estimation would be essential and a more flexible model
   should be used.

5. **Synthetic landscape:** The RBF landscape is a fixed mixture. Real experimental
   landscapes (e.g., solubility surfaces) may have different smoothness, multimodality,
   and noise structure, potentially showing stronger CATE heterogeneity.

6. **CATE heterogeneity is modest and untested formally:** The CATE std is 0.028 and the
   difference between the best and worst subgroup ATEs is only 0.042. No formal omnibus
   test of heterogeneity (e.g., best linear predictor test) was run. The heterogeneity
   detected by the forest is descriptively small and unlikely to change a deployment
   decision.

---

## 8. Reproducibility

Key numbers were verified by interactive cell-by-cell execution in `execution_trace.ipynb`
(SEED=42, N_CAMPAIGNS=500). The standalone `uc9_cate.ipynb` contains the same code but
was not re-executed as a fresh independent run; its cells have no saved outputs. Numbers
below are taken directly from printed cell outputs in `execution_trace.ipynb`.

| Quantity | Value from cell output |
|---|---|
| Global CATE | +0.2293 [0.1730, 0.2856] |
| Fraction positive | 100% |
| CATE std | 0.0281 |
| Top moderator | SNR (importance = 0.744) |
| Median SNR (subgroup split) | 3.01 |
| Median budget (subgroup split) | 40 |
| High SNR, Long budget ATE | +0.244 [0.198, 0.290] |
| Low SNR, Short budget ATE | +0.223 [0.150, 0.296] |
| OLS ATE | +0.2090 [0.177, 0.241] |
| OLS T×SNR p | 0.261 |
| OLS T×budget p | 0.069 |
| SNR break-even | None (min CATE = +0.187) |
| Budget break-even | None (min CATE = +0.206) |

---

## 9. Connections to Other Use Cases

- **UC1** (backdoor ATE): UC1 estimates the global average treatment effect of strategy
  choice on campaign outcome using backdoor adjustment. UC9 complements this by estimating
  per-campaign CATEs and identifying which campaign features moderate the effect. The two
  use cases use different simulation runs (different N, landscape seeds, and budget ranges),
  so their ATE estimates are not directly comparable numerically.
- **UC2** (SNR moderation): UC2 tests whether the ATE is moderated by SNR via a linear
  interaction model. UC9 confirms SNR is the dominant moderator (importance = 0.744) using
  a non-parametric estimator, providing convergent evidence.
- **UC8** (mediation): UC8 asks *why* BO outperforms RS (via exploitation quality). UC9
  asks *when* the advantage is largest. Together they characterise both the mechanism and
  the boundary conditions of the BO advantage.
- **Adaptive planner**: DR-H-subgroup and DR-H-varimp feed directly into the planner's
  strategy-selection logic. The null break-even result simplifies the planner: no
  SNR/budget threshold is needed for this landscape regime.

---

## 10. Files

| File | Description |
|---|---|
| `uc9_cate.png` | 4-panel figure: CATE distribution, subgroup ATEs, partial dependence vs SNR, variable importance |
| `uc9_cate_reasoning.md` | This document |
| `uc9_cate.ipynb` | Self-contained notebook (16 cells; code verified via execution_trace.ipynb) |
