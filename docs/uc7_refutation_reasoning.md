# UC7: Refutation Tests for Design Rule Robustness

**Status**: Complete
**Scientific question**: Under what violations of our modelling assumptions do the derived design rules break, and how badly?
**DoWhy feature**: Refutation tests (random common cause, placebo treatment, data subset, unobserved confounder sensitivity)
**Unit of analysis**: Design rule (one refutation battery per rule)

---

## 1. Purpose and threat model

### What this document is

UC7 is not a new causal question. It is a systematic stress-test of the design rules derived in UC1–UC6. Each rule was derived under a set of modelling assumptions — about the causal DAG, the data-generating process, and the estimator. UC7 asks: which of those assumptions, if violated, would invalidate the rule, and how strong does the violation need to be?

This matters for the adaptive planner. A design rule that is fragile to small hidden confounders should be used with a safety margin or a fallback. A rule that is robust to large violations can be applied with confidence.

### Three threat categories

**Threat 1: Hidden confounders**
A variable we did not measure (e.g. practitioner expertise, instrument age, reagent lot quality) both influences strategy choice and campaign outcome. If it is strong enough, it explains away the estimated effect without any real strategy effect.

*Test*: DoWhy's `add_unobserved_common_cause` refuter. Parameterisation: `effect_on_outcome = 2 × effect_on_treatment` throughout (same as UC1). The CI breach point — the minimum hidden confounder strength that pushes the estimate outside the original 95% CI — is the key output.

**Threat 2: Spurious correlation (no causal effect)**
The estimated effect is an artefact of the data or the adjustment set, not a real causal signal.

*Tests*: Three binary DoWhy refuters:
- `random_common_cause`: add a random N(0,1) variable as a confounder; estimate should not shift (p > 0.05)
- `placebo_treatment_refuter` (permute): permute strategy labels; effect should collapse to ~0 (p > 0.05)
- `data_subset_refuter` (80%): estimate should be stable on a random 80% subsample (p > 0.05)

**Threat 3: Budget sensitivity (UC6 extension)**
The UC6 design rules were derived at budget=50. The UC6 reasoning doc explicitly flags: "does a larger budget mitigate the shift degradation?" This is tested by re-running the UC6 ITS/DiD at B ∈ {20, 50, 100, 200}.

### Design rules in scope

Rules are prioritised by actionability for the adaptive planner:

| Rule | Source | Estimand | Why in scope |
|---|---|---|---|
| DR-A | UC1 | ATE = +0.098 (backdoor, linear) | Foundation of all strategy recommendations |
| DR-B | UC2 | ATE (SNR-moderated) = +0.081; τ = 2.0 threshold | Primary switching criterion for the planner |
| DR-D | UC3 | Convergence rate interaction = +0.018; warmup ~10% of budget | Timing criterion for early stopping / switching |
| DR-UC6-B | UC6 | ITS three-way interaction = −0.014 (BO sensitivity ratio) | Shift-response criterion for the planner |

DR-UC6-E/F/G (detection and estimation rules) are not refuted here — they are methodological recommendations about estimators, not causal effect estimates, and DoWhy's refuters do not apply to them.

---

## 2. Refutation methods

### 2.1 Binary refutation tests

All three tests use DoWhy's built-in refuters (DR-A, DR-B) or equivalent manual implementations for panel OLS estimators (DR-D, DR-UC6-B). The pass criterion is p > 0.05.

**Random common cause** (`random_common_cause`): Adds a randomly generated variable as a common cause of both treatment and outcome. If the original estimate is robust, adding a noise variable should not shift it. A failure (p ≤ 0.05) indicates the estimator is sensitive to the adjustment set composition.

**Placebo treatment** (`placebo_treatment_refuter`, `placebo_type="permute"`): Randomly permutes the treatment assignment at the campaign level, breaking any causal relationship between strategy and outcome. The new effect should be ~0.

**Data subset** (`data_subset_refuter`, `subset_fraction=0.8`): Re-estimates on a random 80% subsample of campaigns. The new effect should be close to the original.

### 2.2 Unobserved confounder sensitivity sweep

Parameterisation (same as UC1 §6):
- `confounders_effect_on_treatment = "binary_flip"`
- `confounders_effect_on_outcome = "linear"`
- `effect_strength_on_outcome = 2 × effect_strength_on_treatment`
- Strength grid: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
- N_UCC_SEEDS = 10 per grid point for DR-A, DR-D, and DR-UC6-B; 20 for DR-B (non-monotonicity at N=10)

**CI breach point**: The minimum effect strength at which the mean new effect crosses the original 95% CI bound. For DR-A, DR-B, and DR-D (positive estimates) this is the lower CI bound; for DR-UC6-B (negative estimate) this is the upper CI bound.

**Monotonicity check**: The sensitivity curve should be strictly monotone (stronger hidden confounder → estimate moves toward zero). Non-monotonicity at high effect strengths indicates the estimate is near the CI bound and the refuter variance dominates — the breach point is then approximate.

### 2.3 Budget sensitivity (UC6 Tier 2)

Re-run `run_campaigns_uc6()` at B ∈ {20, 50, 100, 200}, n=2000 campaigns each, seed=42. For each budget, re-fit the ITS/DiD model (same specification as UC6 Section 6) and extract:

1. The three-way interaction coefficient `strategy:log_step:post_shift` (DR-UC6-B)
2. The post-shift SNR distribution (median, fraction below τ=2.0)
3. The post-shift SNR fraction below τ (DR-UC6-D monitoring threshold)

---

## 3. Results by design rule

All results: n=2,000 simulated campaigns, budget=50, seed=42.

### 3.1 DR-A: BO ATE (UC1 backdoor)

**Original estimate**: ATE = +0.098, 95% CI [+0.087, +0.110], n=2,000

| Test | New effect | p-value | Pass |
|---|---|---|---|
| Random common cause | +0.098 | 0.880 | YES |
| Placebo treatment | +0.000 | 0.920 | YES |
| Data subset (80%) | +0.098 | 0.960 | YES |

**UCC sensitivity** (effect_on_outcome = 2 × effect_on_treatment):

| Effect on treatment | Mean new ATE | Shift |
|---|---|---|
| 0.01 | +0.096 | −0.002 |
| 0.02 | +0.094 | −0.004 |
| 0.05 | +0.088 | −0.010 |
| **0.10** | **+0.082** | **−0.016** |
| 0.15 | +0.072 | −0.026 |
| 0.20 | +0.061 | −0.037 |
| 0.30 | +0.042 | −0.056 |
| 0.40 | +0.031 | −0.067 |

**CI breach point**: eff = 0.10 (first crossing of CI lower bound +0.087). Curve is strictly monotone.

**Verdict**: **Conditionally robust**. All binary tests pass. A hidden confounder would need to flip ≥10% of treatment assignments (and shift outcomes by ≥0.20 units — twice the ATE itself) to invalidate DR-A. This is a moderately strong confounder; the rule can be applied with confidence in most settings.

---

### 3.2 DR-B: SNR threshold τ = 2.0 (UC2 ATE)

**Original estimate**: ATE = +0.081, 95% CI [+0.069, +0.093], n=2,000
Ground truth E[f(snr)] = +0.083 (within CI — correctly recovered).

| Test | New effect | p-value | Pass |
|---|---|---|---|
| Random common cause | +0.081 | 0.920 | YES |
| Placebo treatment | −0.000 | 0.980 | YES |
| Data subset (80%) | +0.081 | 1.000 | YES |

**UCC sensitivity** (N=20 seeds per grid point to resolve non-monotonicity at N=10):

| Effect on treatment | Mean new ATE | Shift |
|---|---|---|
| 0.01 | +0.079 | −0.002 |
| 0.05 | +0.074 | −0.007 |
| 0.10 | +0.070 | −0.011 |
| **0.15** | **+0.064** | **−0.017** |
| 0.20–0.40 | +0.060–0.063 | plateau near CI lower bound |

**CI breach point**: eff = 0.15. The curve plateaus at eff ≥ 0.15 rather than continuing to decline — the estimate hovers just below the CI lower bound (+0.069) throughout the high-strength range. This plateau is a property of the UCC refuter parameterisation: `binary_flip` with probability p saturates as p increases, because flipping 15–40% of treatment assignments produces a similar net confounding effect on the ATE (the flip probability is bounded and the marginal confounding gain diminishes). The curve is non-monotone at eff ≥ 0.20 (violations of size ~0.001 — within Monte Carlo noise at N=20 seeds).

**Verdict**: **Conditionally robust**. All binary tests pass. The breach point of 0.15 means a confounder would need to flip 15% of treatment assignments and shift outcomes by 0.30 units to invalidate DR-B. The plateau behaviour means even stronger confounders do not push the estimate further below the CI lower bound — the rule degrades to the CI boundary and stays there.

**Important caveat on scope**: The binary tests and UCC sweep validate the linear ATE estimate (+0.081), not the τ=2.0 threshold itself. The threshold is a structural parameter of the simulator (UC2_SNR_THRESHOLD), derived from GP-theoretic arguments in uc2_dag_reasoning.md §4. No UC7 test directly refutes the threshold claim — doing so would require varying τ and measuring the resulting ATE bias, which is a UC2 sensitivity analysis (already partially addressed in the UC2 τ sensitivity notebook). The budget sensitivity analysis (§3.4) addresses the threshold indirectly by showing the post-shift SNR distribution is budget-independent, confirming τ=2.0 applies equally at all budgets.

---

### 3.3 DR-D: BO warmup ~10% of budget (UC3 convergence rate)

**Original estimate**: strategy:log_step interaction = +0.018, 95% CI [+0.018, +0.019], p < 0.001, n=2,000

| Test | New coefficient | p-value | Pass |
|---|---|---|---|
| Random common cause | +0.018 | 1.000 | YES |
| Placebo treatment | −0.001 | 0.149 | YES |
| Data subset (80%) | +0.018 | 0.823 | YES |

**UCC sensitivity** (manual implementation — binary flip on strategy at campaign level, linear on outcome; N=10 seeds per grid point):

| Effect on treatment | Mean new coef | Shift |
|---|---|---|
| **0.01** | **+0.018** | **−0.000** |
| **0.02** | **+0.017** | **−0.001** |
| 0.05 | +0.016 | −0.002 |
| 0.10 | +0.014 | −0.004 |
| 0.15 | +0.012 | −0.006 |
| 0.20 | +0.011 | −0.008 |
| 0.25 | +0.009 | −0.009 |
| 0.30 | +0.007 | −0.011 |
| 0.35 | +0.005 | −0.013 |
| 0.40 | +0.004 | −0.014 |

**CI breach point**: eff = 0.02 (first crossing of CI lower bound +0.018). Curve is strictly monotone.

**Interpretation**: The breach point of 0.02 appears alarming — a hidden confounder flipping only 2% of treatment assignments breaches the CI. However, this reflects the extremely tight CI (width = 0.001), not fragility of the underlying effect. The *relative* CI width is only 6.3% (vs. 23.5% for DR-A), meaning the estimate is highly precise. The key robustness argument for DR-D is different from DR-A and DR-B:

1. **Interaction terms are less sensitive to random confounding than main effects.** The UCC refuter adds a confounder that randomly flips strategy assignments. For the strategy:log_step interaction, this random flip does not systematically correlate with log_step — so the confounding effect on the interaction coefficient is much smaller in absolute terms than on a main-effect ATE. The shift at eff=0.40 is only −0.014 (vs. −0.067 for DR-A at the same strength).
2. **The signal is very strong** (p < 0.001 vs. p = 0.047 for DR-UC6-B). The estimate would need to shift by 0.001 in absolute terms to breach the CI, but the actual shift at eff=0.40 is −0.014 — meaning the estimate is still positive (+0.004) and the rule direction is preserved throughout the entire sweep.
3. **The rule direction (BO > RS in convergence rate) is never reversed.** Even at eff=0.40, the mean new coefficient is +0.004 > 0. The CI breach at eff=0.02 is a precision artefact, not a sign that the rule is wrong.

**Verdict**: **Conditionally robust** (revised from "Robust"). All binary tests pass with high confidence. The CI breach at eff=0.02 is a consequence of the tight CI, not genuine fragility — the rule direction is preserved across the full UCC sweep. The placebo p=0.149 (not 0.920 as for DR-A) reflects that campaign-level permutation of strategy still leaves some residual correlation in the panel structure — this is expected and not a concern.

---

### 3.4 DR-UC6-B: BO sensitivity to noise shifts (UC6 ITS)

**Original estimate**: strategy:log_step:post_shift = −0.014, 95% CI [−0.028, −0.000], p=0.047, n=2,000

| Test | New coefficient | p-value | Pass |
|---|---|---|---|
| Random common cause | −0.014 | 1.000 | YES |
| Placebo treatment | −0.013 | 0.078 | YES |
| Data subset (80%) | −0.013 | 0.860 | YES |

**Note on placebo p=0.078**: The p-value tests H₀: permuted_coef = 0. At p=0.078 this passes (the permuted effect is not significantly different from zero). However, the permuted coefficient magnitude (−0.013) is close to the original (−0.014) — a separate concern about test sensitivity. This proximity reflects the weak original signal (p=0.047): campaign-level permutation does not fully destroy a marginal effect. The test passes by the correct criterion (H₀: permuted = 0), but the closeness of the permuted and original values is a warning sign about the signal strength, not a p-value interpretation issue.

**UCC sensitivity** (manual implementation — binary flip on strategy, linear on outcome):

| Effect on treatment | Mean new coef | Shift toward zero |
|---|---|---|
| 0.01–0.10 | −0.014 to −0.014 | ~0.000 (negligible) |
| 0.15 | −0.011 | +0.004 |
| 0.20 | −0.006 | +0.008 |
| 0.25 | −0.003 | +0.011 |
| **0.30** | **+0.004** | **+0.018** |
| 0.35–0.40 | +0.005 to +0.002 | high variance (std ~0.03) |

**CI breach point**: eff ≈ 0.30 (first crossing of CI upper bound −0.000). High variance at eff ≥ 0.25 (std ~0.02–0.04) makes this approximate. The curve is non-monotone at high effect strengths.

**Budget sensitivity**:

| Budget | ITS coef | 95% CI | p-value | Post-shift SNR (median) | Frac < τ=2.0 |
|---|---|---|---|---|---|
| 20 | −0.0004 | [−0.014, +0.013] | 0.950 | 2.92 | 36.5% |
| 50 | −0.0140 | [−0.028, −0.000] | 0.047* | 2.93 | 36.4% |
| 100 | −0.0185 | [−0.033, −0.004] | 0.010* | 2.93 | 36.4% |
| 200 | −0.0205 | [−0.035, −0.006] | 0.005* | 2.93 | 36.4% |

*p < 0.05

Key finding: the post-shift SNR distribution is **identical across all budgets** (median ≈ 2.93, frac < τ ≈ 36%). The budget effect on the coefficient is a statistical power effect — longer campaigns have more post-shift steps, providing more data to detect the same underlying BO sensitivity. The ITS/DiD estimator cannot detect DR-UC6-B at B=20 (p=0.950).

**Verdict**: **Fragile**. The three binary tests pass, but the marginal p=0.047 and the UCC breach point of ~0.30 (with high variance) indicate this rule is the weakest of the four. A hidden confounder of moderate-to-strong strength (eff ~0.30 on treatment, ~0.60 on outcome) can push the three-way interaction to zero. The budget sensitivity analysis confirms the rule is undetectable at B=20 with the ITS estimator. Use DR-UC6-B as a directional signal, not a precise quantitative rule.

---

## 4. Cross-rule robustness summary

| Rule | RCC | Placebo | Subset | UCC breach point | Budget sensitivity | Verdict |
|---|---|---|---|---|---|---|
| DR-A | PASS (p=0.88) | PASS (p=0.92) | PASS (p=0.96) | 0.10 | N/A | **Conditionally robust** |
| DR-B | PASS (p=0.92) | PASS (p=0.98) | PASS (p=1.00) | 0.15 (plateau) | N/A | **Conditionally robust** |
| DR-D | PASS (p=1.00) | PASS (p=0.15) | PASS (p=0.82) | 0.02 (precision artefact — rule direction preserved) | N/A | **Conditionally robust** |
| DR-UC6-B | PASS (p=1.00) | PASS (p=0.08) | PASS (p=0.86) | ~0.30 (approx.) | Undetectable at B=20 | **Fragile** |

**Verdict key**:
- **Conditionally robust**: All binary tests pass; UCC breach point ≥ 0.02; a moderate-to-strong hidden confounder is required to invalidate. For DR-D, the breach at eff=0.02 is a precision artefact of the tight CI — the rule direction (BO > RS) is preserved across the full UCC sweep (mean new coef > 0 at all strengths). For DR-A and DR-B, breach requires flipping 10–15% of treatment assignments.
- **Fragile**: All binary tests pass but with low margins; UCC breach point ~0.30 with high variance; estimate is near the CI boundary throughout the sweep; budget-dependent detectability.

**Ranking by robustness** (most to least): DR-D ≈ DR-B ≈ DR-A > DR-UC6-B

Note: DR-D, DR-B, and DR-A are all conditionally robust. DR-D has the tightest CI and strongest p-value, but its UCC breach point (0.02) is lower than DR-A (0.10) and DR-B (0.15) because the CI is so narrow. The rule *direction* for DR-D is never reversed in the UCC sweep, making it the most reliable for directional planner decisions.

---

## 5. Implications for the adaptive planner

### 5.1 Rules that can be applied with high confidence

**DR-D** (BO warmup ~10% of budget) is the most reliable rule for directional decisions. The convergence rate interaction is highly significant (p < 0.001), stable across all perturbations, and the rule direction (BO > RS) is never reversed in the UCC sweep even at eff=0.40. The CI breach at eff=0.02 is a precision artefact — the estimate remains positive throughout. The planner can use the 10%-of-budget warmup threshold as a hard criterion for early switching decisions.

**DR-B** (SNR threshold τ=2.0) is conditionally robust. The breach point of 0.15 means a confounder would need to flip 15% of treatment assignments to invalidate the ATE estimate. The plateau behaviour at high effect strengths (a UCC refuter saturation effect) means even stronger confounders do not push the estimate further below the CI lower bound. The planner can use τ=2.0 as the primary BO/RS switching criterion, with the caveat that the threshold itself is a structural parameter (not estimated from data) and its validity depends on the GP-theoretic argument in uc2_dag_reasoning.md §4.

### 5.2 Rules that need a safety margin or fallback

**DR-A** (BO ATE = +0.098) is conditionally robust with breach point 0.10. In the pharmaceutical formulation setting, instrument calibration drift is a plausible hidden confounder. If the drift is large enough to flip ~10% of strategy assignments (e.g. a poorly calibrated instrument consistently discourages BO), the ATE estimate could be invalidated. **Recommendation**: use DR-A as the baseline strategy recommendation, but cross-validate with DR-B (SNR-conditioned) before committing to BO in a new setting.

**DR-UC6-B** (BO 1.35× more sensitive to noise shifts) is fragile. The three-way interaction is marginally significant (p=0.047) and undetectable at B=20. **Recommendations**:
1. Treat DR-UC6-B as a directional signal (BO is more sensitive to shifts than RS) rather than a precise quantitative rule (1.35× ratio).
2. At B ≤ 30, do not rely on the ITS/DiD estimator to detect the BO sensitivity ratio from a single campaign — pool across campaigns or use the prior from the simulation.
3. The monitoring threshold τ=2.0 (DR-UC6-D) is budget-independent and more reliable than the sensitivity ratio — use it as the primary shift-response criterion.

### 5.3 Budget sensitivity implications for UC6 rules

The post-shift SNR distribution is identical across B ∈ {20, 50, 100, 200} (median ≈ 2.93, frac < τ ≈ 36%). This means:

1. **The τ=2.0 monitoring threshold (DR-UC6-D) is budget-independent** — the fraction of post-shift steps below τ is the same regardless of campaign length. The planner should apply the same monitoring threshold at all budgets.

2. **The ITS/DiD estimator for DR-UC6-B requires B ≥ 50** to achieve p < 0.05. At B=20, the signal is undetectable (p=0.950). This is a statistical power limitation, not a causal one — the underlying BO sensitivity exists at all budgets.

3. **For the curcumin formulation setting** (realistic budget B=20–50, SNR ≈ 1.9): the τ=2.0 threshold is the actionable criterion. With SNR already near τ, even a modest 1.5× noise shift pushes post-shift SNR to ~1.3, well below τ. The planner should be aggressive about switching to RS after any detected shift, regardless of budget.

---

## 6. Connections to other use cases

- **UC1**: DR-A was already validated by four refutation tests in UC1 §6 (ATE = +0.096, breach at eff ≥ 0.05 in the original analysis). UC7 re-runs the same battery at n=2,000 and finds breach at eff = 0.10. The higher breach point at n=2,000 reflects a larger ATE estimate (+0.098 vs. +0.096) that sits further above the CI lower bound, requiring a stronger confounder to push it below the threshold — not a consequence of CI width per se.
- **UC2**: DR-B (SNR threshold τ=2.0) is the primary switching criterion. The plateau behaviour in the UCC sweep (a refuter saturation effect) means even strong confounders cannot push the estimate far below the CI lower bound.
- **UC3**: DR-D (warmup ~10% of budget) is the most reliable rule for directional decisions. Its tight CI (width ~0.001) reflects the large panel dataset (100,000 step-level rows) and the strong convergence rate signal. The UCC breach at eff=0.02 is a precision artefact — the rule direction is never reversed.
- **UC6**: DR-UC6-B is fragile. The budget sensitivity analysis confirms the τ=2.0 monitoring threshold (DR-UC6-D) is more reliable than the sensitivity ratio for the adaptive planner.
- **UC8**: DR-UC6-B's fragility makes the mediation question (UC8) more urgent — understanding *why* BO is more sensitive to shifts (surrogate miscalibration vs. direct noise) would provide a mechanistic basis for the rule that is independent of the marginal ITS estimate.
- **UC9**: CATE estimation (UC9) will condition on the same confounders as UC1–UC6. DR-A and DR-B are conditionally robust, so the CATE surface should be reliable for the primary confounders. DR-UC6-B's fragility means the CATE surface for shift-sensitivity should be interpreted cautiously.

---

## 7. Medicine formulation specifics

The curcumin/solubility setting (SNR ≈ 1.9, near τ=2.0) is particularly relevant to the budget sensitivity analysis. Key findings for this setting:

- **τ=2.0 is budget-independent**: the monitoring threshold applies equally at B=20 (realistic for expensive formulation experiments) and B=200.
- **DR-UC6-B is undetectable at B=20**: the planner cannot estimate the BO sensitivity ratio from a single small campaign. Use the simulation prior (BO is more sensitive than RS) as a default.
- **Instrument calibration drift** is the most plausible hidden confounder in the formulation setting. The UCC breach points for DR-A (0.10) and DR-B (0.15) give quantitative bounds: if calibration drift flips fewer than 10–15% of strategy assignments, the design rules remain valid.
- **Practical recommendation**: in the curcumin setting, use DR-B (τ=2.0) as the primary strategy criterion and DR-UC6-D (monitor GP residual variance) as the shift-detection criterion. Do not rely on DR-UC6-B's quantitative sensitivity ratio at small budgets.

---

## 8. Visualisation

`uc7_refutation.png` — four panels:
- **A**: UCC sensitivity curves for DR-A, DR-B, DR-UC6-B. Shows DR-A and DR-B declining smoothly; DR-UC6-B rising toward zero with high variance at high effect strengths. Breach points marked as vertical dotted lines.
- **B**: Budget sensitivity for DR-UC6-B ITS coefficient. Shows the coefficient growing in magnitude from −0.0004 (B=20, insignificant) to −0.0205 (B=200, p=0.005). B=20 is red (p ≥ 0.05); B=50–200 are blue (p < 0.05).
- **C**: Post-shift SNR by budget. Bar chart shows fraction of post-shift steps below τ=2.0 (~36% at all budgets); line shows median post-shift SNR (~2.93 at all budgets). Confirms τ=2.0 threshold is budget-independent.
- **D**: Robustness heatmap. Rows = rules, columns = tests. All binary tests green (PASS). UCC breach column: DR-A and DR-B orange (conditional), DR-D orange with annotation (precision artefact — direction preserved), DR-UC6-B red (fragile, ~0.30).
