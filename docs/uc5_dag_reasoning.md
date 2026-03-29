# UC5 DAG Reasoning: ATT Estimation for Strategy Switching

**Status**: Complete
**Scientific question**: Among campaigns that used BO, what would their outcome have been under random search?
**Estimand**: ATT = E[Y(1) − Y(0) | T = 1]
**DoWhy feature**: Counterfactual estimation (AIPW doubly-robust ATT; interaction OLS for individual counterfactuals)
**Design rule**: DR-H

---

## 1. Estimand definition

### ATE vs ATT

UC1 estimates the **Average Treatment Effect (ATE)**:

$$\text{ATE} = E[Y(1) - Y(0)]$$

This is the average effect of BO across *all* campaigns, including those where BO was never used. It answers: "what would happen if we randomly assigned BO to a new campaign drawn from the full landscape distribution?"

UC5 estimates the **Average Treatment Effect on the Treated (ATT)**:

$$\text{ATT} = E[Y(1) - Y(0) \mid T = 1]$$

This is the average effect among campaigns that *actually used* BO. It answers: "for the campaigns where a practitioner chose BO, was that choice justified?"

### Why ATT for UC5?

The ATT is the natural estimand for retrospective evaluation:

1. **Uses existing data structure**: BO campaigns are the treated group; no new data needed.
2. **Directly actionable**: answers whether past BO deployments were justified.
3. **Comparable to UC1**: ATT and ATE are on the same scale (normalised outcome units), so the gap quantifies positive selection bias.
4. **Avoids extrapolation**: the ATT only requires predicting Y(0) for treated units, not for the full population. This is less demanding than the ATE when the treated and control groups have different covariate distributions.

### What ATT is NOT

The ATT does not answer: "what would happen if we switched strategy mid-campaign?" That is a within-campaign do-calculus query requiring a structural BO surrogate model. It is deferred to UC8/UC9.

---

## 2. Causal structure

### DAG

UC5 uses the **UC2 DAG** unchanged. No new nodes or edges are needed — the ATT is an estimation-layer change, not a structural one.

```
landscape_smoothness ──┬──→ strategy ──→ norm_best_outcome
                       │                      ↑
noise_level ───────────┼──→ snr ──────────────┤
                       │                      │
budget ────────────────┤                      │
                       │                      │
dimensionality ────────┘                      │
                                              │
initial_design_quality ───────────────────────┘
batch_id ─────────────────────────────────────┘
```

The SNR node moderates the strategy effect (UC2 DR-B): BO's advantage is largest at high SNR and collapses below τ = 2.0.

### Identification

The ATT is identified by the same backdoor criterion as the ATE:

**Adjustment set**: {landscape_smoothness, budget, noise_level, dimensionality, snr}

The backdoor criterion holds for ATT as well as ATE: conditioning on the adjustment set blocks all non-causal paths from strategy to outcome, regardless of whether we restrict to T=1 units.

---

## 3. Positive selection hypothesis

Because practitioners preferentially apply BO to easier campaigns, the treated group (BO campaigns) is positively selected:

| Covariate | BO mean | RS mean | Difference | Direction |
|---|---|---|---|---|
| landscape_smoothness | 0.773 | 0.666 | +0.107 | BO easier |
| noise_level | 0.107 | 0.142 | −0.035 | BO easier |
| budget | 61.4 | 57.3 | +4.2 | BO easier |
| dimensionality | 4.15 | 4.25 | −0.10 | BO easier |
| snr | 19.4 | 11.2 | +8.1 | BO easier |

All five confounders are shifted in the direction that makes BO more effective. This is the mechanism behind ATT > ATE: BO campaigns are drawn from the high-SNR tail of the landscape distribution, where BO's advantage is largest.

**True values (from DGP, n=2000, seed=42)**:
- True ATE = +0.083
- True ATT = +0.088
- True gap (ATT − ATE) = +0.005 (+5.7% of ATE)

The gap is real but small — selection is moderate, not extreme.

---

## 4. Estimators

### Why not DoWhy backdoor linear regression for ATT?

DoWhy's `estimate_effect(..., target_units='att')` with `method_name='backdoor.linear_regression'` returns the same value as the ATE. This is a known property of linear regression: for $Y = \beta_0 + \beta_1 T + \beta_2 X + \varepsilon$, the coefficient $\beta_1$ is the same regardless of which subgroup we evaluate. To get ATT ≠ ATE, we need an estimator that allows treatment effect heterogeneity.

### AIPW (doubly-robust) — primary ATT estimator

$$\hat{\text{ATT}}_{\text{AIPW}} =
  \underbrace{E[\hat{\mu}_1(X) - \hat{\mu}_0(X) \mid T=1]}_{\text{outcome model}}
  + \underbrace{\frac{E[T(Y - \hat{\mu}_1(X))]}{P(T=1)}}_{\text{treated residual}}
  - \underbrace{\frac{E\left[\frac{(1-T)\hat{e}(X)}{1-\hat{e}(X)}(Y - \hat{\mu}_0(X))\right]}{P(T=1)}}_{\text{control residual}}$$

where $\hat{e}(X) = P(T=1 \mid X)$ is the propensity score (logistic regression) and $\hat{\mu}_t(X) = E[Y \mid T=t, X]$ are the outcome models (OLS, fit separately on treated and control).

AIPW is consistent if *either* the propensity score *or* the outcome model is correctly specified ("doubly robust"), and has lower variance than plain IPW.

### Interaction OLS — individual counterfactuals

For per-campaign Y(0) predictions, we use OLS with a `strategy × snr_s` interaction:

$$Y = \beta_0 + \beta_1 T + \beta_2 \text{snr}_s + \beta_3 (T \times \text{snr}_s) + \beta_4 X + \varepsilon$$

This captures the SNR-dependent treatment effect heterogeneity. The ATT from this model is $\hat{\beta}_1 + \hat{\beta}_3 \cdot \bar{\text{snr}}_s^{(1)}$, where $\bar{\text{snr}}_s^{(1)}$ is the mean standardised SNR among treated units.

### Estimator comparison

| Estimator | ATE | ATT | ATT > ATE | ATT error |
|---|---|---|---|---|
| True (DGP) | +0.083 | +0.088 | Yes | — |
| DoWhy backdoor LR | +0.075 | same | No (by construction) | — |
| Interaction OLS | +0.068 | +0.073 | Yes | −0.015 |
| Hajek IPW | — | +0.083 | Yes | −0.005 |
| AIPW (doubly-robust) | +0.069 | +0.073 | Yes | −0.015 |

All estimators that allow heterogeneity give ATT > ATE. The underestimation bias (~0.015) is common to all linear estimators because the true SNR effect is sigmoid (nonlinear).

---

## 5. Results

### ATT estimate

- **AIPW ATT**: +0.073 (bootstrap 95% CI: [+0.060, +0.085])
- **True ATT**: +0.088
- **Error**: −0.015 (linear model bias from sigmoid SNR approximation)

### Individual counterfactuals

Among 990 BO-treated campaigns:
- Mean switching loss Y(1) − Ŷ(0) = +0.073
- Median: +0.066
- Std: 0.130
- 68% of campaigns: BO helped (positive switching loss)
- 32% of campaigns: BO hurt or made no difference

The 32% where BO hurt are predominantly low-SNR campaigns where the GP surrogate is unreliable.

### SNR-stratified ATT

| SNR bin | n treated | True ATT | Estimated ATT | Note |
|---|---|---|---|---|
| Low (SNR < 1.5) | 38 | −0.007 | +0.017 | **sign error** — underpowered |
| Mid (1.5 ≤ SNR < 3.0) | 87 | +0.035 | +0.031 | directional |
| High (SNR ≥ 3.0) | 865 | +0.097 | +0.080 | reliable |

Ordering holds in both true and estimated values: low < mid < high.

Key observations:
1. **High-SNR dominates**: 865/990 (87%) of BO campaigns are in the high-SNR bin — positive selection concentrates BO use where it works best.
2. **True ATT is negative at low SNR**: BO slightly hurts when the surrogate is unreliable (true ATT_low = −0.007). The estimated ATT_low = +0.017 has the **wrong sign** due to linear model extrapolation bias and severe underpowering (n=38).
3. **Sign error in estimated ATT_low**: the interaction OLS model extrapolates linearly into the low-SNR regime where the true effect is nonlinear (sigmoid) and negative. Do not use the estimated ATT_low as a point estimate; use the true ATT_low (known by DGP construction) to confirm the direction.
4. **Interaction term marginal significance**: the `strategy:snr_s` interaction coefficient has p=0.057, which is marginally non-significant at α=0.05. The stratified heterogeneity claim is supported by the true DGP but is statistically weak in the estimated values. Treat stratified ATT estimates as directional evidence, not precise estimates.

---

## 6. Estimator limitations

### Linear model bias

Both the propensity score model (logistic regression) and the outcome models (OLS) are linear approximations to the true sigmoid SNR effect. The true structural equation is:

$$\text{strategy\_effect}(snr) = \text{floor} + (\text{ceiling} - \text{floor}) \times \sigma\left(\frac{snr - \tau}{\text{scale}}\right)$$

A linear model cannot capture this sigmoid shape, causing systematic underestimation of the ATT by ~0.015. This bias is:
- **Consistent across estimators**: all linear methods underestimate by a similar amount.
- **Not a threat to the main conclusions**: the ATT > ATE ordering and the SNR stratification ordering are both robust.
- **Reducible**: a nonparametric outcome model (e.g. gradient boosting) would reduce this bias at the cost of interpretability.

### ATT-ATE gap precision

The true ATT-ATE gap is only +0.005 (5.7% of ATE). This is within the estimation noise of all linear estimators. The gap is real (confirmed by the DGP) but not precisely estimable with linear models at n=2000. The main result is the ATT magnitude, not the gap.

### Low-SNR bin instability and sign error

The low-SNR bin has only 38 treated campaigns. The estimated ATT_low (+0.017) has the **wrong sign** relative to the true ATT_low (−0.007). This is a combination of:
- **Small sample size** (n=38): insufficient power to detect the small negative effect.
- **Linear extrapolation**: the interaction OLS model is fit on the full dataset where the high-SNR regime dominates. It extrapolates a positive linear trend into the low-SNR region where the true effect is negative and nonlinear.
- **Interaction term non-significance**: the `strategy:snr_s` interaction (p=0.057) is marginally non-significant, meaning the model has weak evidence for heterogeneity overall.

The true ATT_low (−0.007, known by DGP construction) is the correct reference. The estimated value should not be used as a point estimate for the low-SNR regime.

---

## 7. Refutation tests

| Test | New effect | p-value | Pass |
|---|---|---|---|
| Random common cause | +0.075 | 0.980 | Yes |
| Placebo treatment | +0.000 | 1.000 | Yes |
| Data subset (80%) | +0.075 | 0.940 | Yes |

All three pass.

**What these tests validate**: DoWhy refutation tests probe the *identification assumptions* — specifically unconfoundedness (no unmeasured common causes) and the absence of spurious associations. These assumptions are shared by both the ATE and ATT estimands, so passing these tests supports the validity of the backdoor adjustment for both targets.

**What these tests do NOT validate**: The refutation uses DoWhy's backdoor linear regression with `target_units='att'`. Because a global linear model gives ATT = ATE by construction, the refuted estimate (+0.075) is the ATE, not the AIPW ATT (+0.073) reported in §5. The refutation tests therefore do **not** directly validate the AIPW point estimate or its confidence interval. They validate the underlying causal identification, which is a prerequisite for any ATT estimator.

---

## 8. Design rule DR-H

> **DR-H (Retrospective BO value)**: Among campaigns where BO was used, switching to random search would have reduced the best outcome by **ATT ≈ +0.073** normalised units on average (AIPW, 95% CI: [+0.060, +0.085]; true ATT = +0.088). The ATT exceeds the population ATE (+0.069 AIPW) because practitioners deploy BO preferentially on high-SNR campaigns where its surrogate model is most reliable.
>
> **Stratified rule**: The switching loss is concentrated in high-SNR campaigns (ATT_high ≈ +0.080, n=865). At low SNR (< 1.5), the *true* ATT is negative (true ATT_low ≈ −0.007, n=38 treated), meaning BO slightly hurts when the surrogate is unreliable. **Caveat**: the *estimated* ATT_low is +0.017 (wrong sign), because the interaction OLS model extrapolates linearly into a region where the true effect is non-linear and the bin is severely underpowered (n=38). The stratified heterogeneity claim rests on the true DGP pattern and on the interaction term (p=0.057, marginally non-significant at α=0.05); treat it as directional evidence, not a precise estimate.
>
> **Practical implication**: Retrospective performance comparisons that report "BO outperformed random search by X" are measuring the ATT, not the ATE. The ATT overstates BO's benefit for a *new* campaign drawn from the full landscape distribution. To estimate the benefit for a new campaign, use the ATE and condition on the campaign's SNR using DR-B.

---

## 9. Connection to other use cases

| UC | Estimand | Key result | Connection to UC5 |
|---|---|---|---|
| UC1 | ATE | +0.075 (LR) | UC5 ATT (+0.073) ≈ UC1 ATE; gap is small |
| UC2 | ATE(SNR) | BO collapses below τ=2.0 | Explains why ATT_low ≈ 0 and ATT_high >> ATT |
| UC3 | ATE(t) | BO warmup ~10% of budget | Warmup applies equally to ATT campaigns |
| UC4 | Bias analysis | Collider −41%, omission +135% | ATT estimation uses correct adjustment set |
| UC6 | Causal discovery | Which features drive BO advantage? | Will refine the SNR stratification |
| UC9 | CATE | Heterogeneous treatment effects | UC5 ATT is a special case of CATE at T=1 |

---

## 10. Visualisation

`uc5_att_counterfactuals.png` — three panels:
- **A**: Observed Y(1) vs counterfactual Y(0) scatter for 990 BO campaigns
- **B**: Distribution of individual switching losses; ATT (orange) > ATE (blue)
- **C**: SNR-stratified ATT bar chart; monotone increase low → mid → high
