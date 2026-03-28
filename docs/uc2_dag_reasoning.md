# Use Case 2: Noise Moderation of the Strategy Effect

**Status**: Reasoning complete, implementation done  
**DoWhy feature**: Moderated backdoor adjustment (OLS with interaction term)  
**Unit of analysis**: Campaign (one row = one complete experimental campaign)  
**Extends**: UC1 — inherits all UC1 nodes, edges, and identification logic

---

## 1. The scientific question

> Does measurement noise moderate the causal effect of strategy choice on campaign outcome? Specifically: is the BO advantage over random search smaller (or negative) at low signal-to-noise ratios, and does it recover above a critical SNR threshold?

UC1 established that BO causally improves campaign outcomes by +0.10 normalised units on average. But that average conceals a critical heterogeneity: the BO advantage depends on how well the GP surrogate can model the objective function, which in turn depends on the ratio of signal to noise.

This matters for the adaptive planner. A design rule that says "always use BO" is wrong in high-noise regimes. A design rule that says "use BO when SNR > τ" is actionable and transferable.

---

## 2. Variable classification

UC2 inherits all UC1 variables and adds one new node.

| Variable | Description | Role | New in UC2? |
|---|---|---|---|
| `strategy` | BO vs. random search | **Treatment** | No |
| `norm_best_outcome` | Best outcome found / true optimum | **Outcome** | No |
| `landscape_smoothness` | GP length-scale proxy | **Confounder** | No |
| `budget` | Number of experiments allowed | **Confounder** | No |
| `noise_level` | Measurement noise standard deviation | **Confounder** | No |
| `dimensionality` | Number of input variables | **Confounder** | No |
| `snr` | Signal-to-noise ratio = landscape_smoothness / noise_level | **Confounder + Moderator** | **Yes** |
| `initial_design_quality` | Space-filling score of seed points | **Non-confounder** | No |
| `batch_id` | Experimental batch identifier | **Noise variable** | No |

### Why `snr` is both a confounder and a moderator

`snr` confounds the strategy–outcome relationship in the same way as `noise_level` and `landscape_smoothness` individually: practitioners are more likely to choose BO when the SNR is high (clean signal, reliable surrogate). This creates a backdoor path `strategy ← snr → norm_best_outcome` that must be blocked.

`snr` also *moderates* the strategy effect: the BO advantage is not a fixed constant but a function of SNR. At high SNR, BO's structured search is highly effective; at low SNR, the GP surrogate is miscalibrated and BO degrades toward (or below) random search performance.

### Why `snr` is a derived variable, not independently drawn

`snr = landscape_smoothness / noise_level`. Both components are already in the UC1 DAG. We introduce `snr` as an explicit node because:
1. It is the mechanistically meaningful quantity (the GP literature characterises surrogate quality in terms of SNR, not the individual components).
2. It makes the moderation mechanism transparent and directly estimable.
3. It is dimensionless, which makes the threshold τ transferable across datasets.

Including `snr` in the adjustment set is technically redundant (conditioning on `landscape_smoothness` and `noise_level` already blocks the `snr` backdoor path), but it is scientifically cleaner and makes the interaction term directly interpretable.

---

## 3. The causal DAG

UC2 extends the UC1 DAG with one new node and two new edges:

```
landscape_smoothness ──► strategy ──────────────────────────► norm_best_outcome
budget               ──► strategy                                     ▲
noise_level          ──► strategy                                     │
dimensionality       ──► strategy                                     │
snr ───────────────────► strategy  (new: SNR confounds strategy)      │
                                                                      │
landscape_smoothness ─────────────────────────────────────────────────┤
budget               ─────────────────────────────────────────────────┤
noise_level          ─────────────────────────────────────────────────┤
dimensionality        ────────────────────────────────────────────────┤
snr ──────────────────────────────────────────────────────────────────┤ (new: direct)
initial_design_quality ───────────────────────────────────────────────┤
batch_id (noise) ─────────────────────────────────────────────────────┘

strategy × snr ───────────────────────────────────────────────────────► norm_best_outcome
                                                                         (moderation)
```

The moderation edge `strategy × snr → norm_best_outcome` is not a standard DAG edge (DAGs do not have product nodes). It is encoded in the structural equation as a non-linear term `f(snr) × strategy`, where `f` is the threshold+smooth-decay function defined in §4.

---

## 4. The moderation mechanism

### 4a. GP-theoretic justification for the threshold

In a Gaussian process surrogate, the posterior variance at a new point x is:

```
σ²_post(x) = k(x,x) - k(x,X) [K(X,X) + σ²_n I]⁻¹ k(X,x)
```

When the noise variance σ²_n is large relative to the signal variance σ²_f (the kernel amplitude), the term `[K(X,X) + σ²_n I]⁻¹` is dominated by the noise term, and the posterior variance barely shrinks from the prior. The acquisition function (e.g. Expected Improvement) then selects points almost uniformly — the GP is providing no useful guidance, and BO degrades toward random search.

The crossover point is SNR = σ²_f / σ²_n = 1.0. We set the threshold at **τ = 2.0** (noise variance = 50% of signal variance) to give a conservative margin: at SNR = 2.0 the GP is still informative but the BO advantage is halved relative to the noise-free case.

### 4b. The threshold + smooth-decay function

The moderation function f(snr) maps SNR to the BO advantage multiplier:

```
f(snr) = BO_FLOOR + (BO_CEILING - BO_FLOOR) × sigmoid((snr - τ) / scale)
```

where:
- `BO_FLOOR   = -0.05` — **asymptotic** floor as snr → 0 (sigmoid parameter, never reached in simulation because noise_level ~ LogNormal is always > 0 and the minimum simulated SNR ≈ 0.14 gives f ≈ −0.032)
- `BO_CEILING = +0.10` — **asymptotic** ceiling as snr → ∞ (sigmoid parameter; recovers UC1 ATE in the noise-free limit; saturates to within 1e-4 by SNR ≈ 20)
- `τ = 2.0`            — SNR threshold (GP-theoretic, dimensionless)
- `scale = 1.0`        — sigmoid steepness (1 SNR unit)

| SNR | f(snr) | Interpretation |
|---|---|---|
| → 0 | → −0.05 | Asymptotic floor (not reached in simulation) |
| **0.14 (sim min)** | **≈ −0.032** | **Actual minimum in simulated data** |
| 0.5 | ≈ −0.02 | BO marginally worse than RS |
| 1.0 | ≈ −0.01 | Near parity |
| **2.0 (τ)** | **+0.025** | **Half the maximum advantage — threshold** |
| 3.0 | +0.060 | BO clearly better |
| 5.0 | +0.093 | Near ceiling |
| → ∞ | → +0.10 | Asymptotic ceiling (recovers UC1 ATE) |

**Important**: ~90% of simulated campaigns have SNR > τ = 2.0 (because landscape_smoothness ~ LogNormal(1.0, 0.5) and noise_level ~ LogNormal(−1.0, 0.5) give a right-skewed SNR distribution with median ≈ 7.4). The threshold is therefore not a 50/50 split — it marks the lower tail where BO advantage collapses. This is intentional: the design rule is about identifying the minority of campaigns where BO should *not* be used, not about splitting the population evenly.

Properties: monotone increasing, bounded, differentiable, dimensionless threshold.

### 4c. Why not a hard threshold?

A hard threshold (f = 0 below τ, f = +0.10 above τ) would be simpler but:
1. It is not physically realistic — the GP surrogate degrades gradually, not abruptly.
2. It creates a discontinuity in the structural equation that makes OLS estimation unstable near τ.
3. The smooth sigmoid is consistent with the empirical Olympus results, which show a gradual rather than abrupt degradation of BO performance with noise.

### 4d. Why not a purely multiplicative interaction?

A purely multiplicative interaction `f(snr) = α × snr` would imply BO is always better than RS for any SNR > 0. This is too optimistic: at very low SNR, the GP surrogate is actively misleading (it confidently predicts high values in noisy regions), causing BO to waste evaluations. The floor at −0.05 captures this.

---

## 5. Identification strategy

### 5a. Adjustment set

The causal effect of `strategy` on `norm_best_outcome` is identified by backdoor adjustment on:

```
{budget, dimensionality, landscape_smoothness, noise_level, snr}
```

This extends the UC1 adjustment set by adding `snr`. The backdoor criterion is satisfied because:
1. No node in the adjustment set is a descendant of `strategy`.
2. The set blocks all backdoor paths, including the new path `strategy ← snr → norm_best_outcome`.

**Note on over-specification (M3)**: The UC2 DAG includes edges `landscape_smoothness → snr` and `noise_level → snr`. This means `snr` is a deterministic function of `landscape_smoothness` and `noise_level`, both of which are already in the UC1 adjustment set. Conditioning on `{landscape_smoothness, noise_level}` already blocks all backdoor paths through `snr` — adding `snr` to the adjustment set is over-specified but not incorrect. The 5-element set `{budget, dimensionality, landscape_smoothness, noise_level, snr}` is valid; the minimal sufficient set is `{budget, dimensionality, landscape_smoothness, noise_level}`. We include `snr` explicitly to make the moderation mechanism directly estimable in the regression model.

### 5b. Estimating the moderated ATE

The moderated ATE is estimated by OLS with a `strategy × log_snr` interaction term:

```
latent_outcome = β₀ + β₁·strategy + β₂·log_snr_std + β₃·(strategy × log_snr_std)
                + β₄·budget + β₅·dimensionality + β₆·initial_design_quality + ε
```

where `log_snr_std = (log(snr) − μ) / σ` is the standardised log-SNR.

**Critical**: Do NOT include `landscape_smoothness` and `noise_level` alongside `snr` in the same regression. Because `snr = clip(landscape_smoothness / noise_level, ...)`, conditioning on all three simultaneously creates near-perfect multicollinearity (VIF > 100), inflating standard errors and making coefficients unreliable. Use `snr` (or `log_snr_std`) as the sole SNR-related regressor.

The conditional ATE at a given standardised log-SNR value is:

```
ATE(log_snr_std) = β₁ + β₃ × log_snr_std
```

This is the linear approximation to f(snr) on the log-SNR scale. It correctly identifies the sign and direction of moderation and gives an unbiased estimate of the average ATE (E[ATE] = β₁ + β₃ × E[log_snr_std] = β₁ since E[log_snr_std] = 0 by construction).

### 5c. Why use latent_outcome (not norm_best_outcome)

Same reason as UC1: ceiling clipping at 1.0 disproportionately censors BO campaigns (which are preferentially assigned to high-SNR, easier problems that are more likely to hit the ceiling). OLS on `latent_outcome` is unbiased; OLS on `norm_best_outcome` underestimates the ATE.

---

## 6. Ground truth specification

### 6a. Transferability of τ = 2.0

The threshold τ = 2.0 is dimensionless (SNR = signal variance / noise variance). To apply it to a new campaign dataset:

1. Estimate signal variance: `Var(y_true)` from the GP surrogate's predicted values across the search space, or from the range of observed outcomes.
2. Estimate noise variance: `Var(ε)` from replicate measurements at the same input point.
3. Compute SNR = signal variance / noise variance.
4. If SNR < 2.0, prefer random search or Latin hypercube; if SNR > 2.0, BO is expected to outperform.

For the curcumin/solubility dataset (Ros et al.): replicate standard deviation ≈ 15–20% of the outcome range. Using the variance-based definition SNR = Var(signal) / Var(noise):
- Var(signal) ≈ Var(observed outcomes) − Var(noise) ≈ (0.30)² − (0.175)² ≈ 0.059 (using σ_obs ≈ 0.30, σ_noise ≈ 0.175 as midpoint of 15–20% range)
- Var(noise) ≈ (0.175)² ≈ 0.031
- SNR ≈ 0.059 / 0.031 ≈ **1.9**

This places the curcumin dataset *near* τ = 2.0, not comfortably above it. The "4–6" figure that arises from squaring the amplitude ratio (σ_signal/σ_noise)² is incorrect — it conflates the amplitude ratio with the variance ratio. The correct SNR ≈ 1.9 suggests BO is marginal for this system, which is consistent with the modest BO advantage reported in Ros et al. (BO outperformed RS, but not by a large margin).

### 6b. Effect sizes

| Effect | Value | Notes |
|---|---|---|
| `strategy → norm_best_outcome` (ceiling) | +0.10 | Noise-free limit; recovers UC1 ATE |
| `strategy → norm_best_outcome` (floor) | −0.05 | Very low SNR; BO slightly worse than RS |
| `strategy → norm_best_outcome` (at τ=2) | +0.025 | Half the maximum advantage |
| `snr → strategy` | +0.30 (log-odds) | Practitioners prefer BO at high SNR |
| `snr → norm_best_outcome` | +0.08 | Direct: high SNR → easier optimisation |
| True ATE = E[f(snr)] | ≈ +0.086 | Depends on SNR distribution; computed from simulator |

---

## 7. Design rules derived from UC2

> **Design rule UC2-A (primary)**: The BO advantage over random search is not a fixed constant — it is modulated by the signal-to-noise ratio of the experimental system. Below SNR ≈ τ, the BO advantage collapses to near zero (statistically indistinguishable from zero). Above SNR ≈ τ, the BO advantage is significantly positive. At the canonical τ = 2.0, the BO advantage approaches its noise-free ceiling of +0.10 normalised outcome units above SNR ≈ 5.
>
> **Actionable threshold**: Before deploying BO, estimate the campaign SNR from replicate measurements. If SNR < τ, use random search or Latin hypercube instead. If SNR > τ, BO is expected to outperform.
>
> **Precision of the claim**: The correct statement is *'BO advantage collapses to near zero below τ'*, not *'BO is worse than RS below τ'*. The sigmoid floor (−0.05) is asymptotic and only reached at SNR → 0; the simulated minimum SNR (~0.14) gives f ≈ −0.032, and the mean f(snr) below τ is near zero for all τ in the sensitivity grid.

> **Design rule UC2-B (secondary)**: The naive (unadjusted) BO advantage is inflated at all SNR levels because BO is preferentially deployed in high-SNR settings. Comparing BO and RS performance without conditioning on SNR will overestimate the BO advantage in low-SNR regimes and underestimate it in high-SNR regimes. Always stratify by SNR when benchmarking strategies.

> **Design rule UC2-C (transferability and robustness)**: The SNR threshold τ = 2.0 is dimensionless and transfers across campaign datasets without rescaling. It is derived from GP posterior variance theory, not from a specific dataset. Estimate SNR as Var(signal) / Var(noise) using replicate measurements.
>
> **τ sensitivity analysis** (`notebooks/uc2_tau_sensitivity.ipynb`): The qualitative design rule is robust to the choice of τ across {1.0, 1.5, 2.0, 2.5, 3.0}. For all values tested:
> - Below-threshold ATE is statistically indistinguishable from zero (p > 0.05, OLS on `latent_outcome`, controlling for `log_snr_std + budget + dimensionality + initial_design_quality`).
> - Above-threshold ATE is strongly positive (p < 10⁻⁴⁰) with adjusted ATEs of approximately **+0.114** across all τ values (N=2000, seed=42).
> - Naive ATE > true ATE for all τ (confounding direction is stable).
>
> The true ATE = E[f(snr; τ)] decreases from +0.093 (τ=1.0) to +0.092 (τ=3.0) as more campaigns fall below the threshold. The canonical τ = 2.0 gives true ATE ≈ +0.092 above threshold, which is the recommended operating point.
>
> **Note on OLS overestimation**: The adjusted above-threshold ATE (~+0.114) exceeds the true E[f(snr) | snr ≥ τ] (~+0.092) by approximately 24%. This residual bias arises because BO campaigns within the above-threshold stratum still have higher SNR than RS campaigns on average; `log_snr_std` controls for this but does not eliminate it entirely. The true f(snr) column in the notebook is the ground truth; the OLS estimate is a conservative upper bound on the BO advantage.

---

## 8. Connections to other use cases

- **UC1**: UC2 extends UC1 by making the strategy effect a function of SNR rather than a fixed scalar. The UC1 ATE (+0.10) is the noise-free ceiling of the UC2 moderation function.
- **UC3** (convergence curves): The SNR moderation will be visible in convergence curves — BO converges faster than RS at high SNR, but the gap narrows at low SNR.
- **UC6** (distribution shift): If the noise level changes mid-campaign (e.g. instrument recalibration), the SNR changes and the optimal strategy may switch. UC6 will model this explicitly.
- **UC7** (budget sensitivity): The SNR moderation interacts with budget — at low SNR, a larger budget is needed for BO to overcome its overhead and outperform RS.

### External benchmark framing (ADA, GDSC)

When applying UC2 design rules to real SDL datasets (e.g. the Berlinguette ADA dataset), the correct framing is **descriptive consistency check**, not causal validation. Specifically:

- ADA is observational: the strategy assignment mechanism is unknown, so the selection model cannot be identified without strong untestable assumptions. Estimating a causal ATE from ADA data would require assuming the same confounding structure as the UC2 DAG, which is circular.
- The appropriate ADA check is: *does the observed BO/RS performance gap correlate with estimated SNR in the expected direction?* This tests whether the design rule is directionally consistent with real data, not whether the causal effect size is correct.
- Do NOT claim that ADA "validates the selection model" — this is logically inconsistent. The selection model is an assumption, not an estimand. ADA can only test whether the observable implications of the model (e.g. BO campaigns have higher SNR, BO advantage increases with SNR) are consistent with the data.

---

## 9. Medicine formulation specifics

When translating to the curcumin/solubility setting:

- `snr` → estimated from replicate standard deviations in the Ros et al. dataset. The paper reports high variability (~15–20% CV). Using the variance-based definition SNR = Var(signal)/Var(noise), and assuming σ_obs ≈ 0.30 and σ_noise ≈ 0.175 (midpoint of the 15–20% CV range applied to the observed outcome range — these are *estimates*, not directly reported values), gives SNR ≈ 1.9. This places the curcumin dataset near τ = 2.0, not comfortably above it. The exact value is sensitive to the assumed σ_noise; the directional conclusion (near-threshold, not high-SNR) is robust across the plausible 15–20% CV range.
- The moderation function f(snr) predicts BO advantage ≈ +0.02 at SNR ≈ 1.9 (near the midpoint), which is consistent with a marginal rather than strong BO advantage.
- The τ sensitivity analysis confirms that the design rule holds for τ ∈ {1.0, …, 3.0}, so the near-threshold curcumin SNR does not undermine the qualitative conclusion.
- If a new formulation campaign has higher noise (e.g. a different analytical method), the SNR should be re-estimated from replicate measurements before deciding whether to use BO.
