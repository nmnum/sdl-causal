# Use Case 3: Convergence Curves — Time-Varying Strategy Effect

**Status**: Reasoning complete, implementation done
**DoWhy feature**: Time-varying ATE via OLS with `strategy × log(t)` interaction + cluster-robust SEs
**Unit of analysis**: (campaign, step) — one row per evaluation step within a campaign
**Extends**: UC2 — inherits all UC2 nodes, edges, and identification logic

---

## 1. The scientific question

> Does Bayesian optimisation converge faster than random search, and at what point in a campaign does the BO advantage emerge?

UC1 established that BO causally improves the *final* campaign outcome by +0.10 normalised units on average. UC2 showed this advantage is modulated by SNR. But both use cases treat the campaign as a single observation — they say nothing about *when* the advantage appears during the campaign.

This matters for the adaptive planner in two ways:

1. **Minimum budget**: BO requires a warm-up period to fit a useful GP surrogate. Below a minimum budget, BO's overhead is not amortised and RS is competitive. The planner must know this threshold before committing to BO.
2. **Early stopping**: If a campaign is running and BO is not yet outperforming RS at step t, should the planner switch strategies? The answer depends on whether t is still within the warmup period or whether BO has genuinely failed to find signal.

The design rule from UC3 will be: *"Do not evaluate BO vs RS on campaigns shorter than 2 × t_warmup evaluations — the comparison is confounded by warmup overhead. The minimum budget for BO to outperform RS is approximately `warmup_fraction × budget` evaluations, where `warmup_fraction` depends on dimensionality, SNR, and initial design quality."*

---

## 2. Variable classification

UC3 inherits all UC2 variables and adds one new node.

| Variable | Description | Role | New in UC3? |
|---|---|---|---|
| `strategy` | BO vs. random search | **Treatment** | No |
| `norm_best_outcome_t` | Best outcome found up to step t | **Outcome** | **Yes** (step-level) |
| `landscape_smoothness` | GP length-scale proxy | **Confounder** | No |
| `budget` | Total number of evaluations allowed | **Confounder** | No |
| `noise_level` | Measurement noise standard deviation | **Confounder** | No |
| `dimensionality` | Number of input variables | **Confounder** | No |
| `snr` | Signal-to-noise ratio | **Confounder + Moderator** | No |
| `initial_design_quality` | Space-filling score of seed points | **Non-confounder** | No |
| `batch_id` | Experimental batch identifier | **Noise variable** | No |
| `step` | Within-campaign evaluation index (1..budget) | **Time index** | **Yes** |

### Why `step` is a time index, not a confounder

`step` does not affect strategy choice — the strategy is fixed at the start of the campaign and does not change within it. `step` is therefore not a confounder. It is a within-campaign time index that moderates the strategy effect: the BO advantage grows with `step` as the GP surrogate accumulates observations.

`step` is also not a mediator in the standard sense — it is not caused by `strategy`. It is an exogenous index variable that interacts with `strategy` to produce the time-varying outcome.

### Why `warmup_fraction` is not a DAG node

`warmup_fraction` is a latent structural parameter, not an observable variable. It is a function of the observable confounders `{dimensionality, snr, initial_design_quality}` and is encoded directly in the structural equation. Adding it as a DAG node would require conditioning on a latent variable, which complicates identification without adding scientific value. The DAG captures the observable causal skeleton; the structural equation captures the functional form of the warmup mechanism.

---

## 3. The causal DAG

UC3 extends the UC2 DAG with one new node (`step`) and two new edges:

```
[All UC2 nodes and edges — see uc2_dag_reasoning.md]

step ──────────────────────────────────────────────────► norm_best_outcome_t
strategy × step ──────────────────────────────────────► norm_best_outcome_t
                                                         (time-varying moderation)
```

The `strategy × step` edge is not a standard DAG edge (DAGs do not have product nodes). It is encoded in the structural equation as a time-varying interaction term `f(snr) × g(t; warmup) × strategy`, where `g` is the warmup sigmoid defined in §4.

**Key structural property**: `step` has no parents in the DAG (it is exogenous — the campaign proceeds one step at a time regardless of what is observed). This means `step` cannot be a confounder and does not need to be in the adjustment set for the campaign-level causal effect. It is included as a covariate in the regression to control for the within-campaign time trend.

---

## 4. The warmup mechanism

### 4a. GP-theoretic justification

BO requires a minimum number of observations to fit a useful GP surrogate. With `n_init` random seed points, the GP has only `n_init` observations at step `n_init`. The posterior variance is high everywhere, and the acquisition function (e.g. Expected Improvement) selects points that are nearly random. As more observations accumulate, the posterior sharpens and the acquisition function begins to exploit the surrogate's predictions.

The warmup period ends when the GP has enough observations to provide useful guidance. Empirically (from the Olympus HPLC benchmark), this occurs around step 10–15 for a 6D landscape with budget=100 (i.e. `warmup_fraction ≈ 0.10`).

### 4b. The warmup function g(t; warmup)

The time-varying BO advantage is modelled as:

```
strategy_effect(t) = f(snr) × g(t; t_warmup, warmup_scale) × strategy
```

where:

```
g(t; t_warmup, warmup_scale) = sigmoid((t - t_warmup) / warmup_scale)
```

- `t_warmup = warmup_fraction × budget` — the step at which the BO advantage is at half its full value
- `warmup_scale = UC3_WARMUP_SCALE = 5.0` — sigmoid steepness (in steps); controls how abruptly the advantage emerges
- `warmup_fraction` — the fraction of budget before BO advantage is at half-maximum

Properties of g(t):
- `g(t → 0) → 0` — no BO advantage at the start of the campaign
- `g(t = t_warmup) = 0.5` — half the full advantage at the warmup step
- `g(t → ∞) → 1` — full BO advantage at the end of the campaign
- Monotone non-decreasing in t

### 4c. Warmup fraction as a function of confounders

```
warmup_fraction = clip(
    UC3_WARMUP_BASE
    + UC3_WARMUP_W_DIM  × dim_z
    - UC3_WARMUP_W_SNR  × log_snr_z
    - UC3_WARMUP_W_IDQ  × idq_z,
    min=0.05, max=0.40
)
```

where:
- `UC3_WARMUP_BASE = 0.10` — base warmup fraction at mean confounders (calibrated to HPLC: warmup ≈ 10/100 steps)
- `UC3_WARMUP_W_DIM = 0.03` — higher dimensionality → longer warmup (GP needs more points to cover the space)
- `UC3_WARMUP_W_SNR = 0.02` — higher SNR → shorter warmup (GP fits faster when signal is clean)
- `UC3_WARMUP_W_IDQ = 0.02` — better initial design → shorter warmup (seeds cover the space better)
- Clipped to [0.05, 0.40] to prevent degenerate warmup fractions

**Empirical basis**: Calibrated to the Olympus HPLC benchmark (6D, budget=100), where BO starts outperforming RS at approximately step 10 (warmup_fraction ≈ 0.10). The directional effects of dimensionality, SNR, and IDQ are theoretically motivated and consistent with the GP literature.

### 4d. Baseline convergence (RS)

Both strategies improve over time via the baseline log-linear convergence:

```
baseline(t) = intercept + UC3_ALPHA_RS × log(t + 1)
```

where:
- `UC3_ALPHA_RS = 0.10` — RS convergence rate (log-scale), calibrated to HPLC: RS reaches 0.78 at t=50 starting from ~0.55 at t=5
- `intercept = 0.40` — baseline at t=1 (before any improvement)

BO's additional convergence rate above RS is encoded in `f(snr) × g(t)`, not as a separate `alpha_bo` parameter. This ensures that the BO advantage at the end of the campaign exactly matches the UC2 true ATE = E[f(snr)], maintaining consistency across use cases.

**Empirical basis**: Log-linear fit to HPLC anchor points from the UC1 benchmark figure:
- RS: Y(t) ≈ 0.40 + 0.10 × log(t+1) (R² ≈ 0.99 on 6 anchor points)
- BO: Y(t) ≈ RS(t) + f(snr) × g(t) (by construction)

---

## 5. Identification strategy

### 5a. Adjustment set

The causal effect of `strategy` on `norm_best_outcome_t` is identified by the same backdoor adjustment as UC2:

```
{budget, dimensionality, landscape_smoothness, noise_level, snr}
```

`step` is not in the adjustment set — it is not a confounder. It is included as a covariate to control for the within-campaign time trend.

### 5b. Estimating the time-varying ATE

The time-varying ATE is estimated by OLS with a `strategy × log(t)` interaction term:

```
norm_best_outcome_t = β₀
                    + β₁ × strategy
                    + β₂ × log(t)
                    + β₃ × (strategy × log(t))
                    + β₄ × log_snr_std
                    + β₅ × budget_std
                    + β₆ × dim_std
                    + β₇ × idq_std
                    + ε
```

where `log(t) = log(step)` is the log of the within-campaign step index.

The time-varying ATE at step t is:

```
ATE(t) = β₁ + β₃ × log(t)
```

- `β₁` = ATE at t=1 (early campaign, before warmup; should be ≈ 0)
- `β₃` = how much the ATE grows per unit of log(t) (convergence rate difference)
- At t=budget: ATE(budget) ≈ E[f(snr)] (should match UC2 true ATE)

**Standard errors**: The (campaign, step) panel has within-campaign autocorrelation because `norm_best_outcome_t` is non-decreasing (best-found is monotone). OLS standard errors are therefore anti-conservative. We use **cluster-robust standard errors** at the campaign level (`cov_type='cluster', groups=campaign_id`) to correct for this.

### 5c. Why not a fixed-effects model?

Campaign fixed effects would absorb all campaign-level variation, including the confounders. This would make the strategy coefficient unidentified (strategy is constant within campaign). We instead control for campaign-level confounders explicitly, which is equivalent to fixed effects for the purpose of identifying the strategy effect when the confounders are correctly specified.

### 5d. Why log(t) rather than t?

The convergence curve is log-concave (diminishing returns). Using `log(t)` as the time covariate linearises the relationship, making the OLS interaction term interpretable as a constant convergence rate difference. Using raw `t` would require a polynomial or spline, adding parameters without improving interpretability.

---

## 6. Ground truth specification

### 6a. True time-varying ATE

The ground truth ATE at step t for a campaign with SNR = snr is:

```
ATE_true(t; snr) = f(snr) × g(t; t_warmup(snr, dim, idq), UC3_WARMUP_SCALE)
```

The population-average true ATE at step t is:

```
ATE_true(t) = E[f(snr) × g(t; t_warmup)]
```

This is computed from the simulator's `snr_modulation` and `g_t` columns.

### 6b. Key parameter values

| Parameter | Value | Empirical basis |
|---|---|---|
| `UC3_ALPHA_RS` | 0.10 | HPLC log-linear fit: RS reaches 0.78 at t=50 from 0.55 at t=5 |
| `UC3_WARMUP_BASE` | 0.10 | HPLC: BO starts outperforming RS at ~t=10 (budget=100) |
| `UC3_WARMUP_SCALE` | 5.0 | Sigmoid steepness: BO advantage emerges over ~5 steps |
| `UC3_WARMUP_W_DIM` | 0.03 | Dimensionality effect on warmup (theoretically motivated) |
| `UC3_WARMUP_W_SNR` | 0.02 | SNR effect on warmup (theoretically motivated) |
| `UC3_WARMUP_W_IDQ` | 0.02 | IDQ effect on warmup (theoretically motivated) |
| `UC3_STEP_NOISE` | 0.02 | Per-step measurement noise std (within-campaign) |

### 6c. Consistency with UC1 and UC2

By construction:
- At t=budget, g(t) ≈ 1.0, so `ATE(budget) ≈ f(snr)` — the campaign-level ATE matches UC2.
- At mean SNR, `E[f(snr)] ≈ 0.083` — consistent with UC2 true ATE.
- The UC1 ATE (+0.10) is the noise-free ceiling of f(snr), which is also the ceiling of ATE(t) as t→∞ and SNR→∞.

---

## 7. Design rules derived from UC3

> **Design rule UC3-A (minimum budget)**: BO requires a warm-up period before its advantage over RS emerges. The warm-up ends at approximately `t_warmup = warmup_fraction × budget` evaluations, where `warmup_fraction ≈ 0.10` at mean confounders (6D landscape, mean SNR). Below this budget, BO's overhead is not amortised and RS is competitive.
>
> **Actionable threshold**: The minimum budget for BO to outperform RS is approximately `2 × t_warmup` evaluations. For a 6D landscape with budget=50, this is ~10 evaluations. For a 3D landscape with high SNR, this is ~5 evaluations.

> **Design rule UC3-B (warmup depends on landscape)**: The warmup period is longer for higher-dimensional landscapes (more points needed to cover the space) and lower SNR (GP fits more slowly when noise dominates). Initial design quality also matters: a well-spread initial design shortens the warmup.
>
> **Actionable implication**: When planning a campaign, estimate `t_warmup` from dimensionality and SNR before committing to BO. If the available budget is less than `2 × t_warmup`, prefer RS or Latin hypercube.

> **Design rule UC3-C (convergence rate)**: After the warmup period, BO converges approximately 1.5× faster than RS on a log-step scale (calibrated to HPLC 6D). The convergence rate advantage is modulated by SNR: at low SNR, the advantage is smaller and emerges later.

> **Design rule UC3-D (early stopping caution)**: If a running BO campaign is not outperforming RS at step t, do not switch strategies if t < t_warmup. The comparison is confounded by warmup overhead. Only consider switching if t > t_warmup and the BO advantage has not emerged.

---

## 8. Connections to other use cases

- **UC1**: UC3 extends UC1 by making the strategy effect time-varying. The UC1 ATE (+0.10) is the value of ATE(t) at t=budget in the noise-free limit.
- **UC2**: UC3 inherits the SNR moderation from UC2. The warmup fraction also depends on SNR, creating an interaction between noise and budget requirements.
- **UC6** (distribution shift): If the noise level changes mid-campaign, the SNR changes and the warmup fraction changes. A campaign that started with adequate SNR may enter a low-SNR regime mid-way, causing the BO advantage to stall. UC6 will model this explicitly.
- **UC7** (budget sensitivity): UC3 provides the mechanistic basis for UC7's budget sensitivity analysis — the minimum budget threshold is derived from the warmup fraction.

---

## 9. Medicine formulation specifics

When translating to the curcumin/solubility setting:

- `step` → each formulation tested (1 to 256 in the Ros et al. dataset)
- `t_warmup` → estimated as `warmup_fraction × 256 ≈ 25` formulations (5D, SNR ≈ 1.9)
- At SNR ≈ 1.9 (near τ = 2.0), the warmup is longer than at mean SNR, and the BO advantage at the end of the campaign is smaller (~+0.02 from UC2). The convergence curve for the curcumin dataset is expected to show BO and RS nearly indistinguishable for the first ~25 formulations, with a small BO advantage emerging thereafter.
- The Ros et al. paper reports BO outperforming RS, but the margin is modest — consistent with the near-threshold SNR and the warmup overhead.
