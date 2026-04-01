# Use Case 6: Mid-Campaign Distribution Shift

**Status**: Reasoning complete, implementation done
**DoWhy feature**: Interrupted time series (ITS) with difference-in-differences; time-varying confounder identification
**Unit of analysis**: (Campaign, step) — one row per evaluation within a campaign
**Extends**: UC3 — inherits all UC3 nodes, edges, and identification logic

---

## 1. The scientific question

> If the noise level of an experimental system increases mid-campaign — due to instrument drift, reagent degradation, or environmental change — does this causally degrade campaign outcome? Does the degradation depend on *when* the shift occurs and *how large* it is? And does BO suffer more than random search?

UC3 established that BO converges faster than random search, with a warmup period before the advantage emerges. But UC3 assumes the noise level is constant throughout a campaign. In real SDL settings, this assumption is routinely violated:

- **Instrument drift**: HPLC baselines shift over a multi-day campaign; the noise floor rises as the column ages.
- **Reagent degradation**: A solvent batch opened mid-campaign may have higher water content, increasing measurement variability.
- **Environmental change**: Temperature or humidity fluctuations between morning and afternoon runs affect reproducibility.

When noise increases mid-campaign, the SNR drops. From UC2, we know that BO's advantage collapses below SNR τ = 2.0. If the shift is large enough to push the effective SNR below τ, BO's surrogate model — which was fitted on pre-shift (low-noise) data — becomes miscalibrated. The GP posterior is overconfident: it predicts low uncertainty in regions where the true noise is now high. This causes the acquisition function to exploit spuriously, wasting evaluations.

Random search is immune to this miscalibration: it does not maintain a surrogate model and therefore cannot be misled by a noise shift.

This matters for the adaptive planner. A design rule that says "use BO when SNR > τ at campaign start" is incomplete: it ignores the possibility that SNR may fall below τ mid-campaign. The planner needs to know:
1. How much does a mid-campaign shift degrade the final outcome?
2. Does the degradation depend on when the shift occurs (early vs. late)?
3. Is BO more sensitive to the shift than random search?
4. What is the correct causal identification strategy when noise is time-varying?

---

## 2. Variable classification

UC6 inherits all UC3 variables and adds three new nodes.

| Variable | Description | Role | New in UC6? |
|---|---|---|---|
| `strategy` | BO vs. random search | **Treatment** | No |
| `norm_best_outcome_t` | Best outcome found up to step t | **Outcome** | No |
| `landscape_smoothness` | GP length-scale proxy | **Confounder** | No |
| `budget` | Number of experiments allowed | **Confounder** | No |
| `noise_level` | Baseline (pre-shift) measurement noise std | **Confounder** | No |
| `dimensionality` | Number of input variables | **Confounder** | No |
| `snr` | Baseline SNR = landscape_smoothness / noise_level | **Confounder** | No |
| `step` | Within-campaign evaluation index (1..budget) | **Time index** | No |
| `initial_design_quality` | Space-filling score of seed points | **Non-confounder** | No |
| `batch_id` | Experimental batch identifier | **Noise variable** | No |
| `shift_time` | Step at which noise level increases | **Time-varying confounder** | **Yes** |
| `shift_magnitude` | Multiplicative factor: post-shift noise = noise_level × shift_magnitude | **Time-varying confounder** | **Yes** |
| `post_shift` | Binary indicator: 1 if step > shift_time, else 0 | **Derived** | **Yes** |

### Why `shift_time` and `shift_magnitude` are confounders

Both variables affect the outcome (a larger shift at an earlier time degrades the outcome more). Both also affect strategy choice in a realistic SDL: a practitioner who notices the noise increase mid-campaign may switch from BO to random search, or may decide to continue with BO hoping the surrogate will adapt. This creates a backdoor path:

```
shift_magnitude → strategy (practitioner switches away from BO)
shift_magnitude → norm_best_outcome_t (higher noise degrades outcome)
```

If we do not condition on `shift_magnitude` (and `shift_time`), the estimated strategy effect is confounded by the shift.

### Why `post_shift` is derived, not independently drawn

`post_shift = 1(step > shift_time)`. It is a deterministic function of `step` and `shift_time`. It is included as an explicit column for convenience in the regression model (the interrupted time series estimator uses it directly), but it is not an independent variable.

### The time-varying confounder problem

In UC1–UC5, all confounders are fixed at campaign start. The backdoor adjustment set `{budget, dimensionality, landscape_smoothness, noise_level, snr}` is sufficient because these variables do not change during the campaign.

In UC6, `noise_level` is no longer constant: it is `noise_level` before `shift_time` and `noise_level × shift_magnitude` after. The effective SNR at step t is:

```
effective_snr(t) = landscape_smoothness / effective_noise(t)
                 = snr                    if t ≤ shift_time
                 = snr / shift_magnitude  if t > shift_time
```

A static adjustment for `noise_level` (or `snr`) captures only the pre-shift value. It does not block the backdoor path through the post-shift noise. The correct adjustment set must include `shift_time` and `shift_magnitude`.

**This is the core identification challenge of UC6**: the static backdoor adjustment set from UC2/UC3 is insufficient when noise is time-varying. The analyst who ignores the shift will estimate a biased ATE.

---

## 3. The causal DAG

UC6 extends the UC3 DAG with three new nodes and their edges.

```
landscape_smoothness ──► strategy ──────────────────────────────► norm_best_outcome_t
budget               ──► strategy                                         ▲
noise_level          ──► strategy                                         │
dimensionality       ──► strategy                                         │
snr                  ──► strategy                                         │
shift_magnitude      ──► strategy  (new: large shifts may cause switch)   │
                                                                          │
landscape_smoothness ─────────────────────────────────────────────────────┤
budget               ─────────────────────────────────────────────────────┤
noise_level          ─────────────────────────────────────────────────────┤
dimensionality       ─────────────────────────────────────────────────────┤
snr                  ─────────────────────────────────────────────────────┤
initial_design_quality ───────────────────────────────────────────────────┤
step                 ─────────────────────────────────────────────────────┤
shift_time           ─────────────────────────────────────────────────────┤ (new)
shift_magnitude      ─────────────────────────────────────────────────────┤ (new)
post_shift           ─────────────────────────────────────────────────────┤ (new: derived)
batch_id (noise)     ─────────────────────────────────────────────────────┘

strategy × step × post_shift ─────────────────────────────────────────────► norm_best_outcome_t
                                                                              (three-way moderation)
```

### New edges

| Edge | Weight | Rationale |
|---|---|---|
| `shift_time → norm_best_outcome_t` | +0.05 | Later shifts degrade less of the campaign (positive: later shift_time_std = better outcome) |
| `shift_magnitude → norm_best_outcome_t` | −0.08 | Larger noise increases degrade outcome more |
| `shift_magnitude → strategy` | −0.15 | Practitioners may switch away from BO when they notice large noise increases |
| `post_shift → norm_best_outcome_t` | −0.04 | Post-shift steps are harder regardless of strategy |

---

## 4. The distribution shift mechanism

### 4a. Effective noise level at step t

```
effective_noise(t) = noise_level                          if t ≤ shift_time
                   = noise_level × shift_magnitude        if t > shift_time
```

The effective SNR at step t:

```
effective_snr(t) = landscape_smoothness / effective_noise(t)
```

### 4b. BO surrogate miscalibration after the shift

Before the shift, the GP surrogate is fitted on `t_shift` observations with noise variance `σ²_n`. After the shift, new observations arrive with noise variance `σ²_n × shift_magnitude²`. The GP does not know the noise has changed — it continues to use the original noise hyperparameter `σ²_n`.

This causes two problems:
1. **Overconfident predictions**: The GP underestimates the noise in post-shift observations, so the posterior variance is too small. The acquisition function (EI) exploits regions that appear certain but are actually noisy.
2. **Biased mean predictions**: The GP mean is pulled toward the noisy post-shift observations, which may not reflect the true objective.

The net effect is that BO's effective advantage after the shift is modulated by the *post-shift* SNR, not the pre-shift SNR:

```
strategy_effect(t) = f(effective_snr(t)) × g(t; t_warmup)
```

where `f` is the UC2 modulation function and `g` is the UC3 warmup sigmoid.

### 4c. Random search is shift-invariant

Random search does not maintain a surrogate model. Its performance at step t depends only on the number of evaluations made so far and the effective noise at step t (which affects the measurement of the best outcome found). The log-linear convergence rate `UC3_ALPHA_RS` is unchanged by the shift.

This means the shift *differentially* degrades BO relative to random search. The BO advantage after the shift is:

```
BO_advantage_post(t) = f(snr / shift_magnitude) × g(t; t_warmup)
                     ≤ f(snr) × g(t; t_warmup)   [since shift_magnitude ≥ 1]
```

### 4d. Shift timing effect

The fraction of the campaign affected by the shift is `(budget - shift_time) / budget`. An early shift (shift_time = 0.2 × budget) degrades 80% of the campaign; a late shift (shift_time = 0.8 × budget) degrades only 20%.

The expected degradation to the final outcome is approximately:

```
Δ_outcome ≈ [f(snr) - f(snr/shift_magnitude)] × [1 - g(shift_time; t_warmup)] × (1 - shift_time/budget)
```

This is maximised when:
- `shift_magnitude` is large (large SNR drop)
- `shift_time` is early (large fraction of campaign affected)
- The pre-shift SNR is near τ (so the shift pushes it below the threshold)

### 4e. Distribution parameters

```
shift_time      ~ Uniform(0.2 × budget, 0.8 × budget)
                  [shift occurs between 20% and 80% through the campaign]
                  [uniform: no prior on when instrument drift occurs]

shift_magnitude ~ LogNormal(log(3), 0.3)
                  [noise triples on average; 95% CI: [1.7×, 5.3×]]
                  [LogNormal: shift_magnitude ≥ 1 by construction (noise only increases)]
                  [log(3) ≈ 1.099: a 3× noise increase is a realistic instrument drift scenario]
                  [σ=0.3: moderate spread; 10th percentile ≈ 2.2×, 90th percentile ≈ 4.1×]
```

**Empirical basis for shift_magnitude**: A 3× noise increase corresponds to a 9× variance increase, which reduces SNR by 9×. For a campaign starting at SNR = 7.4 (the median in UC2/UC3), a 3× noise increase gives post-shift SNR ≈ 0.82 — well below τ = 2.0. This is a realistic scenario: HPLC column degradation can increase baseline noise by 2–5× over a multi-day campaign.

---

## 5. Identification strategy

### 5a. The static adjustment failure

The UC2/UC3 adjustment set `{budget, dimensionality, landscape_smoothness, noise_level, snr}` is **insufficient** for UC6. Here is why:

The backdoor path `strategy ← shift_magnitude → norm_best_outcome_t` is not blocked by the static set. `shift_magnitude` is not in the UC2/UC3 adjustment set, and it is not a function of any variable in that set (it is drawn independently). Conditioning on `noise_level` blocks the path through the pre-shift noise, but not through `shift_magnitude`.

Empirically: a naive OLS regression using the UC2/UC3 adjustment set on UC6 data will underestimate the BO advantage. The reason is that large `shift_magnitude` values both degrade BO (via surrogate miscalibration) and are associated with strategy switching (practitioners move away from BO when they notice the noise increase). This creates a spurious negative correlation between BO and outcome that is not present in the true causal effect.

### 5b. The correct adjustment set

The correct adjustment set for UC6 is:

```
{budget, dimensionality, landscape_smoothness, noise_level, snr, shift_time, shift_magnitude}
```

This extends the UC2/UC3 set by adding `shift_time` and `shift_magnitude`. The backdoor criterion is satisfied because:
1. No node in the adjustment set is a descendant of `strategy`.
2. The set blocks all backdoor paths, including the new paths through `shift_time` and `shift_magnitude`.

### 5c. The interrupted time series estimator

The natural estimator for UC6 is an **interrupted time series (ITS)** model on the step-level panel:

```
norm_best_outcome_t =
    β₀
  + β₁ × strategy
  + β₂ × log(step + 1)
  + β₃ × strategy × log(step + 1)
  + β₄ × post_shift
  + β₅ × post_shift × strategy
  + β₆ × post_shift × log(step + 1)
  + β₇ × post_shift × strategy × log(step + 1)
  + [UC2/UC3 confounder terms]
  + [shift_time_std + shift_magnitude_std]
  + ε
```

Key coefficients:
- `β₁`: BO advantage at step 1, pre-shift (baseline)
- `β₃`: BO convergence rate advantage, pre-shift
- `β₄`: Level shift in outcome at the shift point (affects both strategies)
- `β₅`: **Differential BO degradation** — the additional drop in BO advantage post-shift
- `β₇`: Differential BO convergence rate change post-shift

The estimand of interest is `β₅`: the causal effect of the shift on the BO advantage. A negative `β₅` confirms that BO is more sensitive to the shift than random search.

Standard errors are clustered at the campaign level (same reason as UC3: within-campaign autocorrelation from the monotone best-found trajectory).

### 5d. Difference-in-differences interpretation

The ITS estimator has a difference-in-differences (DiD) interpretation:

```
β₅ = [E[Y_BO(post) - Y_BO(pre)] - E[Y_RS(post) - Y_RS(pre)]]
```

This is the DiD estimator with BO as the "treated" group and RS as the "control" group, where "treatment" is the noise shift. The parallel trends assumption requires that, absent the shift, BO and RS would have continued to diverge at the same rate as pre-shift. This is satisfied by construction in the simulator (the pre-shift convergence rates are fixed by the UC3 structural equation).

---

## 6. Ground truth specification

### 6a. True causal effects

| Effect | Value | Notes |
|---|---|---|
| `shift_magnitude → norm_best_outcome_t` | −0.08 | Direct: larger shift → worse outcome |
| `shift_time → norm_best_outcome_t` | +0.05 | Later shift → less of campaign degraded (positive on standardised scale) |
| `shift_magnitude → strategy` | −0.15 | Log-odds: large shifts reduce P(BO) |
| `post_shift → norm_best_outcome_t` | −0.04 | Post-shift steps are harder |
| BO differential degradation (β₅) | ≈ −0.03 to −0.05 | Depends on shift_magnitude distribution |

### 6b. True ATE under static vs. correct adjustment

| Adjustment set | Expected ATE estimate | Bias direction |
|---|---|---|
| Static (UC2/UC3 set, ignoring shift) | Underestimate | Negative (shift confounds BO negatively) |
| Correct (+ shift_time, shift_magnitude) | Unbiased | None |
| Oracle (known shift parameters) | Unbiased | None |

The bias from ignoring the shift is expected to be approximately −0.01 to −0.03 (10–30% of the true ATE), depending on the shift_magnitude distribution.

---

## 7. Design rules derived from UC6

> **Design rule UC6-A (primary)**: A mid-campaign noise increase causally degrades campaign outcome. The degradation is larger when the shift occurs earlier in the campaign and when the shift magnitude is larger. For a 3× noise increase at the median shift time (50% of budget), the expected outcome degradation is approximately −0.04 normalised units.

> **Design rule UC6-B (BO sensitivity)**: BO is more sensitive to mid-campaign noise increases than random search. After a noise shift that pushes the effective SNR below τ = 2.0, the BO advantage collapses — the GP surrogate is miscalibrated on the post-shift data. Random search is shift-invariant (no surrogate to miscalibrate).
>
> **Actionable implication**: If instrument noise increases mid-campaign, consider switching from BO to random search if the estimated post-shift SNR falls below τ = 2.0. The switching cost (losing the BO advantage) is smaller than the cost of continuing with a miscalibrated surrogate.

> **Design rule UC6-C (identification)**: The static backdoor adjustment set from UC2/UC3 is insufficient when noise is time-varying. Ignoring the shift confounders (`shift_time`, `shift_magnitude`) underestimates the BO advantage by approximately 10–30%. Always include shift parameters in the adjustment set when analysing campaigns with known or suspected noise changes.

> **Design rule UC6-D (monitoring)**: The shift is detectable from the step-level outcome trajectory: a sudden increase in outcome variance (or a drop in the running maximum improvement rate) at step t is a signal of a noise shift. An adaptive planner should monitor the residual variance of the GP surrogate and trigger a strategy re-evaluation when it exceeds a threshold.

---

## 8. Connections to other use cases

- **UC2**: UC6 is the dynamic version of UC2's noise moderation. UC2 shows that high noise degrades BO at campaign start; UC6 shows that a mid-campaign noise increase has the same effect, but only for the post-shift portion of the campaign.
- **UC3**: UC6 extends UC3's step-level panel with a noise shift. The warmup sigmoid `g(t; t_warmup)` is unchanged; the shift modifies the SNR modulation `f(snr)` for post-shift steps only.
- **UC5**: The ATT estimated in UC5 is biased if some BO campaigns experienced a noise shift that caused practitioners to switch strategies. UC6 provides the identification fix.
- **UC7** (budget sensitivity): The interaction between shift timing and budget is a UC7 question — does a larger budget mitigate the shift degradation?
- **UC8** (mediation): The mechanism by which the shift degrades BO (surrogate miscalibration vs. direct noise effect on measurements) is a mediation question deferred to UC8.

---

## 9. Medicine formulation specifics

When translating to the curcumin/solubility setting:

- `shift_time` → the formulation batch number at which the analytical instrument was recalibrated or a new reagent lot was introduced. In the Ros et al. dataset (256 formulations over multiple days), instrument drift between days is a plausible source of noise shifts.
- `shift_magnitude` → estimated from the ratio of within-day to between-day measurement variability. If the between-day CV is 2× the within-day CV, shift_magnitude ≈ 2.
- The design rule UC6-B implies that if a noise shift is detected mid-campaign (e.g. by monitoring replicate measurements), the planner should consider switching from BO to random search or Latin hypercube until the noise is characterised and the surrogate is recalibrated.
- The curcumin dataset (SNR ≈ 1.9, near τ = 2.0) is particularly vulnerable to noise shifts: even a modest 1.5× noise increase would push the effective SNR to ≈ 0.84, well below τ. This makes UC6 especially relevant for this system.

---

## 10. Visualisation

`uc6_distribution_shift.png` — four panels:
- **A**: Step-level convergence curves (BO vs RS) for a representative campaign, with the shift point marked. Shows the BO advantage collapsing post-shift.
- **B**: ATE bias plot — estimated ATE under static vs. correct adjustment set, across 20 seeds. Shows the static set underestimates the BO advantage.
- **C**: Shift timing effect — final outcome degradation as a function of shift_time / budget. Shows earlier shifts cause more degradation.
- **D**: Shift magnitude effect — BO vs RS outcome degradation as a function of shift_magnitude. Shows BO is more sensitive than RS.
