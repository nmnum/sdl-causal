# Use Case 4: Confounder Identification Sensitivity

**Status**: Reasoning complete, implementation done
**DoWhy feature**: Adjustment set misspecification — omission, spurious inclusion, collider bias
**Unit of analysis**: Campaign (one row = one complete experimental campaign)
**Extends**: UC2 — inherits all UC2 nodes, edges, and structural equations
**Data**: All results are from **synthetic simulation** (n=2000, seed=42). True ATE is known
by construction as E[snr_modulation]. This is a simulation study, not an empirical analysis.

---

## 1. The scientific question

> Are the confounders in our simulation design correctly identified? What is the bias
> from using the wrong adjustment set — omitting true confounders, adding spurious
> variables, or conditioning on a collider?

UC1–UC3 assumed the correct adjustment set was known and used it faithfully. UC4 stress-tests
that assumption by systematically varying what gets adjusted for and measuring the resulting
bias against the known ground truth ATE.

This matters for the adaptive planner. A planner that selects the wrong adjustment set will
produce biased ATE estimates, leading to incorrect strategy recommendations. UC4 quantifies
how much each type of misspecification costs.

---

## 2. Variable classification

UC4 inherits all UC2 variables and adds one new node.

| Variable | Description | Role | New in UC4? |
|---|---|---|---|
| `strategy` | BO vs. random search | **Treatment** | No |
| `norm_best_outcome` | Best outcome found / true optimum | **Outcome** | No |
| `landscape_smoothness` | GP length-scale proxy | **Confounder** | No |
| `budget` | Number of experiments allowed | **Confounder** | No |
| `noise_level` | Measurement noise standard deviation | **Confounder** | No |
| `dimensionality` | Number of input variables | **Confounder** | No |
| `snr` | Signal-to-noise ratio = ls / nl | **Confounder (derived)** | No |
| `initial_design_quality` | Space-filling score of seed points | **Non-confounder** | No |
| `batch_id` | Experimental batch identifier | **Noise variable** | No |
| `observed_flag` | Binary success flag | **Collider** | **Yes** |

### Why `observed_flag` is a collider, not a confounder

`observed_flag` is caused by **both** `strategy` and `norm_best_outcome`:

```
P(observed_flag = 1) = sigmoid(
    +1.5 × strategy              # BO campaigns flagged more often
    +2.0 × norm_best_outcome_std # better outcomes flagged more often
    −1.0                         # base intercept → ~27% base rate
)
```

This is a realistic SDL scenario: an experimenter marks a campaign as "successful" based
on both the strategy used (BO is perceived as more rigorous) and the result obtained
(better outcomes are more likely to be flagged). The flag is a **downstream consequence**
of both treatment and outcome — it is not a cause of either.

A collider is the opposite of a confounder:
- **Confounder**: common cause of treatment and outcome → must adjust for it
- **Collider**: common effect of treatment and outcome → must NOT adjust for it

Conditioning on a collider opens a non-causal path (Berkson's paradox) and introduces
bias even when the correct confounders are otherwise included.

---

## 3. The causal DAG

UC4 extends the UC2 DAG with one new node and two new edges:

```
landscape_smoothness ──► strategy ──────────────────────────► norm_best_outcome
budget               ──► strategy                                     ▲
noise_level          ──► strategy                                     │
dimensionality       ──► strategy                                     │
snr ───────────────────► strategy                                     │
                                                                      │
landscape_smoothness ─────────────────────────────────────────────────┤
budget               ─────────────────────────────────────────────────┤
noise_level          ─────────────────────────────────────────────────┤
dimensionality        ────────────────────────────────────────────────┤
snr ──────────────────────────────────────────────────────────────────┤
initial_design_quality ───────────────────────────────────────────────┤
batch_id (noise) ─────────────────────────────────────────────────────┘

strategy         ──► observed_flag   (new: collider — caused by treatment)
norm_best_outcome ──► observed_flag  (new: collider — caused by outcome)
```

The structural equations for `strategy` and `norm_best_outcome` are **identical to UC2**.
Only the collider column is added downstream. The true ATE is therefore identical to UC2:
E[f(snr)] ≈ 0.083 (varies slightly with seed).

---

## 4. The 8 adjustment sets

| # | Label | Covariates | Bias type | Expected bias direction |
|---|---|---|---|---|
| 1 | `correct` | {budget, dim, ls, nl, snr} | none | zero |
| 2 | `drop_snr` | {budget, dim, ls, nl} | omission | zero (snr redundant given ls, nl) |
| 3 | `drop_ls` | {budget, dim, nl, snr} | omission | positive |
| 4 | `drop_nl` | {budget, dim, ls, snr} | omission | negative |
| 5 | `drop_snr_related` | {budget, dim} | omission | positive |
| 6 | `empty` | {} | omission | positive |
| 7 | `add_non_confounder` | correct + idq | spurious_inclusion | zero |
| 8 | `collider` | correct + observed_flag | collider | negative (Berkson) |

**Note on sets 1 and 2**: The minimal sufficient adjustment set is {budget, dim, ls, nl}
(`drop_snr`). `snr` is a deterministic function of `ls` and `nl` — conditioning on both
parents already blocks all backdoor paths through `snr`. Set 1 (`correct`) includes `snr`
as well, which is valid but introduces slight over-specification in OLS due to near-collinearity
with `ls` and `nl`. Set 2 (`drop_snr`) is the better OLS specification and recovers the
true ATE with lower bias.

---

## 5. Identification strategy

### 5a. Correct adjustment set

The causal effect of `strategy` on `norm_best_outcome` is identified by backdoor adjustment on:

```
{budget, dimensionality, landscape_smoothness, noise_level}
```

This is the **minimal sufficient set** — it blocks all backdoor paths without over-specifying.
Adding `snr` is valid but redundant (see UC2 §5a). The OLS estimator is:

```
latent_outcome = β₀ + β₁·strategy + β₂·ls_std + β₃·nl_std + β₄·bud_std + β₅·dim_std + ε
```

### 5b. Why `latent_outcome` not `norm_best_outcome`

Same reason as UC1/UC2: ceiling clipping at 1.0 disproportionately censors BO campaigns.
OLS on `latent_outcome` is unbiased; OLS on `norm_best_outcome` underestimates the ATE.

### 5c. Why `observed_flag` must not be in the adjustment set

`observed_flag` is a descendant of both `strategy` and `norm_best_outcome`. Conditioning
on it violates the backdoor criterion by opening the path:

```
strategy → observed_flag ← norm_best_outcome
```

This is a v-structure (collider). In an unblocked DAG, this path is inactive (d-separated).
Conditioning on the collider **activates** it, inducing spurious correlation between
`strategy` and `norm_best_outcome` that is not causal.

---

## 6. Empirical results (synthetic simulation, n=2000, seed=42)

**True ATE = 0.083** (known by construction as E[snr_modulation])

| Adjustment set | ATE estimate | Bias | Bias % | CI covers truth |
|---|---|---|---|---|
| correct {ls, nl, bud, dim} | 0.084 | +0.001 | +1.6% | Yes |
| drop_snr {ls, nl, bud, dim} | 0.084 | +0.001 | +1.6% | Yes |
| drop_ls {nl, snr, bud, dim} | 0.091 | +0.009 | +10.3% | Yes |
| drop_nl {ls, snr, bud, dim} | 0.069 | −0.014 | −16.9% | Yes |
| drop_snr_related {bud, dim} | 0.195 | +0.112 | +135% | No |
| empty {} | 0.220 | +0.137 | +165% | No |
| add_non_confounder (+ idq) | 0.085 | +0.002 | +2.7% | Yes |
| collider (+ observed_flag) | 0.049 | −0.034 | −41% | No |

### Key observations

1. **SNR is redundant given ls and nl**: Sets 1 and 2 give identical results. The derived
   confounder `snr = ls/nl` adds no information once both parents are adjusted for.

2. **Asymmetric omission bias**: Dropping `ls` inflates the ATE (+10%) while dropping `nl`
   deflates it (−17%). This asymmetry reflects the opposing signs of their direct effects
   on outcome: `ls → outcome` is positive (+0.20), `nl → outcome` is negative (−0.15).
   Omitting a positive-effect confounder that also positively predicts treatment inflates
   the ATE; omitting a negative-effect confounder that negatively predicts treatment deflates it.

3. **Catastrophic omission**: Dropping all SNR-related confounders (+135%) or using no
   adjustment (+165%) produces estimates 2–3× the true ATE. The naive ATE is dominated
   by selection bias: BO is preferentially assigned to high-SNR, easy campaigns.

4. **Non-confounder is harmless**: Adding `initial_design_quality` changes the ATE by
   only +2.7% and the CI still covers the truth. This confirms the theoretical prediction:
   a variable that affects outcome but not treatment does not bias the ATE estimate.

5. **Collider bias is downward (Berkson's paradox)**: Conditioning on `observed_flag`
   introduces −41% bias relative to the true ATE (0.083). The direction is negative
   because within each stratum of `observed_flag`, the strategy–outcome correlation is
   attenuated relative to the true causal effect:
   - Among flagged campaigns: BO campaigns were flagged partly because of strategy
     (not just outcome), so their outcomes are lower than expected given the flag.
   - Among unflagged campaigns: the same logic applies in reverse.
   The within-stratum OLS slopes (0.052 unflagged, 0.078 flagged) are both below the
   true ATE (0.083) — a 6–37% attenuation relative to truth. For reference, the naive
   (unadjusted) slope is 0.220, which is 2.6× inflated by confounding; the collider
   set does not "fix" confounding, it adds a second source of bias on top of it.

   **Important caveat**: The downward direction of collider bias is specific to this
   construction (both collider weights positive: strategy +1.5, outcome +2.0). If one
   or both weights were negative, the induced correlation could be upward, producing
   positive (upward) bias instead. The magnitude and direction of Berkson's bias depend
   on the signs and magnitudes of the collider's parent weights — it is not universally
   downward.

---

## 7. Design rules derived from UC4

> **Design rule DR-E (omission — primary)**: Omitting any single SNR-related confounder
> introduces 10–17% bias. Omitting all SNR-related confounders inflates the ATE by >100%.
> The naive (unadjusted) ATE is 2.6× the true ATE. The minimal sufficient adjustment set
> for this SDL simulation is {budget, dimensionality, landscape_smoothness, noise_level}.
> Always include at minimum these four variables.
>
> **Corollary**: The derived confounder `snr` is redundant given `landscape_smoothness`
> and `noise_level`. Including it adds no bias reduction but does enable direct estimation
> of the SNR moderation effect (UC2). Including all three simultaneously over-specifies
> the model and introduces slight downward bias (−13%) due to near-collinearity.

> **Design rule DR-F (spurious inclusion — secondary)**: Including a non-confounder
> (a variable that affects outcome but not treatment) is safe. It changes the ATE estimate
> by <3% and does not invalidate the CI. Over-specification of the adjustment set is
> acceptable when the extra variable is not a descendant of treatment. This gives the
> adaptive planner a conservative default: when uncertain whether a variable is a
> confounder, including it is safer than excluding it — provided it is not post-treatment.

> **Design rule DR-G (collider — critical)**: Conditioning on a post-treatment collider
> introduces large bias via Berkson's paradox. In this simulation, conditioning on
> `observed_flag` (caused by both strategy and outcome) introduces −41% bias and the CI
> no longer covers the truth.
>
> **Direction caveat**: The bias direction is not universally downward. In this construction
> both collider weights are positive (strategy: +1.5, outcome: +2.0), which produces
> downward (negative) bias via Berkson's paradox. If one or both weights were negative,
> the induced spurious correlation could be upward, producing positive (upward) bias.
> The direction depends on the signs of the collider's parent weights; the magnitude
> depends on the strength of those weights and the collider's prevalence.
>
> **Actionable rule**: Never condition on variables that are caused by both the treatment
> and the outcome, regardless of the expected bias direction. In SDL contexts, this
> includes: campaign success flags, publication decisions, follow-up experiment decisions,
> and any variable measured after the campaign outcome is known. These are all potential
> colliders.

---

## 8. Connections to other use cases

- **UC1**: UC4 validates that the UC1 adjustment set {budget, dim, ls, nl} is correct.
  The UC1 ATE (+0.10) is recoverable with <2% bias using this set.
- **UC2**: UC4 confirms that `snr` is redundant given `ls` and `nl` (sets 1 and 2 are
  identical). The UC2 moderation story requires `snr` for interpretability, not for
  identification.
- **UC3**: The UC3 panel estimator uses the same adjustment set as UC2. UC4 confirms
  this set is correct at the campaign level; the panel extension does not change the
  identification argument.
- **Adaptive planner**: DR-E, DR-F, DR-G are directly actionable rules for the planner's
  covariate selection module. The planner should: (1) always include the four core
  confounders, (2) treat non-confounders as safe additions, (3) flag post-treatment
  variables as potential colliders and exclude them from adjustment.

---

## 9. Limitations and caveats

1. **Simulation only**: All bias estimates are from synthetic data with a known DGP.
   Real SDL datasets have unknown confounding structure; the true ATE cannot be computed
   directly. UC4 validates the identification strategy under the assumed DGP, not under
   all possible DGPs.

2. **Linear OLS estimator**: The bias table uses OLS throughout. Non-linear estimators
   (e.g. doubly-robust, IPW) may have different bias profiles under the same misspecification.

3. **Collider construction**: The `observed_flag` collider is constructed with specific
   weights (strategy: +1.5, outcome: +2.0). Different weights would change both the
   magnitude and potentially the direction of the collider bias. With both weights
   positive, the induced spurious correlation is negative (downward bias). If one weight
   were negative, the direction could reverse. Berkson's paradox is a structural property
   of conditioning on a collider, but the sign of the resulting bias depends on the
   signs of the collider's parent weights.

4. **n=2000**: The bias table uses n=2000 for sufficient power. At n=500 (the UC1/UC2
   default), the correct-set bias is still <5% but the CI is wider, and some omission
   sets may still have CIs that cover the truth by chance.
