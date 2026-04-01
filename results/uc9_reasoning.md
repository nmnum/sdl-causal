# UC9 Reasoning Document: Heterogeneous Treatment Effects (CATE)

## 1. Scientific question

Does the advantage of RBF-surrogate BO over random search vary systematically with
campaign characteristics — specifically budget, SNR, and input dimensionality?

**Why this matters:** DR-A (UC1) gives a single average treatment effect. The adaptive
controller needs to know *when* to use BO — i.e., which regime of (budget, SNR, n_dims)
space makes BO worth deploying. UC9 answers this by estimating a separate CATE for every
campaign and identifying break-even thresholds.

## 2. Data

All data is synthetic simulation output (RBF-surrogate BO vs random search on a 2D
Branin-like landscape). This is intentional — the project studies optimisation strategy
behaviour under controlled conditions. Results are not from a real wet-lab experiment.

- N_CAMPAIGNS = 500 per strategy (1000 total observations)
- Budget: varied in {30, 35, 40, 45, 50} steps
- Noise: varied uniformly in [0.05, 0.60] (SNR proxy = 1/noise_std)
- n_dims = 2 (constant)

## 3. Method: CausalForestDML

CausalForestDML (econml 0.16) with:
- Outcome nuisance: GradientBoostingRegressor (100 trees, max_depth=3)
- Propensity nuisance: LogisticRegression
- Forest: 500 honest trees, min_samples_leaf=25
- Inference: pointwise 95% CIs via forest variance estimator

**Honest splitting:** Separate subsamples for tree construction and leaf estimation.
Reduces overfitting bias at the cost of higher variance — appropriate here because
we prioritise unbiased CATE estimates over predictive accuracy.

**Identification assumption:** Unconfoundedness conditional on (budget, SNR, n_dims).
In this simulation, treatment assignment is random within each campaign pair, so
unconfoundedness holds by design.

## 4. Results

### 4.1 Global CATE
- Global CATE = +0.2210 [0.1708, 0.2711]
- 100% of campaigns have positive CATE
- CATE std = 0.0347 (heterogeneity indicator)

### 4.2 Subgroup ATEs
Split at median SNR=3.05, median budget=40:
- high_snr_long_budget (n=286): ATE=+0.1893 [0.1470, 0.2315]
- high_snr_short_budget (n=214): ATE=+0.1933 [0.1535, 0.2330]
- low_snr_long_budget (n=294): ATE=+0.2603 [0.1999, 0.3206]
- low_snr_short_budget (n=206): ATE=+0.2377 [0.1800, 0.2953]

### 4.3 Break-even thresholds
- SNR break-even:    not found (CATE always positive)
- Budget break-even: not found (CATE always positive)

### 4.4 Variable importance
- snr: 0.8206
- budget: 0.1794
- n_dims: 0.0000
Top moderator: snr

## 5. Design rules

**DR-H-subgroup:** Use BO when SNR and budget are above their respective break-even
thresholds. In regimes below these thresholds, BO's advantage vanishes and random
search is equally effective at lower computational cost.

**DR-H-varimp:** The top moderator of the treatment effect is the covariate the
adaptive controller should monitor most closely. If SNR dominates, monitor noise
estimates; if budget dominates, monitor remaining budget.

## 6. Caveats and limitations

1. **Synthetic data only.** All results are from simulation. Thresholds must be
   validated on real experimental data before deployment.
2. **n_dims is constant** within this simulation — its variable importance is not
   identifiable and should be treated as noise.
3. **Honest splitting** reduces bias but increases variance. With n<200 per arm,
   CATE estimates for individual campaigns are noisy.
4. **Subgroup CIs** are averaged pointwise — they are not joint CIs over the subgroup.
5. **Unconfoundedness** holds by design in simulation. In real data, unmeasured
   confounders (operator choices, equipment state) may bias CATE estimates.
