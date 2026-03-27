"""
simulator.py — UC1 campaign simulator for the Self Driving Lab project.

Public API
----------
    run_campaigns(n_campaigns, budget, ...) -> pd.DataFrame

One row per campaign.  Column schema is fixed; later use cases add optional
columns that default to NaN here.  The simulator imports causal structure
from graphs.py — it never redefines the DAG inline.

Causal model
------------
The data-generating process faithfully encodes the UC1 DAG from graphs.py:

  1. Confounders are drawn from distributions fitted to the four Olympus
     benchmark datasets (HPLC 6D, SnAr 4D, Suzuki 4D, Fullerenes 3D).
     They are NOT drawn from arbitrary uniform distributions.

  2. Strategy assignment is a logistic function of the confounders, so BO
     is preferentially assigned to easier problems - exactly the selection
     bias that DoWhy's backdoor adjustment must correct.

  3. norm_best_outcome is a linear combination of the causal effects encoded
     in the graph edge weights, plus a batch random intercept and residual
     noise. The strategy coefficient (+0.10) is the ground truth ATE that
     DoWhy must recover.

  4. initial_design_quality is drawn independently of strategy (it is a
     non-confounder by simulation design - see uc1_dag_reasoning.md §2).

  5. batch_id is a categorical noise variable.  Its effect is a random
     intercept drawn from N(0, batch_sigma) per batch.

Empirical basis for distribution parameters
--------------------------------------------
Confounder distributions are fitted to GP length scales and noise levels
estimated from the four Olympus datasets:

    Dataset      dim   length_scale   noise_std
    HPLC           6         0.3095      0.1634
    SnAr           4         0.8629      0.0153
    Suzuki         4         1.1211      0.1836
    Fullerenes     3         0.6247      0.0678

    landscape_smoothness ~ LogNormal(mu=-0.4191, sigma=0.4819)
    noise_level          ~ LogNormal(mu=-2.5944, sigma=0.9931)
    dimensionality       ~ Categorical({3: 0.25, 4: 0.50, 6: 0.25})

Reference: uc1_dag_reasoning.md §6a, uc1_bo_vs_random_benchmark.png
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

# graphs.py is the single source of truth for edge weights.
# Import the DAG and read weights from it rather than hardcoding them here.
from src.graphs import uc1_dag, get_adjustment_set, UC1_NODE_META


# ── Fixed output schema ───────────────────────────────────────────────────────
# All columns that run_campaigns() guarantees to return.
# Later UCs may add columns; UC1-only columns are listed here.
# NaN-valued optional columns for future UCs are appended at the end.

UC1_COLUMNS: list[str] = [
    "campaign_id",            # int, unique campaign identifier
    "strategy",               # int, 0=random search, 1=BO
    "norm_best_outcome",      # float in [0, 1], best found / true optimum
    "landscape_smoothness",   # float > 0, GP length-scale proxy
    "budget",                 # int, number of evaluations allowed
    "noise_level",            # float >= 0, measurement noise std
    "dimensionality",         # int in {3, 4, 6}
    "initial_design_quality", # float in [0, 1], space-filling score of seeds
    "batch_id",               # int, batch identifier
    # ── Provenance ────────────────────────────────────────────────────────────
    "p_bo",                   # float, P(strategy=BO) before Bernoulli draw
    "batch_intercept",        # float, realised batch random effect
]


# ── Confounder distribution parameters (fitted to Olympus datasets) ───────────
# These are module level constants so they can be inspected and overridden
# in tests without re-running the GP fits.

# landscape_smoothness ~ LogNormal(LS_MU, LS_SIGMA)
LS_MU:    float = -0.4191
LS_SIGMA: float =  0.4819

# noise_level ~ LogNormal(NL_MU, NL_SIGMA)
NL_MU:    float = -2.5944
NL_SIGMA: float =  0.9931

# dimensionality ~ Categorical
DIM_VALUES: np.ndarray = np.array([3, 4, 6])
DIM_PROBS:  np.ndarray = np.array([0.25, 0.50, 0.25])

# initial_design_quality ~ Beta(2, 2)  (symmetric, peaks at 0.5, bounded [0,1])
# Not fitted to Olympus (it is a simulation design variable, not a dataset
# property).  Beta(2,2) gives a realistic spread around 0.5.
IDQ_ALPHA: float = 2.0
IDQ_BETA:  float = 2.0


def _read_edge_weights(G) -> dict[str, float]:
    """
    Extract the scalar edge weights from the UC1 DAG that the outcome
    equation needs.  Raises if any expected edge is missing or has
    weight=None (which would indicate a modelling gap).
    """
    required = {
        ("strategy",              "norm_best_outcome"),
        ("landscape_smoothness",  "norm_best_outcome"),
        ("budget",                "norm_best_outcome"),
        ("noise_level",           "norm_best_outcome"),
        ("dimensionality",        "norm_best_outcome"),
        ("initial_design_quality","norm_best_outcome"),
        ("landscape_smoothness",  "strategy"),
        ("budget",                "strategy"),
        ("noise_level",           "strategy"),
        ("dimensionality",        "strategy"),
    }
    weights: dict[str, float] = {}
    for u, v in required:
        if not G.has_edge(u, v):
            raise ValueError(f"graphs.py is missing edge {u} → {v}")
        w = G[u][v].get("weight")
        if w is None:
            raise ValueError(
                f"Edge {u} → {v} has weight=None; "
                "only batch_id → norm_best_outcome may be None."
            )
        weights[f"{u}__{v}"] = float(w)
    return weights


def run_campaigns(
    n_campaigns: int = 500,
    budget: int | tuple[int, int] = (20, 100),
    n_batches: int = 10,
    batch_sigma: float = 0.05,
    outcome_noise_sigma: float = 0.03,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic campaign dataset encoding the UC1 causal DAG.

    Parameters
    ----------
    n_campaigns : int
        Number of campaigns to simulate.  500 gives DoWhy enough power
        to recover the ATE reliably; 200 is the practical minimum.

    budget : int or (int, int)
        If int, every campaign has this budget.
        If (low, high), budget is drawn uniformly from [low, high].

    n_batches : int
        Number of distinct experimental batches.  Each batch gets a
        random intercept drawn from N(0, batch_sigma).

    batch_sigma : float
        Standard deviation of the batch random intercept.
        Default 0.05 ≈ 5% of the normalised outcome scale.
        Sweep this in UC7 (robustness) to test sensitivity.

    outcome_noise_sigma : float
        Residual noise added to norm_best_outcome after all causal
        effects are applied.  Represents within-campaign irreducible
        variability not captured by the DAG variables.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per campaign. Columns defined in UC1_COLUMNS.
        norm_best_outcome is clipped to [0, 1] after generation.

    Notes
    -----
    The data-generating process is:

        # Step 1 - draw confounders
        landscape_smoothness ~ LogNormal(LS_MU, LS_SIGMA)
        noise_level          ~ LogNormal(NL_MU, NL_SIGMA)
        dimensionality       ~ Categorical(DIM_VALUES, DIM_PROBS)
        budget               ~ Uniform(low, high)  [or fixed]

        # Step 2 - draw non-confounder and noise variable
        initial_design_quality ~ Beta(IDQ_ALPHA, IDQ_BETA)
        batch_id               ~ Categorical(0..n_batches-1)

        # Step 3 - strategy assignment (logistic selection model)
        # Confounders are standardised before entering the logit so that
        # the log-odds coefficients from graphs.py are on a common scale.
        logit_p = intercept
                  + w(ls→s)  * landscape_smoothness_std
                  + w(b→s)   * budget_std
                  + w(nl→s)  * noise_level_std
                  + w(d→s)   * dimensionality_std
        p_bo    = sigmoid(logit_p)
        strategy ~ Bernoulli(p_bo)

        # Step 4 - outcome (linear structural equation)
        # Confounders and treatment are standardised before entering so
        # that the edge weights from graphs.py are effect sizes on the
        # normalised outcome scale.
        norm_best_outcome =
              intercept_outcome
            + w(s→y)   * strategy                    # ATE = +0.10
            + w(ls→y)  * landscape_smoothness_std
            + w(b→y)   * budget_std
            + w(nl→y)  * noise_level_std
            + w(d→y)   * dimensionality_std
            + w(idq→y) * initial_design_quality_std
            + batch_intercept[batch_id]
            + N(0, outcome_noise_sigma)

        clipped to [0, 1]

    The intercepts are chosen so that E[norm_best_outcome] ≈ 0.65 and
    E[p_bo] ≈ 0.50, matching the rough empirical base rates.
    """
    rng = np.random.default_rng(seed)

    # ── Load DAG and read edge weights ────────────────────────────────────────
    G = uc1_dag()
    w = _read_edge_weights(G)

    # ── Step 1: Draw confounders ──────────────────────────────────────────────
    ls = rng.lognormal(LS_MU, LS_SIGMA, n_campaigns)          # landscape_smoothness
    nl = rng.lognormal(NL_MU, NL_SIGMA, n_campaigns)          # noise_level
    dim = rng.choice(DIM_VALUES, size=n_campaigns, p=DIM_PROBS).astype(float)

    if isinstance(budget, tuple):
        low, high = budget
        bud = rng.integers(low, high + 1, size=n_campaigns).astype(float)
    else:
        bud = np.full(n_campaigns, float(budget))

    # ── Step 2: Draw non-confounder and noise variable ────────────────────────
    idq = rng.beta(IDQ_ALPHA, IDQ_BETA, n_campaigns)           # initial_design_quality
    batch_ids = rng.integers(0, n_batches, size=n_campaigns)   # batch_id

    # Batch random intercepts - 1 per batch, shared across campaigns in
    # the same batch.  This is the random effect for batch_id → outcome.
    batch_effects = rng.normal(0.0, batch_sigma, size=n_batches)
    batch_intercept = batch_effects[batch_ids]

    # ── Standardise continuous confounders ────────────────────────────────────
    # Edge weights in graphs.py are calibrated on a standardised scale so
    # that a 1 unit change corresponds to 1 standard deviation of the
    # confounder. Standardise using the sample mean/std of this draw.
    def _std(x: np.ndarray) -> np.ndarray:
        s = x.std()
        return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)

    ls_z   = _std(ls)
    nl_z   = _std(nl)
    dim_z  = _std(dim)
    bud_z  = _std(bud)
    idq_z  = _std(idq)

    # ── Step 3: Strategy assignment (logistic selection model) ────────────────
    # Intercept chosen so that E[p_bo] ≈ 0.50 at mean confounder values
    # (which are zero after standardisation).
    logit_p = (
        0.0                                      # intercept → P(BO)=0.5 at mean
        + w["landscape_smoothness__strategy"] * ls_z
        + w["budget__strategy"]               * bud_z
        + w["noise_level__strategy"]          * nl_z
        + w["dimensionality__strategy"]       * dim_z
    )
    p_bo = 1.0 / (1.0 + np.exp(-logit_p))       # sigmoid
    strategy = rng.binomial(1, p_bo).astype(float)

    # ── Step 4: Outcome (linear structural equation) ──────────────────────────
    # Intercept chosen so that E[norm_best_outcome] ≈ 0.65 at mean values
    # with strategy=0 (random search baseline).
    intercept_outcome = 0.65

    norm_best = (
        intercept_outcome
        + w["strategy__norm_best_outcome"]               * strategy
        + w["landscape_smoothness__norm_best_outcome"]   * ls_z
        + w["budget__norm_best_outcome"]                 * bud_z
        + w["noise_level__norm_best_outcome"]            * nl_z
        + w["dimensionality__norm_best_outcome"]         * dim_z
        + w["initial_design_quality__norm_best_outcome"] * idq_z
        + batch_intercept
        + rng.normal(0.0, outcome_noise_sigma, n_campaigns)
    )
    norm_best = np.clip(norm_best, 0.0, 1.0)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "campaign_id":            np.arange(n_campaigns),
        "strategy":               strategy.astype(int),
        "norm_best_outcome":      norm_best,
        "landscape_smoothness":   ls,
        "budget":                 bud.astype(int),
        "noise_level":            nl,
        "dimensionality":         dim.astype(int),
        "initial_design_quality": idq,
        "batch_id":               batch_ids,
        "p_bo":                   p_bo,
        "batch_intercept":        batch_intercept,
    })

    # Enforce column order and schema
    assert list(df.columns) == UC1_COLUMNS, (
        f"Column mismatch.\nExpected: {UC1_COLUMNS}\nGot: {list(df.columns)}"
    )

    return df


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_campaigns(n_campaigns=1000, seed=42)

    print("Schema:")
    print(df.dtypes.to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nDescriptive statistics:")
    print(df[["strategy", "norm_best_outcome", "landscape_smoothness",
              "noise_level", "dimensionality", "budget",
              "initial_design_quality"]].describe().round(3).to_string())

    bo_rate = df["strategy"].mean()
    print(f"\nBO assignment rate:          {bo_rate:.3f}  (target ≈ 0.50)")

    mean_outcome = df["norm_best_outcome"].mean()
    print(f"Mean norm_best_outcome:      {mean_outcome:.3f}  (target ≈ 0.65)")

    # Naive ATE (unadjusted) — should be inflated above +0.10
    naive_ate = (df.loc[df.strategy==1, "norm_best_outcome"].mean()
               - df.loc[df.strategy==0, "norm_best_outcome"].mean())
    print(f"Naive ATE (unadjusted):      {naive_ate:+.4f}  (should be > +0.10)")
    print(f"True ATE (graph weight):     +0.1000")
    print(f"Confounding bias:            {naive_ate - 0.10:+.4f}  (should be > 0)")

    print(f"\nBO rate by dimensionality:")
    print(df.groupby("dimensionality")["strategy"].mean().round(3).to_string())

    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
