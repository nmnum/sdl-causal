"""
simulator.py — UC1 / UC2 campaign simulator for the Self-Driving Lab project.

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
     is preferentially assigned to easier problems — exactly the selection
     bias that DoWhy's backdoor adjustment must correct.

  3. norm_best_outcome is a linear combination of the causal effects encoded
     in the graph edge weights, plus a batch random intercept and residual
     noise.  The strategy coefficient (+0.10) is the ground truth ATE that
     DoWhy must recover.

  4. initial_design_quality is drawn independently of strategy (it is a
     non-confounder by simulation design — see uc1_dag_reasoning.md §2).

  5. batch_id is a categorical noise variable.  Its effect is a random
     intercept drawn from N(0, batch_sigma) per batch.

Empirical basis for distribution parameters
--------------------------------------------
Confounder distributions are anchored to GP length-scales and noise levels
estimated from the four Olympus datasets (one scalar per dataset — 4 anchor
points total).  LogNormal parameters are moment-matched to these 4 values;
they are NOT statistically fitted distributions in the usual sense.  The
distributions are consistent with the empirical anchors but should be
treated as plausible simulation priors, not validated population models.

    Dataset      dim   length_scale   noise_std
    HPLC           6         0.3095      0.1634
    SnAr           4         0.8629      0.0153
    Suzuki         4         1.1211      0.1836
    Fullerenes     3         0.6247      0.0678

    landscape_smoothness ~ LogNormal(mu=-0.4191, sigma=0.4819)
      [anchored to 4 values; sigma driven largely by HPLC vs. Suzuki spread]
    noise_level          ~ LogNormal(mu=-2.5944, sigma=0.9931)
      [wide sigma reflects SnAr outlier (0.015) vs. experimental datasets (0.16-0.18)]
    dimensionality       ~ Categorical({3: 0.25, 4: 0.50, 6: 0.25})
      [empirical PMF of the 4 datasets]

Reference: uc1_dag_reasoning.md §6a, uc1_bo_vs_random_benchmark.png
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

# graphs.py is the single source of truth for edge weights.
# Import the DAG and read weights from it rather than hardcoding them here.
from src.graphs import (
    uc1_dag, uc2_dag,
    get_adjustment_set, get_uc2_adjustment_set,
    UC1_NODE_META,
    UC2_SNR_THRESHOLD, UC2_SNR_SCALE, UC2_BO_FLOOR, UC2_BO_CEILING,
)


# ── Fixed output schemas ──────────────────────────────────────────────────────
# UC1_COLUMNS: guaranteed columns for use_case="UC1"
# UC2_COLUMNS: extends UC1 with snr and snr_modulation

UC1_COLUMNS: list[str] = [
    "campaign_id",            # int, unique campaign identifier
    "strategy",               # int, 0=random search, 1=BO
    "norm_best_outcome",      # float in [0, 1], best found / true optimum (clipped)
    "latent_outcome",         # float, norm_best_outcome BEFORE clipping to [0,1]
                              #   Use this column for causal estimation (DoWhy).
                              #   norm_best_outcome is for reporting only.
                              #   Clipping disproportionately censors BO campaigns
                              #   (72% of clipped rows are BO vs 50% overall),
                              #   causing downward bias in the linear estimator.
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

UC2_COLUMNS: list[str] = UC1_COLUMNS + [
    "snr",                    # float > 0, landscape_smoothness / noise_level
    "snr_modulation",         # float in [UC2_BO_FLOOR, UC2_BO_CEILING],
                              #   f(snr): the realised BO advantage multiplier
                              #   for this campaign.  Ground truth ATE = E[f(snr)].
]


# ── Confounder distribution parameters (anchored to Olympus datasets) ────────
# Moment-matched to 4 scalar anchor points (one per Olympus dataset).
# These are module-level constants so they can be inspected and overridden
# in tests without re-running the GP fits.
# WARNING: LogNormal parameters are derived from only 4 data points.
# Treat as simulation priors, not statistically validated distributions.

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


def snr_modulation_fn(
    snr: np.ndarray,
    tau: Optional[float] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """
    Threshold + smooth-decay moderation function f(snr; τ, scale).

    Maps SNR → BO advantage multiplier in [UC2_BO_FLOOR, UC2_BO_CEILING].

    Functional form
    ---------------
    f(snr; τ, scale) = UC2_BO_FLOOR
                       + (UC2_BO_CEILING - UC2_BO_FLOOR)
                       × sigmoid((snr - τ) / scale)

    Default: τ = UC2_SNR_THRESHOLD = 2.0, scale = UC2_SNR_SCALE = 1.0.

    Properties (for any τ > 0, scale > 0)
    --------------------------------------
    - f(snr → 0)   → UC2_BO_FLOOR    (asymptotic; never reached in sim)
    - f(snr = τ)   = midpoint = (UC2_BO_FLOOR + UC2_BO_CEILING) / 2 = +0.025
    - f(snr → ∞)   → UC2_BO_CEILING  (asymptotic; recovers UC1 ATE)
    - Actual simulated range at τ=2.0: f ∈ [−0.032, +0.100] (SNR ∈ [0.14, 305])
    - Monotone non-decreasing in SNR for any τ, scale > 0
    - Dimensionless: τ transfers across datasets without rescaling

    GP-theoretic basis
    ------------------
    When noise variance σ²_n ≥ signal variance σ²_f (SNR ≤ 1), the GP
    posterior variance barely shrinks from the prior, and the acquisition
    function selects points near-uniformly (≈ random search).  τ = 2.0
    gives a conservative margin: at SNR = 2.0 the GP is still informative
    but the BO advantage is halved.  See uc2_dag_reasoning.md §4.

    The τ sensitivity analysis (uc2_tau_sensitivity.ipynb) shows that the
    qualitative design rule — BO underperforms RS below threshold, outperforms
    above — holds for τ ∈ {1.0, 1.5, 2.0, 2.5, 3.0}.

    Parameters
    ----------
    snr : np.ndarray
        Per-campaign SNR values (landscape_smoothness / noise_level).
    tau : float or None
        SNR threshold.  If None, uses UC2_SNR_THRESHOLD (= 2.0).
        Pass a non-default value for sensitivity analysis only.
    scale : float or None
        Sigmoid steepness.  If None, uses UC2_SNR_SCALE (= 1.0).

    Returns
    -------
    np.ndarray
        Per-campaign BO advantage multiplier, same shape as snr.
    """
    _tau   = UC2_SNR_THRESHOLD if tau   is None else float(tau)
    _scale = UC2_SNR_SCALE     if scale is None else float(scale)
    sig = 1.0 / (1.0 + np.exp(-(snr - _tau) / _scale))
    return UC2_BO_FLOOR + (UC2_BO_CEILING - UC2_BO_FLOOR) * sig


def _read_edge_weights(G) -> dict[str, float]:
    """
    Extract the scalar edge weights from a UC1 or UC2 DAG.

    Reads all edges that the outcome and selection equations need.
    UC2-specific edges (snr → strategy, snr → norm_best_outcome) are
    included only if present in the graph.

    Raises if any required edge is missing or has weight=None.
    """
    # UC1 required edges (always present)
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
    # UC2 additional edges (present only in UC2 DAG)
    optional = {
        ("snr", "strategy"),
        ("snr", "norm_best_outcome"),
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

    for u, v in optional:
        if G.has_edge(u, v):
            w = G[u][v].get("weight")
            if w is None:
                raise ValueError(f"Optional edge {u} → {v} has weight=None.")
            weights[f"{u}__{v}"] = float(w)

    # ── Ground truth ATE guard ────────────────────────────────────────────────
    # The strategy → norm_best_outcome weight is the documented ceiling ATE
    # (+0.10).  In UC1 this is the fixed ATE; in UC2 it is the noise-free
    # limit of f(snr).  Either way it must be +0.10.
    _ate = weights["strategy__norm_best_outcome"]
    if abs(_ate - 0.10) > 1e-9:
        raise ValueError(
            f"graphs.py strategy → norm_best_outcome weight is {_ate:+.4f}, "
            f"expected +0.10 (the UC1/UC2 ground truth ATE ceiling). "
            f"If you intentionally changed the ATE, update this guard in "
            f"_read_edge_weights() and the design rule documentation."
        )

    return weights


def run_campaigns(
    n_campaigns: int = 500,
    budget: int | tuple[int, int] = (20, 100),
    n_batches: int = 10,
    batch_sigma: float = 0.05,
    outcome_noise_sigma: float = 0.03,
    seed: Optional[int] = None,
    return_curves: bool = False,
    use_case: str = "UC1",
    tau: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic campaign dataset encoding the UC1 or UC2 causal DAG.

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

    return_curves : bool
        If True, raise NotImplementedError.  Reserved for UC3 (time-varying
        strategy switches), which requires step-by-step BO replay.

    use_case : {"UC1", "UC2"}
        Which causal DAG / structural equation to use.

        "UC1" (default):
            strategy effect is a fixed scalar (+0.10).
            Returns UC1_COLUMNS.

        "UC2":
            strategy effect is modulated by SNR via snr_modulation_fn().
            Adds 'snr' and 'snr_modulation' columns.
            Returns UC2_COLUMNS.
            The ground truth ATE is E[f(snr; tau)] (depends on the SNR
            distribution and tau), NOT the fixed +0.10 of UC1.
            Use latent_outcome for causal estimation (same reason as UC1:
            ceiling clipping censors BO campaigns disproportionately).

    tau : float or None
        SNR threshold passed to snr_modulation_fn().  Only used when
        use_case="UC2".  If None (default), uses UC2_SNR_THRESHOLD = 2.0.
        Pass a non-default value for τ sensitivity analysis only — the
        canonical value for all other analyses is 2.0.
        Ignored when use_case="UC1".

    Returns
    -------
    pd.DataFrame
        One row per campaign.
        UC1: columns defined in UC1_COLUMNS.
        UC2: columns defined in UC2_COLUMNS (superset of UC1_COLUMNS).
        norm_best_outcome is clipped to [0, 1] after generation.

    Notes
    -----
    UC1 structural equation:

        norm_best_outcome =
              0.65
            + 0.10 * strategy                        # fixed ATE
            + w(ls→y)  * landscape_smoothness_std
            + w(b→y)   * budget_std
            + w(nl→y)  * noise_level_std
            + w(d→y)   * dimensionality_std
            + w(idq→y) * initial_design_quality_std
            + batch_intercept[batch_id]
            + N(0, outcome_noise_sigma)

    UC2 structural equation (replaces the strategy term):

        strategy_effect = snr_modulation_fn(snr) * strategy
                        = f(snr) * strategy
        where f(snr) ∈ [UC2_BO_FLOOR, UC2_BO_CEILING] = [-0.05, +0.10]

        snr = landscape_smoothness / noise_level  (per campaign)

        The SNR also enters the selection model (snr → strategy) and the
        outcome directly (snr → norm_best_outcome), making it a confounder
        that must be in the adjustment set.

    The intercepts are chosen so that E[norm_best_outcome] ≈ 0.65 and
    E[p_bo] ≈ 0.50, matching the rough empirical base rates.
    """
    if use_case not in ("UC1", "UC2"):
        raise ValueError(f"use_case must be 'UC1' or 'UC2', got {use_case!r}")

    if return_curves:
        raise NotImplementedError(
            "return_curves=True is reserved for UC3 (time-varying strategy "
            "switches). Implement step-by-step BO replay in UC3 before "
            "enabling this."
        )

    rng = np.random.default_rng(seed)

    # ── Load DAG and read edge weights ────────────────────────────────────────
    G = uc2_dag() if use_case == "UC2" else uc1_dag()
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

    # Batch random intercepts — one per batch, shared across campaigns in
    # the same batch.  This is the random effect for batch_id → outcome.
    batch_effects = rng.normal(0.0, batch_sigma, size=n_batches)
    batch_intercept = batch_effects[batch_ids]

    # ── Standardise continuous confounders ────────────────────────────────────
    # Edge weights in graphs.py are calibrated on a standardised scale so
    # that a one-unit change corresponds to one standard deviation of the
    # confounder.  Standardise using the sample mean/std of this draw.
    def _std(x: np.ndarray) -> np.ndarray:
        s = x.std()
        return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)

    ls_z   = _std(ls)
    nl_z   = _std(nl)
    dim_z  = _std(dim)
    bud_z  = _std(bud)
    idq_z  = _std(idq)

    # ── Step 2b (UC2 only): Compute SNR ──────────────────────────────────────
    # SNR = landscape_smoothness / noise_level.
    # Clipped to a small positive floor to avoid division-by-zero when
    # noise_level is near zero (rare but possible with LogNormal draws).
    snr = ls / np.maximum(nl, 1e-6)
    snr_z = _std(snr)

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
    if use_case == "UC2":
        # SNR additionally confounds strategy choice in UC2
        logit_p += w["snr__strategy"] * snr_z

    p_bo = 1.0 / (1.0 + np.exp(-logit_p))       # sigmoid
    strategy = rng.binomial(1, p_bo).astype(float)

    # ── Step 4: Outcome (linear structural equation) ──────────────────────────
    # Intercept chosen so that E[norm_best_outcome] ≈ 0.65 at mean values
    # with strategy=0 (random search baseline).
    intercept_outcome = 0.65

    if use_case == "UC1":
        # Fixed strategy effect: ATE = +0.10 (ground truth)
        strategy_contribution = w["strategy__norm_best_outcome"] * strategy
        snr_modulation = np.full(n_campaigns, w["strategy__norm_best_outcome"])
    else:
        # UC2: strategy effect is modulated by SNR
        # f(snr; tau) ∈ [UC2_BO_FLOOR, UC2_BO_CEILING] = [-0.05, +0.10]
        # Ground truth ATE = E[f(snr; tau)] (depends on SNR distribution and tau)
        snr_modulation = snr_modulation_fn(snr, tau=tau)
        strategy_contribution = snr_modulation * strategy

    latent_outcome = (
        intercept_outcome
        + strategy_contribution
        + w["landscape_smoothness__norm_best_outcome"]   * ls_z
        + w["budget__norm_best_outcome"]                 * bud_z
        + w["noise_level__norm_best_outcome"]            * nl_z
        + w["dimensionality__norm_best_outcome"]         * dim_z
        + w["initial_design_quality__norm_best_outcome"] * idq_z
        + batch_intercept
        + rng.normal(0.0, outcome_noise_sigma, n_campaigns)
    )
    if use_case == "UC2":
        # SNR has a direct effect on outcome (separate from moderation)
        latent_outcome += w["snr__norm_best_outcome"] * snr_z

    # norm_best_outcome is the clipped version for reporting.
    # latent_outcome is the pre-clip value for causal estimation.
    #
    # Why latent_outcome is needed for DoWhy:
    #   Ceiling clipping (norm_best ≥ 1.0) disproportionately censors BO
    #   campaigns (~72% of clipped rows are BO vs 50% overall), because BO
    #   is preferentially assigned to easier problems that are more likely to
    #   hit the ceiling.  This introduces a Tobit-style censoring bias that
    #   causes OLS on norm_best_outcome to underestimate the ATE.
    norm_best = np.clip(latent_outcome, 0.0, 1.0)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "campaign_id":            np.arange(n_campaigns),
        "strategy":               strategy.astype(int),
        "norm_best_outcome":      norm_best,
        "latent_outcome":         latent_outcome,
        "landscape_smoothness":   ls,
        "budget":                 bud.astype(int),
        "noise_level":            nl,
        "dimensionality":         dim.astype(int),
        "initial_design_quality": idq,
        "batch_id":               batch_ids,
        "p_bo":                   p_bo,
        "batch_intercept":        batch_intercept,
    })

    if use_case == "UC2":
        df["snr"]            = snr
        df["snr_modulation"] = snr_modulation

    # Enforce column order and schema
    expected_cols = UC2_COLUMNS if use_case == "UC2" else UC1_COLUMNS
    assert list(df.columns) == expected_cols, (
        f"Column mismatch.\nExpected: {expected_cols}\nGot: {list(df.columns)}"
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

    # Naive ATE on latent_outcome (pre-clip) — should be inflated above +0.10
    naive_ate = (df.loc[df.strategy==1, "latent_outcome"].mean()
               - df.loc[df.strategy==0, "latent_outcome"].mean())
    print(f"Naive ATE on latent_outcome: {naive_ate:+.4f}  (should be > +0.10)")
    print(f"True ATE (graph weight):     +0.1000")
    print(f"Confounding bias:            {naive_ate - 0.10:+.4f}  (should be > 0)")

    print(f"\nBO rate by dimensionality:")
    print(df.groupby("dimensionality")["strategy"].mean().round(3).to_string())

    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
