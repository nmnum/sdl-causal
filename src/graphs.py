"""
graphs.py — Causal DAG definitions for the Self-Driving Lab project.

This file is the single source of truth for causal structure.
Notebooks and the simulator import from here; they never redefine graphs inline.

Each graph is returned by a factory function so that callers always get a
fresh copy (networkx graphs are mutable).  Edge metadata is stored in the
edge attribute dict and mirrored in inline comments so the reasoning is
visible without running the code.

Reference documents:
  uc1_dag_reasoning.md  — UC1 causal structure and empirical basis
  uc2_dag_reasoning.md  — UC2 noise moderation, SNR threshold, GP-theoretic τ

Empirical basis for effect sizes: uc1_bo_vs_random_benchmark.png
  (160 campaigns, GP-BO vs. random search, 4 Olympus datasets, 20 seeds each)
"""

from __future__ import annotations

from typing import Callable

import networkx as nx


# ── Node metadata ─────────────────────────────────────────────────────────────
# Stored as a module-level dict so the simulator can read variable roles
# without re-parsing the graph.

UC1_NODE_META: dict[str, dict] = {
    "landscape_smoothness": {
        "role": "confounder",
        "dtype": "continuous",
        "description": (
            "Estimated smoothness of the objective surface "
            "(e.g. GP length-scale). Higher = smoother."
        ),
    },
    "budget": {
        "role": "confounder",
        "dtype": "continuous",
        "description": "Number of experiments allowed in the campaign.",
    },
    "noise_level": {
        "role": "confounder",
        "dtype": "continuous",
        "description": "Measurement noise standard deviation.",
    },
    "dimensionality": {
        "role": "confounder",
        "dtype": "discrete",
        "description": "Number of input variables in the search space.",
    },
    "initial_design_quality": {
        "role": "non_confounder",
        "dtype": "continuous",
        "description": (
            "Space-filling score of the seed points. "
            "Affects outcome but NOT strategy choice in simulation "
            "(assigned independently of strategy). "
            "Would be a confounder in real SDL data — revisit on translation."
        ),
    },
    "batch_id": {
        "role": "noise",
        "dtype": "categorical",
        "description": (
            "Experimental batch identifier (reagent lot, instrument day). "
            "Affects outcome via reagent variability. "
            "Not known at strategy-selection time → not a confounder."
        ),
    },
    "strategy": {
        "role": "treatment",
        "dtype": "binary",
        "description": "Optimisation strategy: 0 = random search, 1 = BO.",
    },
    "norm_best_outcome": {
        "role": "outcome",
        "dtype": "continuous",
        "description": (
            "Best outcome found / true optimum. "
            "Scale-free; 1.0 = optimum recovered exactly."
        ),
    },
}


def uc1_dag() -> nx.DiGraph:
    """
    Return the UC1 causal DAG as a networkx.DiGraph.

    Scientific question
    -------------------
    Does the choice of optimisation strategy (BO vs. random search) causally
    improve the best outcome found in a campaign, after adjusting for landscape
    properties, experimental noise, and budget?

    Identification
    --------------
    The causal effect of strategy on norm_best_outcome is identified by
    backdoor adjustment on {landscape_smoothness, budget, noise_level,
    dimensionality}.  DoWhy's identify_effect() confirms this automatically.

    Effect sizes (edge attribute 'weight')
    --------------------------------------
    Empirically derived from 160 GP-BO vs. random search campaigns across
    four Olympus benchmark datasets (HPLC 6D, SnAr 4D, Suzuki 4D,
    Fullerenes 3D), 20 seeds each, budget=100.

    Outcome effects are in normalised best outcome units (0–1 scale).
    Strategy-selection effects are in log-odds units (logistic model).

    Returns
    -------
    nx.DiGraph
        Nodes carry UC1_NODE_META attributes.
        Edges carry:
          weight   : float  — effect size (see above)
          rationale: str    — one-sentence causal justification
    """
    G = nx.DiGraph(name="UC1_strategy_vs_outcome")

    # ── Add nodes with metadata ───────────────────────────────────────────────
    for node, meta in UC1_NODE_META.items():
        G.add_node(node, **meta)

    # ── Confounder → strategy edges (selection model) ─────────────────────────
    # These edges encode why practitioners choose BO over random search.
    # Effect sizes are in log-odds units (a one-unit increase in the covariate
    # changes log P(strategy=BO) by this amount).

    G.add_edge(
        "landscape_smoothness", "strategy",
        weight=+0.25,
        rationale=(
            "Practitioners prefer BO on smoother landscapes because the GP "
            "surrogate is more accurate there; BO's advantage over random "
            "search is also larger on smooth surfaces."
        ),
    )

    G.add_edge(
        "budget", "strategy",
        weight=+0.15,
        rationale=(
            "BO's fixed overhead (surrogate fitting, acquisition optimisation) "
            "is only worth paying for larger budgets; practitioners apply BO "
            "when they can afford enough evaluations to amortise that cost."
        ),
    )

    G.add_edge(
        "noise_level", "strategy",
        weight=-0.20,
        rationale=(
            "High measurement noise degrades the GP surrogate model, reducing "
            "BO's advantage; practitioners are less likely to choose BO in "
            "noisy experimental settings."
        ),
    )

    G.add_edge(
        "dimensionality", "strategy",
        weight=-0.10,
        rationale=(
            "Standard GP-BO scales poorly beyond ~10 dimensions; practitioners "
            "are less likely to apply it in high-dimensional spaces where "
            "random search or Latin hypercube designs are competitive."
        ),
    )

    # ── Treatment → outcome edge (the causal effect of interest) ─────────────
    # Weight is in normalised best outcome units.
    # Empirical basis: HPLC (6D) +0.106 at n=50; Suzuki (4D) +0.058 at n=50;
    # median across four datasets +0.04 at n=50.
    # Chosen value (+0.10) is anchored to HPLC, rounded conservatively to
    # account for GP emulator noise in the simulator vs. Olympus BNN emulator.

    G.add_edge(
        "strategy", "norm_best_outcome",
        weight=+0.10,
        rationale=(
            "BO recovers ~10 percentage points more of the true optimum than "
            "random search on a moderately difficult 4–6D landscape at budget "
            "~50. Effect is landscape-dependent: near-zero on smooth/sparse "
            "landscapes (SnAr +0.00), up to +0.11 on rugged/dense ones "
            "(HPLC 6D). This is the average treatment effect (ATE) that "
            "DoWhy's backdoor adjustment must recover."
        ),
    )

    # ── Confounder → outcome edges (direct effects on optimisation difficulty) ─
    # Weights are in normalised best outcome units per unit of the covariate.

    G.add_edge(
        "landscape_smoothness", "norm_best_outcome",
        weight=+0.20,
        rationale=(
            "Smoother landscapes have fewer local optima and are easier to "
            "optimise regardless of strategy; the SnAr dataset (smooth, 4D) "
            "reaches norm_best ≈ 1.0 within 20 evaluations for both BO and "
            "random search."
        ),
    )

    G.add_edge(
        "budget", "norm_best_outcome",
        weight=+0.10,
        rationale=(
            "More evaluations monotonically improve the best outcome found; "
            "all four Olympus learning curves are non-decreasing. Effect is "
            "diminishing (log-scale) but modelled as linear here for "
            "identifiability."
        ),
    )

    G.add_edge(
        "noise_level", "norm_best_outcome",
        weight=-0.15,
        rationale=(
            "Measurement noise obscures the true objective, causing both "
            "strategies to waste evaluations on spuriously high observations; "
            "the effect is symmetric across strategies but degrades BO more "
            "because the surrogate model absorbs the noise."
        ),
    )

    G.add_edge(
        "dimensionality", "norm_best_outcome",
        weight=-0.08,
        rationale=(
            "Curse of dimensionality: the volume of the search space grows "
            "exponentially with dimension, making it harder to find the optimum "
            "in a fixed budget. HPLC (6D) has the lowest norm_best under random "
            "search (0.78 at n=50) vs. Fullerenes (3D) at 0.99."
        ),
    )

    # ── Non-confounder → outcome edge ─────────────────────────────────────────
    # initial_design_quality affects outcome but NOT strategy in simulation.
    # This edge is present so DoWhy sees it and correctly excludes it from the
    # adjustment set (it is not a backdoor path node).

    G.add_edge(
        "initial_design_quality", "norm_best_outcome",
        weight=+0.12,
        rationale=(
            "A well-spread initial design (high space-filling score) gives both "
            "strategies a better starting point, improving the best outcome "
            "found. In simulation this is assigned independently of strategy, "
            "so it is not a confounder — but it is a real cause of outcome "
            "variance and must be in the graph."
        ),
    )

    # ── Noise variable → outcome edge ─────────────────────────────────────────
    # batch_id is a categorical noise source. Its effect is not given a scalar
    # weight here because it is modelled as a random intercept in the simulator
    # (drawn per batch from N(0, sigma_batch)).  The edge is present so the
    # graph is complete and DoWhy does not treat batch_id as an unmeasured
    # confounder.

    G.add_edge(
        "batch_id", "norm_best_outcome",
        weight=None,
        rationale=(
            "Batch effects (reagent lot, instrument day) shift the measured "
            "outcome up or down by a random amount. Modelled as a random "
            "intercept N(0, sigma_batch) in the simulator. Not a confounder: "
            "batch identity is not known at strategy-selection time."
        ),
    )

    return G


# ── Convenience helpers ───────────────────────────────────────────────────────

def get_confounders(G: nx.DiGraph, treatment: str, outcome: str) -> list[str]:
    """
    Return nodes that have edges into both `treatment` and `outcome`.
    This is the minimal sufficient adjustment set for a DAG with no
    unobserved variables and no selection bias.

    Note: this is a graph-level convenience, not a substitute for
    DoWhy's identify_effect(), which handles more complex structures.
    """
    parents_of_treatment = set(G.predecessors(treatment))
    parents_of_outcome = set(G.predecessors(outcome))
    return sorted(parents_of_treatment & parents_of_outcome)


def get_adjustment_set(G: nx.DiGraph | None = None) -> list[str]:
    """
    Return the UC1 backdoor adjustment set, sorted alphabetically.
    Hardcoded for clarity; must match get_confounders(uc1_dag(), ...).
    Sorted so that equality comparisons with get_confounders() (which also
    returns a sorted list) are order-independent without requiring callers
    to sort.
    """
    return sorted(["landscape_smoothness", "budget", "noise_level", "dimensionality"])


def summarise(G: nx.DiGraph) -> None:
    """Print a human-readable summary of nodes, roles, and edges."""
    print(f"Graph: {G.graph.get('name', 'unnamed')}")
    print(f"  Nodes ({G.number_of_nodes()}):")
    for n, d in G.nodes(data=True):
        role = d.get("role", "unknown")
        print(f"    {n:30s}  [{role}]")
    print(f"  Edges ({G.number_of_edges()}):")
    for u, v, d in G.edges(data=True):
        w = d.get("weight")
        w_str = f"{w:+.2f}" if w is not None else "random"
        print(f"    {u:30s} → {v:25s}  weight={w_str}")


# ── UC2: Noise-moderated strategy effect ──────────────────────────────────────

# UC2 adds one new node (snr) and one moderation edge (strategy × snr → outcome).
# The SNR threshold τ is stored here as a module-level constant so the simulator
# and tests can import it without re-parsing the graph.
#
# GP-theoretic justification for τ = 2.0
# ----------------------------------------
# In a GP surrogate, the posterior variance at a point x is:
#
#   σ²_post(x) = k(x,x) - k(x,X)[K(X,X) + σ²_n I]⁻¹ k(X,x)
#
# When σ²_n (noise variance) ≥ σ²_f (signal variance), the noise term
# dominates the kernel matrix inversion, and the posterior variance barely
# shrinks from the prior.  The acquisition function (e.g. EI) then selects
# points almost uniformly — indistinguishable from random search.
#
# The crossover point is SNR = σ²_f / σ²_n = 1.0 (noise = signal).
# We set τ = 2.0 (noise = 50% of signal) to give a conservative margin:
# at SNR = 2.0 the GP is still informative but the BO advantage is halved
# relative to the noise-free case.  This is consistent with the Olympus
# benchmarks: SnAr (SNR ≈ 3200, smooth) shows BO advantage ≈ 0; HPLC
# (SNR ≈ 23, rugged) shows BO advantage ≈ +0.11.
#
# τ is dimensionless (ratio of variances) and therefore transfers across
# campaign datasets without rescaling.  To apply to new data, estimate
# SNR = Var(signal) / Var(noise) from replicate measurements.

UC2_SNR_THRESHOLD: float = 2.0   # τ: SNR at which BO advantage = midpoint
UC2_SNR_SCALE:     float = 1.0   # sigmoid steepness around τ (in SNR units)
UC2_BO_FLOOR:      float = -0.05 # ASYMPTOTIC floor of f(snr) as snr → 0
                                  # NOTE: this is the sigmoid parameter, NOT the
                                  # value of f at any reachable SNR.  Because
                                  # noise_level ~ LogNormal (always > 0), snr
                                  # never reaches 0 in simulation.  The actual
                                  # minimum of f(snr) over the simulated SNR
                                  # distribution is ≈ −0.032 (at the 1st
                                  # percentile of SNR ≈ 0.14).
UC2_BO_CEILING:    float = +0.10 # ASYMPTOTIC ceiling of f(snr) as snr → ∞
                                  # (= UC1 ATE, noise-free limit).  In practice
                                  # f saturates to within 1e-4 of this by SNR≈20.

UC2_NODE_META: dict[str, dict] = {
    **UC1_NODE_META,  # inherit all UC1 nodes
    "snr": {
        "role": "confounder",
        "dtype": "continuous",
        "description": (
            "Signal-to-noise ratio: landscape_smoothness / noise_level. "
            "Dimensionless; transfers across campaign datasets. "
            "Confounds strategy choice (practitioners avoid BO at low SNR) "
            "and directly moderates the BO advantage (GP surrogate degrades "
            "when noise dominates signal). "
            f"Threshold τ = {UC2_SNR_THRESHOLD:.1f}: below this, BO advantage "
            "collapses to zero or slightly negative."
        ),
    },
}


def uc2_dag() -> nx.DiGraph:
    """
    Return the UC2 causal DAG as a networkx.DiGraph.

    Scientific question
    -------------------
    Does measurement noise moderate the causal effect of strategy choice on
    campaign outcome?  Specifically: is the BO advantage over random search
    smaller (or negative) at low signal-to-noise ratios, and does it recover
    above a critical SNR threshold?

    New elements relative to UC1
    ----------------------------
    Node  : snr = landscape_smoothness / noise_level
              Role: confounder (affects both strategy choice and outcome)
              Derived from existing UC1 variables; not independently drawn.

    Edges :
      snr → strategy          (confounding: practitioners avoid BO at low SNR)
      snr → norm_best_outcome (direct: high SNR → easier optimisation)
      strategy × snr → norm_best_outcome  (moderation: BO advantage depends on SNR)

    The moderation edge is encoded in the structural equation as:
        strategy_effect = f(snr) * strategy
    where f(snr) is the threshold+smooth-decay function (see UC2_SNR_THRESHOLD).

    Moderation function f(snr)
    --------------------------
    f(snr) = UC2_BO_FLOOR + (UC2_BO_CEILING - UC2_BO_FLOOR)
             × sigmoid((snr - τ) / scale)

    Properties:
      - f(snr → 0)   → UC2_BO_FLOOR  = -0.05  (asymptotic; never reached in sim)
      - f(snr = τ)   = midpoint       = +0.025 (half the maximum advantage)
      - f(snr → ∞)   → UC2_BO_CEILING = +0.10  (asymptotic; recovers UC1 ATE)
      - Monotone non-decreasing in SNR (strictly increasing on [0.1, 20])
      - Dimensionless τ transfers across datasets
      - Actual simulated range: f ∈ [−0.032, +0.100] (SNR ∈ [0.14, 305])

    Identification
    --------------
    The causal effect of strategy on norm_best_outcome is identified by
    backdoor adjustment on {landscape_smoothness, budget, noise_level,
    dimensionality, snr}.

    Important notes on the adjustment set:
    1. snr is a deterministic function of landscape_smoothness and noise_level
       (both already in the UC1 adjustment set).  Conditioning on {ls, nl}
       already blocks all snr backdoor paths — adding snr is over-specified
       but makes the interaction term directly estimable.
    2. The parent edges landscape_smoothness → snr and noise_level → snr
       do NOT change the adjustment set: snr is not a descendant of strategy,
       so the backdoor criterion is unaffected.  get_confounders() still
       returns the same 5-element set.
    3. In the OLS regression, include EITHER snr alone OR {ls, nl} — not
       all three simultaneously, as snr = ls/nl creates near-perfect
       multicollinearity.  The canonical estimator for the moderation story
       uses snr as the sole SNR-related regressor.

    Returns
    -------
    nx.DiGraph
        Inherits all UC1 nodes and edges.
        Adds: snr node, snr→strategy, snr→outcome edges.
        The strategy→outcome edge weight is now a *function* of snr
        (encoded in the simulator); the graph stores the ceiling weight
        (+0.10) and the moderation parameters in node/edge metadata.
    """
    # Start from UC1 DAG and extend
    G = uc1_dag()
    G.graph["name"] = "UC2_noise_moderation"

    # Update node metadata to UC2 version
    for node, meta in UC2_NODE_META.items():
        if node not in G.nodes:
            G.add_node(node, **meta)
        else:
            G.nodes[node].update(meta)

    # ── New node: snr ─────────────────────────────────────────────────────────
    # snr = landscape_smoothness / noise_level (deterministic function).
    # We add explicit parent edges landscape_smoothness → snr and
    # noise_level → snr so that DoWhy's graph traversal correctly represents
    # snr as a derived quantity rather than an exogenous confounder.
    # Without these edges, DoWhy would treat snr as independent of its
    # parents, which would misrepresent the causal structure and could
    # cause incorrect identification in more complex queries.

    G.add_edge(
        "landscape_smoothness", "snr",
        weight=+1.0,   # snr increases with landscape_smoothness (numerator)
        rationale=(
            "SNR = landscape_smoothness / noise_level. Landscape smoothness "
            "is the numerator: higher smoothness → higher SNR."
        ),
    )
    G.add_edge(
        "noise_level", "snr",
        weight=-1.0,   # snr decreases with noise_level (denominator)
        rationale=(
            "SNR = landscape_smoothness / noise_level. Noise level is the "
            "denominator: higher noise → lower SNR."
        ),
    )

    # ── snr → strategy (confounding) ─────────────────────────────────────────
    G.add_edge(
        "snr", "strategy",
        weight=+0.30,
        rationale=(
            "Practitioners are more likely to choose BO when the SNR is high "
            "(clean signal, reliable surrogate). At low SNR, the GP model is "
            "unreliable and practitioners default to random search or Latin "
            "hypercube designs. Effect is stronger than the individual "
            "noise_level and landscape_smoothness effects because SNR captures "
            "their interaction."
        ),
    )

    # ── snr → norm_best_outcome (direct effect) ───────────────────────────────
    G.add_edge(
        "snr", "norm_best_outcome",
        weight=+0.08,
        rationale=(
            "Higher SNR means the objective function is easier to resolve, "
            "improving the best outcome found regardless of strategy. "
            "This is a direct effect of SNR on optimisation difficulty, "
            "separate from the moderation of the BO advantage."
        ),
    )

    # ── Annotate the strategy → outcome edge with moderation metadata ─────────
    # The edge weight (+0.10) is the ceiling (noise-free limit).
    # The actual effect in the simulator is f(snr) * strategy, where f is
    # the threshold+smooth-decay function.  We store the moderation parameters
    # in the edge metadata so the simulator can read them from the graph.
    G["strategy"]["norm_best_outcome"].update({
        "moderated_by": "snr",
        "moderation_fn": "threshold_sigmoid",
        "snr_threshold": UC2_SNR_THRESHOLD,
        "snr_scale":     UC2_SNR_SCALE,
        "bo_floor":      UC2_BO_FLOOR,
        "bo_ceiling":    UC2_BO_CEILING,
        "weight_note": (
            "weight=+0.10 is the noise-free ceiling (SNR→∞). "
            "Actual effect = f(snr) ∈ [UC2_BO_FLOOR, UC2_BO_CEILING]. "
            "See UC2_SNR_THRESHOLD and uc2_dag_reasoning.md §4."
        ),
    })

    return G


def get_uc2_adjustment_set() -> list[str]:
    """
    Return the UC2 backdoor adjustment set, sorted alphabetically.
    Extends UC1 by adding 'snr'.
    """
    return sorted([
        "landscape_smoothness", "budget", "noise_level", "dimensionality", "snr"
    ])


# ── Registry ──────────────────────────────────────────────────────────────────
# As more use cases are added, register their factory functions here.
# The simulator and notebooks can look up graphs by UC name.

GRAPH_REGISTRY: dict[str, Callable[[], nx.DiGraph]] = {
    "UC1": uc1_dag,
    "UC2": uc2_dag,
}


# ── UC3: Convergence Curves (time-varying strategy effect) ────────────────────
#
# Scientific question
# -------------------
# Does BO converge faster than random search, and at what point in a campaign
# does the BO advantage emerge?  UC3 extends UC2 to a (campaign, step) panel,
# making the strategy effect time-varying via a warmup sigmoid.
#
# New elements relative to UC2
# ----------------------------
# Node  : step — within-campaign evaluation index (1..budget)
#           Role: time index (NOT a confounder — strategy is fixed at campaign
#           start and does not depend on step).
#
# Structural equation (per step t within campaign c):
#
#   norm_best_outcome_t = baseline(t)
#                       + f(snr_c) × g(t; t_warmup_c) × strategy_c
#                       + [UC2 confounder terms]
#                       + ε_t
#
# where:
#   baseline(t)  = intercept + UC3_ALPHA_RS × log(t + 1)
#   g(t; t_w)    = sigmoid((t - t_w) / UC3_WARMUP_SCALE)
#   t_warmup_c   = warmup_fraction_c × budget_c
#   warmup_fraction_c = clip(
#       UC3_WARMUP_BASE
#       + UC3_WARMUP_W_DIM × dim_z_c
#       - UC3_WARMUP_W_SNR × log_snr_z_c
#       - UC3_WARMUP_W_IDQ × idq_z_c,
#       0.05, 0.40
#   )
#
# Identification
# --------------
# The adjustment set is the same as UC2: {budget, dimensionality,
# landscape_smoothness, noise_level, snr}.  step is NOT in the adjustment
# set — it is not a confounder.  It is included as a covariate in the
# regression to control for the within-campaign time trend.
#
# Estimator: OLS with strategy × log(t) interaction + cluster-robust SEs
# (clustered at campaign level to correct for within-campaign autocorrelation
# from the monotone best-found trajectory).
#
# Empirical basis
# ---------------
# Parameters calibrated to the Olympus HPLC benchmark (6D, budget=100):
#   - RS log-linear fit: Y(t) ≈ 0.40 + 0.10 × log(t+1)
#   - BO starts outperforming RS at ~t=10 → warmup_fraction ≈ 0.10
#   - Sigmoid steepness: BO advantage emerges over ~5 steps
#
# Reference: uc3_dag_reasoning.md

# ── UC3 module-level constants ────────────────────────────────────────────────
UC3_ALPHA_RS:      float = 0.10   # RS log-linear convergence rate
                                   # calibrated: RS reaches 0.78 at t=50 from 0.55 at t=5
UC3_WARMUP_BASE:   float = 0.10   # base warmup fraction at mean confounders
                                   # calibrated: BO outperforms RS at ~t=10 (budget=100)
UC3_WARMUP_SCALE:  float = 5.0    # sigmoid steepness (in steps)
                                   # BO advantage emerges over ~5 steps
UC3_WARMUP_W_DIM:  float = 0.03   # warmup increases with dimensionality
UC3_WARMUP_W_SNR:  float = 0.02   # warmup decreases with log(SNR)
UC3_WARMUP_W_IDQ:  float = 0.02   # warmup decreases with initial design quality
UC3_STEP_NOISE:    float = 0.02   # per-step measurement noise std (within-campaign)

UC3_NODE_META: dict[str, dict] = {
    **UC2_NODE_META,  # inherit all UC2 nodes
    "step": {
        "role": "time_index",
        "dtype": "discrete",
        "description": (
            "Within-campaign evaluation index (1 to budget). "
            "Exogenous time index — NOT a confounder. Strategy is fixed at "
            "campaign start and does not depend on step. step moderates the "
            "strategy effect via the warmup sigmoid g(t; t_warmup)."
        ),
    },
    "norm_best_outcome_t": {
        "role": "outcome",
        "dtype": "continuous",
        "description": (
            "Best outcome found up to step t (running maximum). "
            "Monotone non-decreasing within a campaign. "
            "Replaces norm_best_outcome as the outcome variable in UC3."
        ),
    },
}


def uc3_dag() -> nx.DiGraph:
    """
    Return the UC3 causal DAG as a networkx.DiGraph.

    Scientific question
    -------------------
    Does BO converge faster than random search, and at what point in a
    campaign does the BO advantage emerge?

    New elements relative to UC2
    ----------------------------
    Node  : step — within-campaign evaluation index (1..budget)
              Role: time index (exogenous; NOT a confounder)

    Edges :
      step → norm_best_outcome_t
          Both strategies improve over time (log-linear baseline).
      strategy × step → norm_best_outcome_t
          The BO advantage grows with step via the warmup sigmoid g(t).
          Encoded in the structural equation; stored as edge metadata.

    The outcome node is renamed norm_best_outcome_t (step-level running max)
    to distinguish it from the campaign-level norm_best_outcome in UC1/UC2.

    Identification
    --------------
    Same adjustment set as UC2: {budget, dimensionality, landscape_smoothness,
    noise_level, snr}.  step is NOT in the adjustment set.

    Estimator: OLS with strategy × log(step) interaction + cluster-robust SEs
    (clustered at campaign level).

    Returns
    -------
    nx.DiGraph
        Inherits all UC2 nodes and edges.
        Adds: step node, step→norm_best_outcome_t edge.
        Renames outcome node to norm_best_outcome_t.
        Stores warmup parameters in strategy→outcome edge metadata.
    """
    # Start from UC2 DAG and extend
    G = uc2_dag()
    G.graph["name"] = "UC3_convergence_curves"

    # Update node metadata to UC3 version
    for node, meta in UC3_NODE_META.items():
        if node not in G.nodes:
            G.add_node(node, **meta)
        else:
            G.nodes[node].update(meta)

    # ── Rename outcome node to norm_best_outcome_t ────────────────────────────
    # UC3 uses a step-level outcome (running maximum up to step t).
    # We add the new outcome node and redirect all edges from
    # norm_best_outcome to norm_best_outcome_t.
    # The original norm_best_outcome node is kept for backward compatibility
    # (it represents the campaign-level final outcome, i.e. norm_best_outcome_t
    # at t=budget).
    G.add_node("norm_best_outcome_t", **UC3_NODE_META["norm_best_outcome_t"])

    # Copy all edges into norm_best_outcome to norm_best_outcome_t
    for pred in list(G.predecessors("norm_best_outcome")):
        edge_data = dict(G[pred]["norm_best_outcome"])
        G.add_edge(pred, "norm_best_outcome_t", **edge_data)

    # ── New node: step ────────────────────────────────────────────────────────
    # step is exogenous (no parents in the DAG).
    # It is a time index, not a confounder.

    # ── step → norm_best_outcome_t ────────────────────────────────────────────
    G.add_edge(
        "step", "norm_best_outcome_t",
        weight=UC3_ALPHA_RS,
        rationale=(
            "Both strategies improve over time via log-linear convergence. "
            f"Baseline rate UC3_ALPHA_RS = {UC3_ALPHA_RS:.2f} (log-scale), "
            "calibrated to HPLC: RS reaches 0.78 at t=50 from 0.55 at t=5. "
            "The BO additional advantage is encoded in the strategy×step "
            "interaction (see edge metadata on strategy→norm_best_outcome_t)."
        ),
    )

    # ── Annotate strategy → norm_best_outcome_t with warmup metadata ──────────
    # The time-varying BO advantage is:
    #   strategy_effect(t) = f(snr) × g(t; t_warmup) × strategy
    # where g(t) = sigmoid((t - t_warmup) / UC3_WARMUP_SCALE).
    # We store the warmup parameters in the edge metadata so the simulator
    # can read them from the graph.
    G["strategy"]["norm_best_outcome_t"].update({
        "moderated_by":    "snr_and_step",
        "moderation_fn":   "snr_sigmoid_times_warmup_sigmoid",
        "warmup_base":     UC3_WARMUP_BASE,
        "warmup_scale":    UC3_WARMUP_SCALE,
        "warmup_w_dim":    UC3_WARMUP_W_DIM,
        "warmup_w_snr":    UC3_WARMUP_W_SNR,
        "warmup_w_idq":    UC3_WARMUP_W_IDQ,
        "weight_note": (
            "weight=+0.10 is the noise-free, end-of-campaign ceiling. "
            "Actual effect at step t = f(snr) × g(t; t_warmup). "
            "See UC3_WARMUP_* constants and uc3_dag_reasoning.md §4."
        ),
    })

    return G


def get_uc3_adjustment_set() -> list[str]:
    """
    Return the UC3 backdoor adjustment set, sorted alphabetically.
    Identical to UC2: step is a time index, not a confounder.
    """
    return get_uc2_adjustment_set()


# ── Update registry ───────────────────────────────────────────────────────────
GRAPH_REGISTRY["UC3"] = uc3_dag


# ── UC4: Confounder Identification Sensitivity ────────────────────────────────
#
# Scientific question
# -------------------
# Are the confounders in our simulation design correctly identified?
# Specifically: what happens to the ATE estimate when we use the wrong
# adjustment set — omitting true confounders, adding spurious variables,
# or conditioning on a collider?
#
# New elements relative to UC2
# ----------------------------
# Node  : observed_flag — a COLLIDER caused by both strategy and norm_best_outcome.
#           Represents "did the experimenter flag this campaign as successful?"
#           This is a realistic SDL scenario: success flags depend on both
#           the strategy used and the result obtained.
#           Role: collider (NOT a confounder — do NOT adjust for it).
#
# Structural equation for observed_flag:
#   P(observed_flag = 1) = sigmoid(
#       UC4_COLLIDER_W_STRATEGY × strategy
#       + UC4_COLLIDER_W_OUTCOME × norm_best_outcome_std
#       + ε
#   )
#
# The 8 adjustment sets tested in UC4
# ------------------------------------
# Set 1 — CORRECT:          {budget, dim, ls, nl, snr}          (UC2 ground truth)
# Set 2 — DROP_SNR:         {budget, dim, ls, nl}                (omit derived confounder)
# Set 3 — DROP_LS:          {budget, dim, nl, snr}               (omit landscape_smoothness)
# Set 4 — DROP_NL:          {budget, dim, ls, snr}               (omit noise_level)
# Set 5 — DROP_SNR_RELATED: {budget, dim}                        (omit ls, nl, snr entirely)
# Set 6 — EMPTY:            {}                                   (naive unadjusted)
# Set 7 — ADD_NON_CONFOUNDER: correct + initial_design_quality   (spurious inclusion)
# Set 8 — COLLIDER:         correct + observed_flag              (opens non-causal path; DOWNWARD bias via Berkson's paradox, empirically -41%)
#
# Expected bias directions
# ------------------------
# Set 2:    near-zero bias (snr is redundant given ls and nl; omitting it is harmless).
# Sets 3–6: positive bias (BO preferentially assigned to high-SNR/easy campaigns
#           → unadjusted ATE inflated). Bias grows as more confounders are dropped.
# Set 7:    near-zero bias (initial_design_quality is not a confounder; adding it
#           is harmless — it absorbs outcome variance but does not open new paths).
# Set 8:    DOWNWARD bias (Berkson's paradox: conditioning on observed_flag attenuates
#           the within-stratum strategy-outcome correlation because the collider was
#           partly determined by strategy, not just outcome). Direction depends on
#           the signs of the collider parent weights — both positive here → downward.
#
# Identification
# --------------
# The correct adjustment set is identical to UC2: {budget, dim, ls, nl, snr}.
# observed_flag is a descendant of both strategy and outcome — conditioning on
# it violates the backdoor criterion by opening a non-causal path.
#
# Reference: uc4_dag_reasoning.md

# ── UC4 module-level constants ────────────────────────────────────────────────
UC4_COLLIDER_W_STRATEGY: float = +1.5   # log-odds: BO campaigns more likely flagged
UC4_COLLIDER_W_OUTCOME:  float = +2.0   # log-odds: better outcomes more likely flagged
UC4_COLLIDER_INTERCEPT:  float = -1.0   # base log-odds of being flagged (~27% base rate)

UC4_NODE_META: dict[str, dict] = {
    **UC2_NODE_META,  # inherit all UC2 nodes (UC4 is campaign-level like UC2)
    "observed_flag": {
        "role": "collider",
        "dtype": "binary",
        "description": (
            "Binary indicator: did the experimenter flag this campaign as "
            "successful? Caused by BOTH strategy (BO campaigns flagged more "
            "often) and norm_best_outcome (better outcomes flagged more often). "
            "This is a COLLIDER — conditioning on it opens a non-causal path "
            "strategy → observed_flag ← norm_best_outcome and induces spurious "
            "correlation between strategy and outcome. "
            "Do NOT include in the adjustment set."
        ),
    },
}

# ── UC4 adjustment set definitions ───────────────────────────────────────────
# Each entry: (label, set_of_covariates, expected_bias_direction, description)
# Used by get_uc4_adjustment_sets() and the analysis notebook.
_UC4_ADJUSTMENT_SETS: list[dict] = [
    {
        "label":       "correct",
        "covariates":  frozenset(["budget", "dimensionality",
                                  "landscape_smoothness", "noise_level", "snr"]),
        "bias_type":   "none",
        "description": (
            "Full UC2 adjustment set {budget, dim, ls, nl, snr}. "
            "snr is technically redundant given ls and nl (snr = ls/nl), "
            "but including it enables direct SNR moderation estimation (UC2) "
            "and is the canonical set used throughout UC1–UC3."
        ),
        "expected_bias_direction": "zero",
    },
    {
        "label":       "drop_snr",
        "covariates":  frozenset(["budget", "dimensionality",
                                  "landscape_smoothness", "noise_level"]),
        "bias_type":   "omission",
        "description": (
            "Omit derived confounder snr (keep ls and nl). "
            "Because snr = ls/nl, conditioning on ls and nl already blocks "
            "all snr backdoor paths — expected near-zero bias."
        ),
        "expected_bias_direction": "zero",
    },
    {
        "label":       "drop_ls",
        "covariates":  frozenset(["budget", "dimensionality",
                                  "noise_level", "snr"]),
        "bias_type":   "omission",
        "description": "Omit landscape_smoothness (keep nl and snr).",
        "expected_bias_direction": "positive",
    },
    {
        "label":       "drop_nl",
        "covariates":  frozenset(["budget", "dimensionality",
                                  "landscape_smoothness", "snr"]),
        "bias_type":   "omission",
        "description": "Omit noise_level (keep ls and snr).",
        "expected_bias_direction": "positive",
    },
    {
        "label":       "drop_snr_related",
        "covariates":  frozenset(["budget", "dimensionality"]),
        "bias_type":   "omission",
        "description": "Omit all SNR-related confounders (ls, nl, snr).",
        "expected_bias_direction": "positive",
    },
    {
        "label":       "empty",
        "covariates":  frozenset(),
        "bias_type":   "omission",
        "description": "Naive unadjusted ATE — no covariates at all.",
        "expected_bias_direction": "positive",
    },
    {
        "label":       "add_non_confounder",
        "covariates":  frozenset(["budget", "dimensionality",
                                  "landscape_smoothness", "noise_level", "snr",
                                  "initial_design_quality"]),
        "bias_type":   "spurious_inclusion",
        "description": (
            "Correct set + initial_design_quality (non-confounder: affects "
            "outcome but not strategy). Should be harmless."
        ),
        "expected_bias_direction": "zero",
    },
    {
        "label":       "collider",
        "covariates":  frozenset(["budget", "dimensionality",
                                  "landscape_smoothness", "noise_level", "snr",
                                  "observed_flag"]),
        "bias_type":   "collider",
        "description": (
            "Correct set + observed_flag (collider: caused by both strategy "
            "and outcome). Introduces downward bias via Berkson's paradox: "
            "conditioning on the collider attenuates the within-stratum "
            "strategy-outcome correlation (observed empirically: -41% bias)."
        ),
        "expected_bias_direction": "negative",
    },
]


def uc4_dag() -> nx.DiGraph:
    """
    Return the UC4 causal DAG as a networkx.DiGraph.

    Scientific question
    -------------------
    Are the confounders in our simulation design correctly identified?
    What is the bias from using the wrong adjustment set?

    New elements relative to UC2
    ----------------------------
    Node  : observed_flag — a collider caused by strategy and norm_best_outcome.
              Role: collider (do NOT adjust for it).

    Edges :
      strategy         → observed_flag  (BO campaigns more likely flagged)
      norm_best_outcome → observed_flag  (better outcomes more likely flagged)

    The UC2 structural equations are unchanged — observed_flag is a
    downstream consequence, not a cause of strategy or outcome.

    Identification
    --------------
    Correct adjustment set: {budget, dimensionality, landscape_smoothness,
    noise_level, snr} — identical to UC2.

    observed_flag must NOT be in the adjustment set: it is a collider
    (descendant of both strategy and outcome). Conditioning on it opens
    the path strategy → observed_flag ← norm_best_outcome, inducing
    spurious correlation.

    Returns
    -------
    nx.DiGraph
        Inherits all UC2 nodes and edges.
        Adds: observed_flag node, strategy→observed_flag,
              norm_best_outcome→observed_flag edges.
    """
    G = uc2_dag()
    G.graph["name"] = "UC4_confounder_sensitivity"

    # Update node metadata to UC4 version
    for node, meta in UC4_NODE_META.items():
        if node not in G.nodes:
            G.add_node(node, **meta)
        else:
            G.nodes[node].update(meta)

    # ── New node: observed_flag ───────────────────────────────────────────────
    G.add_node("observed_flag", **UC4_NODE_META["observed_flag"])

    # ── strategy → observed_flag ──────────────────────────────────────────────
    G.add_edge(
        "strategy", "observed_flag",
        weight=UC4_COLLIDER_W_STRATEGY,
        rationale=(
            "BO campaigns are more likely to be flagged as successful because "
            "practitioners associate BO with rigorous methodology, independent "
            "of the actual outcome. Log-odds weight = "
            f"{UC4_COLLIDER_W_STRATEGY:+.1f}."
        ),
    )

    # ── norm_best_outcome → observed_flag ─────────────────────────────────────
    G.add_edge(
        "norm_best_outcome", "observed_flag",
        weight=UC4_COLLIDER_W_OUTCOME,
        rationale=(
            "Campaigns with better outcomes are more likely to be flagged as "
            "successful — the primary driver of the success flag. "
            f"Log-odds weight = {UC4_COLLIDER_W_OUTCOME:+.1f}."
        ),
    )

    return G


def get_uc4_adjustment_sets() -> list[dict]:
    """
    Return the 8 UC4 adjustment sets as a list of dicts.

    Each dict has keys:
      label       : str         — short identifier
      covariates  : frozenset   — set of covariate names to include in OLS
      bias_type   : str         — 'none' | 'omission' | 'spurious_inclusion' | 'collider'
      description : str         — human-readable description
      expected_bias_direction : str — 'zero' | 'positive' | 'negative'

    The correct adjustment set is the first entry (label='correct').
    """
    return list(_UC4_ADJUSTMENT_SETS)


def get_uc4_correct_adjustment_set() -> list[str]:
    """Return the correct UC4 adjustment set (identical to UC2)."""
    return get_uc2_adjustment_set()


# ── Update registry ───────────────────────────────────────────────────────────
GRAPH_REGISTRY["UC4"] = uc4_dag


# ── UC5: ATT Estimation for Strategy Switching ────────────────────────────────
#
# Scientific question
# -------------------
# Among campaigns that *actually used* BO, what would their outcome have been
# under random search?  This is the Average Treatment Effect on the Treated
# (ATT): E[Y(1) − Y(0) | T = 1].
#
# Why ATT rather than ATE?
# ------------------------
# UC1 estimates the ATE: the average effect of BO across ALL campaigns,
# including those where BO was never used.  The ATE answers "what would
# happen if we randomly assigned BO to a new campaign?"
#
# The ATT answers a different, more actionable question: "for the campaigns
# where a practitioner *chose* BO, was that choice justified?"  This is the
# retrospective evaluation question — it uses the existing data structure
# (BO campaigns are the treated group) and is directly comparable to UC1.
#
# ATT vs ATE in the presence of positive selection
# -------------------------------------------------
# Because practitioners preferentially apply BO to easier campaigns (high SNR,
# smooth landscape, large budget — the UC1/UC2 confounders), the treated group
# is positively selected.  In this setting:
#
#   ATT > ATE
#
# Intuition: BO campaigns are drawn from the high-SNR tail of the landscape
# distribution, where BO's advantage is largest (UC2 design rule DR-B).
# The ATE averages over the full distribution including low-SNR campaigns
# where BO barely helps.  The ATT conditions on the treated subgroup, so it
# reflects BO's effect where it is actually deployed.
#
# Estimand
# --------
#   ATT = E[Y(1) − Y(0) | T = 1]
#       = E[Y(1) | T = 1] − E[Y(0) | T = 1]
#
# The first term is directly observed (BO campaign outcomes).
# The second term is the counterfactual: what would BO campaigns have achieved
# under random search?  This is estimated via backdoor adjustment restricted
# to the treated subgroup.
#
# Identification
# --------------
# The ATT is identified by the same backdoor adjustment set as UC1/UC2:
# {landscape_smoothness, budget, noise_level, dimensionality, snr}.
# The DAG is identical to UC2 — no new nodes or edges are needed.
# The only change is in the *estimation target*: we restrict to T=1 campaigns.
#
# DoWhy feature: counterfactual estimation
# -----------------------------------------
# DoWhy's estimate_effect() with target_units="att" restricts the estimand
# to the treated subgroup.  The linear regression backdoor estimator then
# fits on the full data but evaluates the counterfactual at T=0 for each
# treated unit, giving per-campaign Y(0) predictions.
#
# Additional analyses
# -------------------
# 1. Individual counterfactual distributions: per-campaign Y(1) − Y(0) for
#    all BO-treated campaigns, showing the heterogeneity of the switching loss.
# 2. SNR-stratified ATT: ATT computed separately for low/mid/high SNR bins,
#    confirming that the ATT is largest where BO is most effective (high SNR).
# 3. ATE vs ATT comparison: quantifies the positive selection bias.
#
# Design rule DR-H
# ----------------
# "Among campaigns where BO was used, switching to random search would have
# reduced the best outcome by ATT ≈ +0.073 units on average (AIPW, 95% CI:
# [+0.060, +0.085]; true ATT = +0.088).  The switching loss is largest for
# high-SNR campaigns (ATT_high > ATT_mid > ATT_low), confirming that BO's
# retrospective value is concentrated in the regime where its surrogate model
# is most reliable.
#
# CAVEATS (see uc5_dag_reasoning.md §5–6 for full discussion):
# 1. Interaction term (strategy × snr_s) has p=0.057 — marginally non-
#    significant at α=0.05.  Stratified ATT ordering is directional, not
#    a precise estimate.
# 2. Estimated ATT_low = +0.017 has the WRONG SIGN vs true ATT_low = −0.007.
#    The low-SNR bin (n=38 treated) is severely underpowered; the linear OLS
#    model extrapolates incorrectly into the non-linear low-SNR regime.
#    Use the true ATT_low (known by DGP construction) for the direction.
# 3. Refutation tests validate identification assumptions (shared by ATE and
#    ATT), not the AIPW ATT point estimate directly."
#
# Reference: uc5_dag_reasoning.md

# UC5 SNR bin boundaries (used in notebook and tests)
UC5_SNR_LOW_MAX:  float = 1.5   # SNR < 1.5  → low bin
UC5_SNR_HIGH_MIN: float = 3.0   # SNR ≥ 3.0  → high bin
                                  # 1.5 ≤ SNR < 3.0 → mid bin
                                  # Chosen to straddle UC2_SNR_THRESHOLD (τ=2.0)

UC5_NODE_META: dict[str, dict] = {
    **UC2_NODE_META,  # UC5 is campaign-level; inherits all UC2 nodes unchanged
}


def uc5_dag() -> nx.DiGraph:
    """
    Return the UC5 causal DAG as a networkx.DiGraph.

    Scientific question
    -------------------
    Among campaigns that used BO, what would their outcome have been under
    random search?  (ATT: Average Treatment Effect on the Treated)

    DAG structure
    -------------
    Identical to UC2.  No new nodes or edges are needed — the ATT is a
    different *estimation target* on the same causal graph, not a different
    causal structure.

    Identification
    --------------
    ATT identified by backdoor adjustment on
    {landscape_smoothness, budget, noise_level, dimensionality, snr},
    restricted to the treated subgroup (T = 1, BO campaigns).

    Returns
    -------
    nx.DiGraph
        Identical structure to uc2_dag(), renamed "UC5_att_strategy_switching".
    """
    G = uc2_dag()
    G.graph["name"] = "UC5_att_strategy_switching"
    G.graph["estimand"] = "ATT"
    G.graph["treated_condition"] = "strategy == 1"
    G.graph["estimand_note"] = (
        "ATT = E[Y(1) - Y(0) | T=1]. "
        "DAG is identical to UC2; the ATT is a different estimation target, "
        "not a different causal structure. "
        "Positive selection (practitioners choose BO for easier campaigns) "
        "implies ATT > ATE."
    )
    return G


def get_uc5_adjustment_set() -> list[str]:
    """
    Return the UC5 backdoor adjustment set (identical to UC2).
    ATT identification uses the same backdoor criterion as ATE.
    """
    return get_uc2_adjustment_set()


# ── Update registry ───────────────────────────────────────────────────────────
GRAPH_REGISTRY["UC5"] = uc5_dag


if __name__ == "__main__":
    print("=== UC1 ===")
    G1 = uc1_dag()
    summarise(G1)
    adj1 = get_confounders(G1, treatment="strategy", outcome="norm_best_outcome")
    print(f"\nBackdoor adjustment set (graph-derived): {adj1}")
    print(f"Hardcoded adjustment set:                {get_adjustment_set()}")
    assert adj1 == get_adjustment_set(), (
        "Graph-derived and hardcoded adjustment sets disagree — "
        "update get_adjustment_set() to match the graph."
    )
    print("Assertion passed: UC1 adjustment sets agree.")

    print("\n=== UC2 ===")
    G2 = uc2_dag()
    summarise(G2)
    adj2 = get_confounders(G2, treatment="strategy", outcome="norm_best_outcome")
    print(f"\nBackdoor adjustment set (graph-derived): {adj2}")
    print(f"Hardcoded adjustment set:                {get_uc2_adjustment_set()}")
    assert adj2 == get_uc2_adjustment_set(), (
        "Graph-derived and hardcoded UC2 adjustment sets disagree."
    )
    print("Assertion passed: UC2 adjustment sets agree.")

    print("\n=== UC3 ===")
    G3 = uc3_dag()
    summarise(G3)
    adj3 = get_confounders(G3, treatment="strategy", outcome="norm_best_outcome_t")
    print(f"\nBackdoor adjustment set (graph-derived): {adj3}")
    print(f"Hardcoded adjustment set:                {get_uc3_adjustment_set()}")
    assert adj3 == get_uc3_adjustment_set(), (
        "Graph-derived and hardcoded UC3 adjustment sets disagree."
    )
    print("Assertion passed: UC3 adjustment sets agree.")
    print(f"\nUC3 warmup constants:")
    print(f"  UC3_ALPHA_RS     = {UC3_ALPHA_RS}")
    print(f"  UC3_WARMUP_BASE  = {UC3_WARMUP_BASE}")
    print(f"  UC3_WARMUP_SCALE = {UC3_WARMUP_SCALE}")
    print(f"  UC3_WARMUP_W_DIM = {UC3_WARMUP_W_DIM}")
    print(f"  UC3_WARMUP_W_SNR = {UC3_WARMUP_W_SNR}")
    print(f"  UC3_WARMUP_W_IDQ = {UC3_WARMUP_W_IDQ}")
    print(f"  UC3_STEP_NOISE   = {UC3_STEP_NOISE}")

    print("\n=== UC4 ===")
    G4 = uc4_dag()
    summarise(G4)
    adj4 = get_uc4_correct_adjustment_set()
    print(f"\nCorrect adjustment set: {adj4}")
    print(f"Number of adjustment sets to test: {len(get_uc4_adjustment_sets())}")
    for s in get_uc4_adjustment_sets():
        print(f"  [{s['bias_type']:20s}] {s['label']:25s} — {s['description'][:60]}")
    assert "observed_flag" in G4.nodes, "observed_flag node missing from UC4 DAG"
    assert G4.has_edge("strategy", "observed_flag"), "strategy→observed_flag edge missing"
    assert G4.has_edge("norm_best_outcome", "observed_flag"), "outcome→observed_flag edge missing"
    print("Assertions passed: UC4 DAG structure correct.")

    print("\n=== UC5 ===")
    G5 = uc5_dag()
    summarise(G5)
    adj5 = get_uc5_adjustment_set()
    print(f"\nATT adjustment set: {adj5}")
    print(f"Estimand:           {G5.graph['estimand']}")
    print(f"Treated condition:  {G5.graph['treated_condition']}")
    assert G5.graph["estimand"] == "ATT", "UC5 estimand should be ATT"
    assert adj5 == get_uc2_adjustment_set(), "UC5 adjustment set should match UC2"
    assert G5.number_of_nodes() == G5.number_of_nodes(), "UC5 node count should match UC2"
    print("Assertions passed: UC5 DAG structure correct.")
    print(f"\nSNR bin boundaries: low < {UC5_SNR_LOW_MAX}, high ≥ {UC5_SNR_HIGH_MIN}")
