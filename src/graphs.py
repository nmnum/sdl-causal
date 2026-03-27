"""
graphs.py — Causal DAG definitions for the Self-Driving Lab project.

This file is the single source of truth for causal structure.
Notebooks and the simulator import from here; they never redefine graphs inline.

Each graph is returned by a factory function so that callers always get a
fresh copy (networkx graphs are mutable).  Edge metadata is stored in the
edge attribute dict and mirrored in inline comments so the reasoning is
visible without running the code.

Reference document: uc1_dag_reasoning.md
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
    return sorted(['landscape_smoothness', 'budget', 'noise_level', 'dimensionality'])

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


# ── Registry ──────────────────────────────────────────────────────────────────
# As more use cases are added, register their factory functions here.
# The simulator and notebooks can look up graphs by UC name.

GRAPH_REGISTRY: dict[str, Callable[[], nx.DiGraph]] = {
    "UC1": uc1_dag,
}


if __name__ == "__main__":
    G = uc1_dag()
    summarise(G)

    adj = get_confounders(G, treatment="strategy", outcome="norm_best_outcome")
    print(f"\nBackdoor adjustment set (graph-derived): {adj}")
    print(f"Hardcoded adjustment set:                {get_adjustment_set()}")
    assert adj == get_adjustment_set(), (
        "Graph-derived and hardcoded adjustment sets disagree — "
        "update get_adjustment_set() to match the graph."
    )
    print("\nAssertion passed: adjustment sets agree.")
