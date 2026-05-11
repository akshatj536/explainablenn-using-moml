"""
main.py
=======
Entry point for the Interpretable Neural Network discovery system.

Runs the multi-objective evolutionary algorithm on two benchmark datasets:
  1. Breast Cancer Wisconsin  (binary classification)
  2. Boston Housing           (regression)

For each dataset the pipeline is:
    load data → evolve population → extract Pareto front
    → save models/matrices → plot objectives → plot history

Output structure
----------------
results/
  breast_cancer/
    pareto_models/
      model_00/  mask1.npy  mask2.npy  weights.pt  metadata.json
      model_01/  ...
      pareto_summary.json
    pareto_3d.png
    pareto_2d.png
    history.png
  boston_housing/
    (same layout)
"""

import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # headless backend — safe for servers / no display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3d projection)
from typing import List, Dict, Any

from data_loader import load_breast_cancer, load_boston_housing
from evolution   import evolve, get_pareto_front, Individual


# ============================================================================
# Saving utilities
# ============================================================================

def save_pareto_models(pareto_front: List[Individual],
                        output_dir: str,
                        feature_names: List[str]) -> None:
    """
    Persist every Pareto-optimal model to disk.

    Per-model files
    ---------------
    mask1.npy      – int8 numpy array  (n_input × n_hidden_max)
    mask2.npy      – int8 numpy array  (n_hidden_max × n_output)
    weights.pt     – PyTorch state dict (W1, b1, W2, b2)
    metadata.json  – fitness values + human-readable feature list
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for i, ind in enumerate(pareto_front):
        model_dir = os.path.join(output_dir, f'model_{i:02d}')
        os.makedirs(model_dir, exist_ok=True)

        np.save(os.path.join(model_dir, 'mask1.npy'), ind.mask1)
        np.save(os.path.join(model_dir, 'mask2.npy'), ind.mask2)

        if ind.model is not None:
            torch.save(ind.model.state_dict(),
                       os.path.join(model_dir, 'weights.pt'))

        # Human-readable: which input features are actually used?
        active_idx   = np.where(ind.mask1.sum(axis=1) > 0)[0].tolist()
        active_names = [feature_names[j] for j in active_idx]

        meta = {
            'model_index':          i,
            'val_loss':             round(ind.fitness[0], 6),
            'complexity_score':     round(ind.fitness[1], 1),
            'active_features_count': round(ind.fitness[2], 0),
            'active_connections':   ind.model.get_active_connections() if ind.model else None,
            'active_neurons':       ind.model.get_active_neurons()     if ind.model else None,
            'active_feature_indices': active_idx,
            'active_feature_names': active_names,
            'pareto_rank':          ind.rank,
        }
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        summary.append(meta)

    with open(os.path.join(output_dir, 'pareto_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"    Saved {len(pareto_front)} Pareto models → {output_dir}/")


# ============================================================================
# Visualisation
# ============================================================================

def plot_pareto_front(pareto_front: List[Individual],
                       all_population: List[Individual],
                       output_dir: str,
                       dataset_name: str) -> None:
    """
    Generate and save Pareto-front visualisations.

    Plot 1 – 3D scatter (Performance × Complexity × Feature Sparsity):
        Non-Pareto population members are shown in light blue; Pareto-front
        members are highlighted as red stars so the trade-off surface is clear.

    Plot 2 – 2D pairwise projections (3 subplots):
        Each pair of objectives is shown side-by-side, useful for identifying
        which trade-offs dominate the search.
    """
    os.makedirs(output_dir, exist_ok=True)

    pareto_ids  = set(id(ind) for ind in pareto_front)
    non_pareto  = [ind for ind in all_population if id(ind) not in pareto_ids]

    # Use non-Pareto members as background so stars are always visible
    bg_fits     = np.array([ind.fitness for ind in non_pareto]) if non_pareto else None
    pareto_fits = np.array([ind.fitness for ind in pareto_front])

    obj_labels = ['Val Loss (↓)', 'Complexity (↓)', 'Active Features (↓)']

    # ------------------------------------------------------------------ #
    # 1. 3D scatter                                                        #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(pareto_fits[:, 0], pareto_fits[:, 1], pareto_fits[:, 2],
               c='crimson', s=100, marker='*', zorder=5, label='Pareto Front')

    ax.set_xlabel(obj_labels[0], fontsize=9, labelpad=8)
    ax.set_ylabel(obj_labels[1], fontsize=9, labelpad=8)
    ax.set_zlabel(obj_labels[2], fontsize=9, labelpad=8)
    ax.set_title(f'{dataset_name}\n3D Pareto Front',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    p = os.path.join(output_dir, 'pareto_3d.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    3D plot  → {p}")

    # ------------------------------------------------------------------ #
    # 2. Pairwise 2D projections                                           #
    # ------------------------------------------------------------------ #
    pairs  = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (xi, yi) in zip(axes, pairs):
        ax.scatter(pareto_fits[:, xi], pareto_fits[:, yi],
                   c='crimson', s=80, marker='*', zorder=5, label='Pareto Front')
        ax.set_xlabel(obj_labels[xi], fontsize=9)
        ax.set_ylabel(obj_labels[yi], fontsize=9)
        xi_short = obj_labels[xi].split(' ')[0]
        yi_short = obj_labels[yi].split(' ')[0]
        ax.set_title(f'{xi_short} vs {yi_short}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_name} – Pairwise Objective Projections',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, 'pareto_2d.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    2D plot  → {p}")


def plot_algorithm_comparison(
        lhfid_pareto:  List[Individual],
        lhfid_pop:     List[Individual],
        nsga2_pareto:  List[Individual],
        nsga2_pop:     List[Individual],
        output_dir:    str,
        dataset_name:  str) -> None:
    """
    Plot LHFiD and NSGA-II Pareto fronts on the same 3D and 2D axes.

    Colour scheme
    -------------
    LHFiD Pareto front  : crimson stars   (★)
    NSGA-II Pareto front : royalblue stars (★)
    Non-Pareto population members are shown as faint background dots
    in their respective colours so the front stands out clearly.
    """
    os.makedirs(output_dir, exist_ok=True)

    obj_labels = ['Val Loss (↓)', 'Complexity (↓)', 'Active Features (↓)']

    def _non_pareto_fits(pareto, pop):
        ids = set(id(ind) for ind in pareto)
        bg  = [ind for ind in pop if id(ind) not in ids]
        return np.array([ind.fitness for ind in bg]) if bg else None

    lhfid_pf  = np.array([ind.fitness for ind in lhfid_pareto])
    nsga2_pf  = np.array([ind.fitness for ind in nsga2_pareto])
    lhfid_bg  = _non_pareto_fits(lhfid_pareto,  lhfid_pop)
    nsga2_bg  = _non_pareto_fits(nsga2_pareto,  nsga2_pop)

    # ------------------------------------------------------------------ #
    # 1. 3D comparison                                                     #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(lhfid_pf[:, 0], lhfid_pf[:, 1], lhfid_pf[:, 2],
               c='crimson', s=120, marker='*', zorder=5, label='LHFiD Pareto')
    ax.scatter(nsga2_pf[:, 0], nsga2_pf[:, 1], nsga2_pf[:, 2],
               c='royalblue', s=120, marker='*', zorder=5, label='NSGA-II Pareto')

    ax.set_xlim(0, 1)
    ax.set_xlabel(obj_labels[0], fontsize=9, labelpad=8)
    ax.set_ylabel(obj_labels[1], fontsize=9, labelpad=8)
    ax.set_zlabel(obj_labels[2], fontsize=9, labelpad=8)
    ax.set_title(f'{dataset_name}\nLHFiD vs NSGA-II — 3D Pareto Front',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    p = os.path.join(output_dir, 'comparison_3d.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Comparison 3D → {p}")

    # ------------------------------------------------------------------ #
    # 2. Pairwise 2D comparison                                            #
    # ------------------------------------------------------------------ #
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (xi, yi) in zip(axes, pairs):
        ax.scatter(lhfid_pf[:, xi], lhfid_pf[:, yi],
                   c='crimson', s=90, marker='*', zorder=5, label='LHFiD')
        ax.scatter(nsga2_pf[:, xi], nsga2_pf[:, yi],
                   c='royalblue', s=90, marker='*', zorder=5, label='NSGA-II')

        if xi == 0:
            ax.set_xlim(0, 1)
        if yi == 0:
            ax.set_ylim(0, 1)
        ax.set_xlabel(obj_labels[xi], fontsize=9)
        ax.set_ylabel(obj_labels[yi], fontsize=9)
        xi_short = obj_labels[xi].split(' ')[0]
        yi_short = obj_labels[yi].split(' ')[0]
        ax.set_title(f'{xi_short} vs {yi_short}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_name} – LHFiD vs NSGA-II Pairwise Projections',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, 'comparison_2d.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Comparison 2D → {p}")


def plot_history(history: List[Dict],
                  output_dir: str,
                  dataset_name: str) -> None:
    """Plot min-loss, mean-complexity, and mean-active-features over generations."""
    os.makedirs(output_dir, exist_ok=True)

    gens   = [h['generation']      for h in history]
    losses = [h['min_loss']        for h in history]
    compl  = [h['mean_complexity'] for h in history]
    feats  = [h['mean_features']   for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(gens, losses, 'o-', color='steelblue',   lw=2, ms=3)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Min Validation Loss')
    axes[0].set_title('Best Loss per Generation')
    axes[0].grid(alpha=0.3)

    axes[1].plot(gens, compl,  's-', color='darkorange',  lw=2, ms=3)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Mean Complexity Score')
    axes[1].set_title('Mean Complexity per Generation')
    axes[1].grid(alpha=0.3)

    axes[2].plot(gens, feats,  '^-', color='seagreen',    lw=2, ms=3)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Mean Active Features')
    axes[2].set_title('Mean Active Features per Generation')
    axes[2].grid(alpha=0.3)

    fig.suptitle(f'{dataset_name} – Evolutionary Progress',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, 'history.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    History  → {p}")


# ============================================================================
# Per-dataset experiment runner
# ============================================================================

def run_experiment(dataset_name: str,
                    data: Dict[str, Any],
                    output_dir: str,
                    evo_config: Dict) -> None:
    """
    Full pipeline for one dataset: load → evolve → save → plot.
    """
    cat_names = [data['feature_names'][i]
                 for i in data.get('categorical_cols', [])]

    print(f"\n{'='*62}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Task    : {data['task']}")
    print(f"  Samples : train={data['X_train'].shape[0]}  "
          f"val={data['X_val'].shape[0]}  "
          f"test={data['X_test'].shape[0]}")
    print(f"  Features: {data['n_features']} total  "
          f"| categorical (kept as-is): {cat_names if cat_names else 'none'}")
    print(f"{'='*62}")

    # ── Evolve ──────────────────────────────────────────────────────── #
    final_pop, history = evolve(data, **evo_config)

    # ── Pareto front ─────────────────────────────────────────────────── #
    pareto = get_pareto_front(final_pop)
    pareto_sorted = sorted(pareto, key=lambda ind: ind.fitness[0])

    print(f"\n  Pareto front: {len(pareto)} non-dominated models")
    print(f"  {'val_loss':>10}  {'complexity':>12}  {'active_feats':>14}")
    print(f"  {'-'*42}")
    for ind in pareto_sorted:
        print(f"  {ind.fitness[0]:>10.4f}  {ind.fitness[1]:>12.0f}  "
              f"{ind.fitness[2]:>14.0f}")

    # ── Save ─────────────────────────────────────────────────────────── #
    model_dir = os.path.join(output_dir, 'pareto_models')
    save_pareto_models(pareto_sorted, model_dir, data['feature_names'])

    # ── Plot ─────────────────────────────────────────────────────────── #
    plot_pareto_front(pareto, final_pop, output_dir, dataset_name)
    plot_history(history, output_dir, dataset_name)

    print(f"\n  All outputs → {output_dir}/")


# ============================================================================
# Entry point
# ============================================================================

def _next_run_dir(base: str) -> str:
    """
    Return a versioned output directory that does not yet exist.

    Pattern: base/run_v1, base/run_v2, base/run_v3, …
    The first version that is absent on disk is returned, so previous
    runs are never overwritten.
    """
    v = 1
    while True:
        candidate = os.path.join(base, f'run_v{v}')
        if not os.path.exists(candidate):
            return candidate
        v += 1


def main() -> None:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    BOSTON_PATH = os.path.join(BASE_DIR, 'archive', 'HousingData.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Shared hyper-parameters ──────────────────────────────────────── #
    # pop_size = n_ref_vectors = 210  (H=19, 3 objectives → C(21,2) = 210 Das-Dennis vectors)
    base_config = dict(
        pop_size               = 210,
        n_generations          = 500,
        n_hidden_max           = 20,
        train_epochs_init      = 200,
        train_epochs_offspring = 120,
        lr                     = 1e-3,
        weight_decay           = 1e-4,
        p_flip                 = 0.05,
        p_add_neuron           = 0.20,
        p_remove_neuron        = 0.20,
        n_ref_partitions       = 19,    # 210 reference vectors (used by LHFiD only)
        seed                   = 42,
        verbose                = True,
        device                 = device,
    )

    bh_out = _next_run_dir(os.path.join(BASE_DIR, 'results', 'boston_housing'))
    print(f"Output dir: {bh_out}\n")

    bh_data = load_boston_housing(BOSTON_PATH)
    dataset_name = 'Boston Housing (Regression)'

    print(f"\n{'='*62}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Samples : train={bh_data['X_train'].shape[0]}  "
          f"val={bh_data['X_val'].shape[0]}  "
          f"test={bh_data['X_test'].shape[0]}")
    print(f"  Features: {bh_data['n_features']} total")
    print(f"{'='*62}")

    # ── Run LHFiD ────────────────────────────────────────────────────── #
    print("\n--- LHFiD ---")
    lhfid_pop, lhfid_hist = evolve(bh_data, **{**base_config, 'method': 'lhfid'})
    lhfid_pareto = sorted(get_pareto_front(lhfid_pop), key=lambda i: i.fitness[0])
    print(f"\n  LHFiD Pareto front: {len(lhfid_pareto)} models")

    # ── Run NSGA-II for the same number of generations LHFiD used ────── #
    n_gens_lhfid = len(lhfid_hist)
    print(f"\n--- NSGA-II ({n_gens_lhfid} generations, matching LHFiD) ---")
    nsga2_pop, nsga2_hist = evolve(bh_data,
                                   **{**base_config,
                                      'method':       'nsga2',
                                      'n_generations': n_gens_lhfid})
    nsga2_pareto = sorted(get_pareto_front(nsga2_pop), key=lambda i: i.fitness[0])
    print(f"\n  NSGA-II Pareto front: {len(nsga2_pareto)} models")

    # ── Save individual results ───────────────────────────────────────── #
    save_pareto_models(lhfid_pareto,
                       os.path.join(bh_out, 'lhfid', 'pareto_models'),
                       bh_data['feature_names'])
    save_pareto_models(nsga2_pareto,
                       os.path.join(bh_out, 'nsga2', 'pareto_models'),
                       bh_data['feature_names'])

    plot_history(lhfid_hist,  os.path.join(bh_out, 'lhfid'),  f'{dataset_name} — LHFiD')
    plot_history(nsga2_hist,  os.path.join(bh_out, 'nsga2'),  f'{dataset_name} — NSGA-II')

    # ── Comparison plots (both algorithms on same axes) ───────────────── #
    plot_algorithm_comparison(
        lhfid_pareto, lhfid_pop,
        nsga2_pareto, nsga2_pop,
        bh_out, dataset_name,
    )

    print(f"\n  All outputs → {bh_out}/")


if __name__ == '__main__':
    main()
