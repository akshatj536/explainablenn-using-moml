"""
evolution.py
============
Multi-Objective Evolutionary Algorithm for discovering Interpretable Neural
Networks.  Implements an LHFiD-style loop:

    Initialise → [Evaluate → Mutate → Train → LHFiD-Select] × n_generations

Three objectives (ALL minimised):
    obj[0] – Performance    : validation loss (BCE for classification, MSE for regression)
    obj[1] – Complexity     : active_connections + active_neurons
    obj[2] – Feature Sparsity: number of active input features

Key design choices
------------------
* Fixed maximum hidden-layer width (n_hidden_max).  The masks determine which
  neurons are alive; dead neurons (no incoming OR no outgoing connections)
  contribute nothing to the output and are excluded from all complexity counts.
* Three mutation operators applied in sequence each generation:
    1. Stochastic bit-flip  – explores the full topology neighbourhood
    2. Add-neuron           – grows the network when it is too small
    3. Remove-neuron        – prunes the network when it is too large
* Lamarckian warm-start: offspring inherit parent weights for surviving
  connections; only newly-added connections are randomly initialised.
  This dramatically reduces the training epochs needed per offspring.
* LHFiD selection (replaces NSGA-II crowding distance):
    - Reference directions (Das-Dennis) guide diversity across the Pareto front
    - Objectives normalised via adaptive ideal + nadir points so all three
      scales (loss, complexity, features) are comparable
    - Localized Pareto dominance within each reference-vector cluster
    - High-fidelity tie-breaker (directional Chebyshev scalarisation)
    - Stabilisation tracker drives automatic early termination when the
      population has converged (no need to wait for the full n_generations)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any

from network import InterpretableNN
from lhfid   import (generate_reference_directions, compute_nadir_point,
                      lhfid_survival_selection, StabilizationTracker)


# ============================================================================
# Individual
# ============================================================================

class Individual:
    """
    A single candidate network in the evolutionary population.

    Attributes
    ----------
    mask1 : np.ndarray, int8, shape (n_input, n_hidden_max)
        Binary connection matrix for the input→hidden layer.
    mask2 : np.ndarray, int8, shape (n_hidden_max, n_output)
        Binary connection matrix for the hidden→output layer.
    model : InterpretableNN | None
        The PyTorch model (None until `build_model` is called).
    fitness : tuple[float, float, float] | None
        (val_loss, complexity, active_features) — all minimised.
    rank : int
        Pareto front rank assigned by NSGA-II (0 = non-dominated front).
    crowding_distance : float
        Crowding distance within its Pareto front (higher = more isolated).
    """

    __slots__ = ('mask1', 'mask2', 'model', 'fitness', 'rank', 'crowding_distance')

    def __init__(self,
                 mask1: np.ndarray,
                 mask2: np.ndarray,
                 model: Optional[InterpretableNN] = None):
        self.mask1             = mask1.astype(np.int8)
        self.mask2             = mask2.astype(np.int8)
        self.model             = model
        self.fitness: Optional[Tuple[float, float, float]] = None
        self.rank              = 0
        self.crowding_distance = 0.0


# ============================================================================
# Population initialisation
# ============================================================================

def _random_mask(n_rows: int, n_cols: int,
                 density: float, rng: np.random.RandomState) -> np.ndarray:
    """Binary mask with approximately `density` fraction of 1-bits."""
    mask = (rng.rand(n_rows, n_cols) < density).astype(np.int8)
    if mask.sum() == 0:                          # ensure at least one connection
        mask[rng.randint(0, n_rows), rng.randint(0, n_cols)] = 1
    return mask


def _ensure_valid_path(mask1: np.ndarray, mask2: np.ndarray,
                        rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Guarantee at least one complete input → hidden → output path.

    Without this guarantee a fully-zeroed network outputs a constant and
    gradients never flow.  We add one random path if none exists.
    """
    m1, m2 = mask1.copy(), mask2.copy()
    n_input, n_hidden = m1.shape
    _, n_output       = m2.shape

    has_in  = m1.sum(axis=0) > 0    # (n_hidden,)  neurons with incoming
    has_out = m2.sum(axis=1) > 0    # (n_hidden,)  neurons with outgoing
    alive   = has_in & has_out

    if not alive.any():
        h = rng.randint(0, n_hidden)
        m1[rng.randint(0, n_input),   h] = 1
        m2[h, rng.randint(0, n_output)] = 1

    return m1, m2


def initialize_population(pop_size: int,
                           n_input: int,
                           n_hidden_max: int,
                           n_output: int,
                           min_density: float = 0.03,
                           max_density: float = 0.40,
                           seed: int = 0) -> List[Individual]:
    """
    Create a diverse initial population with linearly-spaced sparsity levels.

    Spacing densities from min_density to max_density ensures the initial
    population already spans a wide range on the complexity axis, giving
    NSGA-II diverse material to work with from generation 0.
    """
    rng = np.random.RandomState(seed)
    densities = np.linspace(min_density, max_density, pop_size)

    population = []
    for d in densities:
        mask1 = _random_mask(n_input,      n_hidden_max, d, rng)
        mask2 = _random_mask(n_hidden_max, n_output,     d, rng)
        mask1, mask2 = _ensure_valid_path(mask1, mask2, rng)
        population.append(Individual(mask1, mask2))

    return population


# ============================================================================
# Mutation operators
# ============================================================================

def _flip_connections(mask1: np.ndarray, mask2: np.ndarray,
                       p_flip: float,
                       rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic bit-flip mutation.

    Each connection bit — whether currently 0 or 1 — is toggled independently
    with probability p_flip.  This is the primary exploration mechanism:
      * Flipping a 0 → 1 adds a new synapse (local complexity increase).
      * Flipping a 1 → 0 removes an existing synapse (local pruning).

    The symmetric treatment of add/remove ensures neither structural growth
    nor pruning is systematically favoured.
    """
    flip1 = (rng.rand(*mask1.shape) < p_flip).astype(np.int8)
    flip2 = (rng.rand(*mask2.shape) < p_flip).astype(np.int8)
    new_mask1 = np.bitwise_xor(mask1, flip1).astype(np.int8)
    new_mask2 = np.bitwise_xor(mask2, flip2).astype(np.int8)
    return new_mask1, new_mask2


def _add_neuron(mask1: np.ndarray, mask2: np.ndarray,
                rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grow the network by activating one dead hidden neuron.

    A neuron is 'dead' if it lacks incoming connections OR outgoing connections
    (it cannot contribute to any output).  We pick one dead neuron at random,
    wire it to 1–3 random input features and 1 output, making it immediately
    part of an active computation path.

    This operator allows the topology to increase in capacity when small
    networks are under-performing.
    """
    m1, m2 = mask1.copy(), mask2.copy()
    n_input, n_hidden = m1.shape
    _, n_output        = m2.shape

    has_in  = m1.sum(axis=0) > 0
    has_out = m2.sum(axis=1) > 0
    dead    = np.where(~(has_in & has_out))[0]

    if len(dead) == 0:
        return m1, m2    # all neurons already active

    h = rng.choice(dead)

    # Wire 1–3 random input features into this neuron
    n_in = rng.randint(1, min(4, n_input + 1))
    for i in rng.choice(n_input, n_in, replace=False):
        m1[i, h] = 1

    # Wire neuron to 1 random output
    m2[h, rng.randint(0, n_output)] = 1

    return m1, m2


def _remove_neuron(mask1: np.ndarray, mask2: np.ndarray,
                    rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prune the network by deactivating one live hidden neuron.

    Zeroes ALL connections of a randomly-chosen active neuron, effectively
    removing it from the computation graph.  We always preserve at least one
    active neuron so the network stays valid.

    This operator drives complexity reduction, pushing the evolutionary search
    toward simpler (more interpretable) solutions.
    """
    m1, m2 = mask1.copy(), mask2.copy()

    has_in  = m1.sum(axis=0) > 0
    has_out = m2.sum(axis=1) > 0
    alive   = np.where(has_in & has_out)[0]

    if len(alive) <= 1:
        return m1, m2    # keep at least one active neuron

    h = rng.choice(alive)
    m1[:, h] = 0
    m2[h, :]  = 0

    return m1, m2


def mutate(individual: Individual,
           p_flip: float = 0.04,
           p_add_neuron: float = 0.15,
           p_remove_neuron: float = 0.15,
           rng: Optional[np.random.RandomState] = None) -> 'Individual':
    """
    Combined mutation operator applied in the order: flip → add → remove.

    Three mutation types allow exploration of three distinct structural changes:
      1. Bit-flip   : fine-grained topology perturbation (always applied)
      2. Add-neuron : coarse-grained growth (applied with prob p_add_neuron)
      3. Remove-neuron: coarse-grained pruning (applied with prob p_remove_neuron)

    Applying add and remove in the same call is intentional: the net effect
    depends on which neurons were active before and after flipping, so they
    can interact non-trivially.

    After mutation, `_ensure_valid_path` is called to guarantee the child
    still has at least one end-to-end computation path.

    Parameters
    ----------
    individual      : parent Individual whose masks are copied then mutated
    p_flip          : per-bit flip probability (typical: 0.03 – 0.06)
    p_add_neuron    : probability of invoking the add-neuron operator
    p_remove_neuron : probability of invoking the remove-neuron operator
    rng             : numpy RandomState for reproducibility

    Returns
    -------
    A new Individual with mutated masks and no trained model.
    """
    if rng is None:
        rng = np.random.RandomState()

    mask1, mask2 = individual.mask1.copy(), individual.mask2.copy()

    # 1. Stochastic connection flips (topology exploration)
    mask1, mask2 = _flip_connections(mask1, mask2, p_flip, rng)

    # 2. Network growth (add one dead neuron)
    if rng.rand() < p_add_neuron:
        mask1, mask2 = _add_neuron(mask1, mask2, rng)

    # 3. Network pruning (remove one live neuron)
    if rng.rand() < p_remove_neuron:
        mask1, mask2 = _remove_neuron(mask1, mask2, rng)

    # Safety: ensure at least one active computation path
    mask1, mask2 = _ensure_valid_path(mask1, mask2, rng)

    return Individual(mask1, mask2)


# ============================================================================
# Model building, warm-start, training, evaluation
# ============================================================================

def build_model(individual: Individual,
                n_input: int, n_hidden_max: int, n_output: int,
                task: str,
                device: torch.device) -> InterpretableNN:
    """Construct a fresh InterpretableNN from an individual's masks."""
    return InterpretableNN(
        n_input, n_hidden_max, n_output,
        individual.mask1, individual.mask2, task
    ).to(device)


def warm_start_model(parent: Individual,
                      child: Individual,
                      n_input: int, n_hidden_max: int, n_output: int,
                      task: str,
                      device: torch.device) -> InterpretableNN:
    """
    Build the child's model and initialise it from the parent's weights.

    Lamarckian warm-start strategy
    --------------------------------
    Connections that survive the mutation (present in both parent and child)
    inherit their trained weight values → learning is not discarded.
    Connections that are new in the child (added by mutation) receive a small
    random initialisation ±0.1 so they do not dominate the output immediately.
    Connections removed by mutation are simply absent from the child mask and
    never accessed.

    This warm-start means offspring typically need far fewer training epochs
    to recover a good loss compared to a fresh random initialisation.
    """
    child_model = build_model(child, n_input, n_hidden_max, n_output, task, device)

    if parent.model is None:
        return child_model

    with torch.no_grad():
        # --- Layer 1 weights ---
        # Start from parent W1 (covers surviving + removed connections)
        child_model.W1.data.copy_(parent.model.W1.data)
        child_model.b1.data.copy_(parent.model.b1.data)

        # New connections in child (added by mutation) → small random values
        new_in_child1 = (~parent.mask1.astype(bool)) & (child.mask1.astype(bool))
        if new_in_child1.any():
            idx_t = torch.tensor(new_in_child1, dtype=torch.bool, device=device)
            n_new = int(new_in_child1.sum())
            child_model.W1.data[idx_t] = torch.empty(n_new, device=device).uniform_(-0.1, 0.1)

        # --- Layer 2 weights ---
        child_model.W2.data.copy_(parent.model.W2.data)
        child_model.b2.data.copy_(parent.model.b2.data)

        new_in_child2 = (~parent.mask2.astype(bool)) & (child.mask2.astype(bool))
        if new_in_child2.any():
            idx_t = torch.tensor(new_in_child2, dtype=torch.bool, device=device)
            n_new = int(new_in_child2.sum())
            child_model.W2.data[idx_t] = torch.empty(n_new, device=device).uniform_(-0.1, 0.1)

    return child_model


def train_model(model: InterpretableNN,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                n_epochs: int,
                lr: float = 1e-3,
                weight_decay: float = 1e-4) -> float:
    """
    Full-batch Adam training.

    The binary masks are buffers (not parameters) so they receive no gradients
    and remain fixed throughout training.  Only W1, b1, W2, b2 are updated,
    but the gradient for a masked weight W[i,j] where mask[i,j]=0 is always
    zero — those weights never change in practice.

    Returns the loss value from the final epoch.
    """
    model.train()
    criterion = nn.BCELoss() if model.task == 'classification' else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    final_loss = float('inf')
    for _ in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return final_loss


def evaluate_individual(individual: Individual,
                         X_val: torch.Tensor,
                         y_val: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute the three fitness objectives for a trained individual.

    Returns
    -------
    (val_loss, complexity, active_features)
      val_loss        – BCE or MSE on the validation set (float)
      complexity      – active_connections + active_neurons  (float)
      active_features – number of input features with ≥1 active connection (float)
    """
    model = individual.model
    model.eval()
    criterion = nn.BCELoss() if model.task == 'classification' else nn.MSELoss()

    with torch.no_grad():
        pred     = model(X_val)
        val_loss = criterion(pred, y_val).item()

    n_conn    = model.get_active_connections()
    n_neurons = model.get_active_neurons()
    n_feats   = model.get_active_features()

    return (val_loss, float(n_conn + n_neurons), float(n_feats))


# ============================================================================
# NSGA-II Pareto selection
# ============================================================================

def dominates(f1: Tuple[float, ...], f2: Tuple[float, ...]) -> bool:
    """
    Return True iff solution f1 Pareto-dominates f2 (minimisation).

    f1 dominates f2 iff:
      • f1 is no worse on ALL objectives  (∀i: f1[i] ≤ f2[i])
      • f1 is strictly better on at least one objective  (∃i: f1[i] < f2[i])
    """
    at_least_one_strictly_better = False
    for a, b in zip(f1, f2):
        if a > b:
            return False                  # f1 is worse on this axis → cannot dominate
        if a < b:
            at_least_one_strictly_better = True
    return at_least_one_strictly_better


def non_dominated_sort(population: List[Individual]) -> List[List[int]]:
    """
    NSGA-II non-dominated sorting.

    Partitions the population into Pareto fronts F0 ⊂ F1 ⊂ F2 … where F0
    is the Pareto front (no other solution dominates any member of F0).

    Algorithm (Deb et al. 2002)
    ---------------------------
    For each pair (i, j):
      If i dominates j:  add j to i's domination set; increment j's count.

    F0 = { i : count[i] == 0 }.
    For every subsequent front: decrement counts of dominated solutions;
    those reaching 0 form the next front.

    Time complexity: O(M · N²) where M = n_objectives, N = population size.

    Returns
    -------
    fronts : list of lists of population indices, ordered by rank.
    """
    n = len(population)
    dominated_by    = [[] for _ in range(n)]   # individuals that i strictly dominates
    domination_count = np.zeros(n, dtype=int)  # how many individuals dominate i

    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = population[i].fitness, population[j].fitness
            if dominates(fi, fj):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(fj, fi):
                dominated_by[j].append(i)
                domination_count[i] += 1

    fronts: List[List[int]] = []
    current_front = [i for i in range(n) if domination_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front: List[int] = []
        for i in current_front:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def crowding_distance_assignment(front_indices: List[int],
                                  population: List[Individual]) -> None:
    """
    Assign crowding distances to all individuals in one Pareto front.

    Crowding distance approximates the perimeter of the cuboid formed by
    a solution's nearest neighbours in objective space.  Large distance =
    solution is in a sparsely-populated region → good for diversity.

    Boundary solutions (smallest and largest on any axis) get infinite
    distance to ensure they are always preserved.

    Normalised by the objective range so all objectives are commensurate.
    """
    n = len(front_indices)
    if n == 0:
        return

    for i in front_indices:
        population[i].crowding_distance = 0.0

    if n <= 2:
        for i in front_indices:
            population[i].crowding_distance = float('inf')
        return

    n_obj = len(population[front_indices[0]].fitness)

    for obj in range(n_obj):
        # Sort this front by objective `obj`
        sorted_idx = sorted(front_indices,
                             key=lambda i: population[i].fitness[obj])

        # Boundary solutions get ∞
        population[sorted_idx[0]].crowding_distance  = float('inf')
        population[sorted_idx[-1]].crowding_distance = float('inf')

        obj_min   = population[sorted_idx[0]].fitness[obj]
        obj_max   = population[sorted_idx[-1]].fitness[obj]
        obj_range = (obj_max - obj_min) if obj_max != obj_min else 1.0

        for k in range(1, n - 1):
            delta = (population[sorted_idx[k + 1]].fitness[obj] -
                     population[sorted_idx[k - 1]].fitness[obj])
            population[sorted_idx[k]].crowding_distance += delta / obj_range


def select_next_generation(combined_population: List[Individual],
                            pop_size: int) -> List[Individual]:
    """
    NSGA-II survivor selection from the combined parent + offspring pool.

    Steps
    -----
    1. Non-dominated sort → fronts F0, F1, F2, …
    2. Assign rank attribute to every individual.
    3. Fill the new population by adding entire fronts in order.
    4. For the last partial front: compute crowding distances and keep
       the `remaining` individuals with the HIGHEST crowding distance
       (most isolated = most diverse).

    This guarantees:
      (a) Elitism: no non-dominated solution is ever discarded if room exists.
      (b) Diversity: crowding distance breaks rank ties, maintaining spread
          across the Pareto front rather than converging to a single point.
    """
    fronts = non_dominated_sort(combined_population)

    # Assign rank to all individuals
    for rank, front in enumerate(fronts):
        for idx in front:
            combined_population[idx].rank = rank

    new_pop_indices: List[int] = []

    for front in fronts:
        crowding_distance_assignment(front, combined_population)

        if len(new_pop_indices) + len(front) <= pop_size:
            new_pop_indices.extend(front)
        else:
            # Trim last partial front by crowding distance (descending)
            remaining = pop_size - len(new_pop_indices)
            trimmed = sorted(front,
                             key=lambda i: -combined_population[i].crowding_distance)
            new_pop_indices.extend(trimmed[:remaining])
            break

    return [combined_population[i] for i in new_pop_indices]


# ============================================================================
# LHFiD selection wrapper
# ============================================================================
# The NSGA-II functions above are kept for reference / fallback.
# The functions below bridge the LHFiD math (lhfid.py) with the Individual
# objects used throughout this module.

def _global_non_dominated(population: List[Individual]) -> List[Individual]:
    """
    Return the globally non-dominated subset of the population.
    Used only for final Pareto-front extraction — NOT for per-generation selection.
    """
    fits     = np.array([ind.fitness for ind in population], dtype=float)
    N        = len(population)
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if np.all(fits[j] <= fits[i]) and np.any(fits[j] < fits[i]):
                dominated[i] = True
                break
    return [population[i] for i in range(N) if not dominated[i]]


# ============================================================================
# Main evolutionary loop
# ============================================================================

def evolve(data: Dict[str, Any],
           pop_size: int = 24,
           n_generations: int = 40,
           n_hidden_max: int = 20,
           train_epochs_init: int = 200,
           train_epochs_offspring: int = 80,
           lr: float = 1e-3,
           weight_decay: float = 1e-4,
           p_flip: float = 0.04,
           p_add_neuron: float = 0.15,
           p_remove_neuron: float = 0.15,
           n_ref_partitions: int = 5,
           method: str = 'lhfid',
           seed: int = 42,
           verbose: bool = True,
           device: Optional[torch.device] = None
           ) -> Tuple[List[Individual], List[Dict]]:
    """
    Run the multi-objective evolutionary algorithm.

    Parameters
    ----------
    data                   : dict from load_breast_cancer / load_boston_housing
    pop_size               : constant population size P
    n_generations          : MAXIMUM number of generations (LHFiD may stop early
                             via stabilisation-based termination)
    n_hidden_max           : maximum hidden-layer width
    train_epochs_init      : full training epochs for initial population
    train_epochs_offspring : brief training epochs for each offspring
    lr, weight_decay       : Adam optimiser hyper-parameters
    p_flip                 : per-connection bit-flip probability
    p_add_neuron           : probability of add-neuron operator per mutation
    p_remove_neuron        : probability of remove-neuron operator per mutation
    n_ref_partitions       : Das-Dennis partition count (controls number of
                             reference directions; 5 → 21 vectors for 3 objectives)
                             Only used when method='lhfid'.
    method                 : 'lhfid' (default) or 'nsga2'
    seed                   : random seed for reproducibility
    verbose                : print per-generation progress
    device                 : torch.device (defaults to CUDA if available)

    Returns
    -------
    final_population : list[Individual] — the full population after the last
                       generation, with rank attribute set.
    history          : list[dict] — per-generation statistics including
                       'terminated_early' flag in the last entry.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    n_input  = data['n_features']
    n_output = data['n_outputs']
    task     = data['task']
    n_obj    = 3                         # loss, complexity, active_features

    # Move tensors to device once
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_val   = data['X_val'].to(device)
    y_val   = data['y_val'].to(device)

    # ── LHFiD state (only when method='lhfid') ───────────────────────── #
    if method == 'lhfid':
        ref_dirs = generate_reference_directions(n_obj, n_ref_partitions)
        n_refs   = len(ref_dirs)
        tracker  = StabilizationTracker(n_refs)
        if verbose:
            print(f"  device={device} | task={task} | n_input={n_input} "
                  f"| n_hidden_max={n_hidden_max} | pop={pop_size} "
                  f"| max_gen={n_generations} | ref_dirs={n_refs} | method=LHFiD")
    else:
        ref_dirs = None
        if verbose:
            print(f"  device={device} | task={task} | n_input={n_input} "
                  f"| n_hidden_max={n_hidden_max} | pop={pop_size} "
                  f"| max_gen={n_generations} | method=NSGA-II")

    # ------------------------------------------------------------------
    # Step 1: Initialise and fully train the initial population
    # ------------------------------------------------------------------
    population = initialize_population(pop_size, n_input, n_hidden_max, n_output,
                                        seed=seed)

    if verbose:
        print(f"\n  [init] Training {pop_size} networks × {train_epochs_init} epochs ...")

    for ind in population:
        ind.model   = build_model(ind, n_input, n_hidden_max, n_output, task, device)
        train_model(ind.model, X_train, y_train, train_epochs_init, lr, weight_decay)
        ind.fitness = evaluate_individual(ind, X_val, y_val)

    # Running ideal point — updated every generation (LHFiD only)
    fits_arr = np.array([ind.fitness for ind in population], dtype=float)

    if method == 'lhfid':
        ideal = fits_arr.min(axis=0)
        nadir = fits_arr.max(axis=0)
        nadir[nadir == ideal] += 1.0          # ensure non-zero range on all axes
        population = lhfid_survival_selection(population, pop_size,
                                               ref_dirs, ideal, nadir)
    else:
        ideal = None
        nadir = None
        population = select_next_generation(population, pop_size)

    history: List[Dict] = []

    def _log_stats(pop: List[Individual], gen: int,
                   early: bool = False) -> Dict:
        fits   = np.array([ind.fitness for ind in pop])
        n_p    = len(_global_non_dominated(pop))
        stats  = {
            'generation':       gen,
            'min_loss':         float(fits[:, 0].min()),
            'mean_loss':        float(fits[:, 0].mean()),
            'mean_complexity':  float(fits[:, 1].mean()),
            'mean_features':    float(fits[:, 2].mean()),
            'pareto_size':      n_p,
            'terminated_early': early,
        }
        return stats

    if verbose:
        s = _log_stats(population, 0)
        print(f"  [gen   0] min_loss={s['min_loss']:.4f}  "
              f"mean_complexity={s['mean_complexity']:.1f}  "
              f"mean_features={s['mean_features']:.1f}  "
              f"pareto_size={s['pareto_size']}")

    # ------------------------------------------------------------------
    # Steps 2–N: Evolutionary loop
    # ------------------------------------------------------------------
    terminated_early = False

    for gen in range(1, n_generations + 1):

        # --- Mutate each parent → one offspring ---
        offspring: List[Individual] = []
        for parent in population:
            child       = mutate(parent, p_flip, p_add_neuron, p_remove_neuron, rng)
            child.model = warm_start_model(parent, child,
                                            n_input, n_hidden_max, n_output,
                                            task, device)
            train_model(child.model, X_train, y_train,
                        train_epochs_offspring, lr, weight_decay)
            child.fitness = evaluate_individual(child, X_val, y_val)
            offspring.append(child)

        combined = population + offspring
        all_fits = np.array([ind.fitness for ind in combined], dtype=float)

        if method == 'lhfid':
            # --- Update ideal point ---
            ideal = np.minimum(ideal, all_fits.min(axis=0))

            # --- Adaptive nadir: triggered once after mild stabilisation ---
            if tracker.should_update_nadir():
                nadir = compute_nadir_point(all_fits, ideal, nadir)

            # --- LHFiD selection ---
            population = lhfid_survival_selection(combined, pop_size,
                                                   ref_dirs, ideal, nadir)

            # --- Update stabilisation tracker ---
            from lhfid import associate_to_reference_vectors, normalize_objectives
            cur_fits  = np.array([ind.fitness for ind in population], dtype=float)
            norm_fits = normalize_objectives(cur_fits.copy(), ideal.copy(), nadir.copy())
            assignments, distances = associate_to_reference_vectors(norm_fits, ref_dirs)
            tracker.update(assignments, distances)

        else:
            # --- NSGA-II selection ---
            population = select_next_generation(combined, pop_size)

        stats = _log_stats(population, gen)
        history.append(stats)

        if verbose and (gen % 5 == 0 or gen == 1):
            if method == 'lhfid':
                stab_info = (f"  mild={tracker.mild_count()}/{tracker.n_s_mild}"
                             f"  strict={tracker.strict_count()}/{tracker.n_s_strict}")
            else:
                stab_info = ''
            print(f"  [gen {gen:3d}] min_loss={stats['min_loss']:.4f}  "
                  f"mean_complexity={stats['mean_complexity']:.1f}  "
                  f"mean_features={stats['mean_features']:.1f}  "
                  f"pareto_size={stats['pareto_size']}{stab_info}")

        # --- LHFiD early termination ---
        if method == 'lhfid' and tracker.should_terminate():
            terminated_early = True
            if verbose:
                print(f"\n  [LHFiD] Strict stabilisation reached at gen {gen} "
                      f"— terminating early (max was {n_generations}).")
            history[-1]['terminated_early'] = True
            break

    return population, history


def get_pareto_front(population: List[Individual]) -> List[Individual]:
    """
    Return the globally non-dominated (Pareto-front) individuals.

    Uses a global dominance check rather than the rank attribute because
    LHFiD's localised selection assigns rank=0 to all survivors —
    the global front must be extracted explicitly.
    """
    return _global_non_dominated(population)
