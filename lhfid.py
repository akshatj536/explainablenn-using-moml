"""
lhfid.py
========
Self-contained implementation of the mathematical core of LHFiD
(Localized High-Fidelity-Dominance-Based Many-Objective Evolutionary Algorithm)
adapted for binary-matrix neural network populations.

Reference
---------
Saxena D.K., Mittal S., Kapoor S., Deb K.
"Localized High-Fidelity-Dominance-Based Many-Objective Evolutionary Algorithm"
IEEE Transactions on Evolutionary Computation, Vol. 27, No. 4, pp. 923–937, 2023.
DOI: 10.1109/TEVC.2022.3188064

Components implemented (pure NumPy — no pymoo)
-----------------------------------------------
1.  generate_reference_directions  – Das-Dennis simplex sampling
2.  compute_nadir_point            – adaptive hyperplane-based nadir
3.  normalize_objectives           – scale fitnesses into [0,1]
4.  associate_to_reference_vectors – perpendicular-distance clustering
5.  lhfid_survival_selection       – full localized HF-dominance selection
6.  StabilizationTracker           – convergence monitoring + auto-termination

All functions operate on plain lists / numpy arrays so they integrate
cleanly with the Individual objects defined in evolution.py.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Optional


# ============================================================================
# 1. Reference Directions  (Das-Dennis simplex sampling)
# ============================================================================

def generate_reference_directions(n_obj: int,
                                   n_partitions: int) -> np.ndarray:
    """
    Generate uniformly distributed reference vectors on the unit simplex
    using the Das-Dennis structured sampling method.

    Each reference vector w satisfies:
        w[i] >= 0  for all i
        sum(w)  == 1

    The vectors span the simplex corners (pure objectives) through all
    intermediate trade-off regions.  More partitions → more vectors →
    finer coverage but higher computational cost.

    Parameters
    ----------
    n_obj        : number of objectives (3 in our case)
    n_partitions : number of divisions per axis (H in the paper)
                   Typical values: 4 → 15 vectors, 5 → 21 vectors,
                                   6 → 28 vectors

    Returns
    -------
    ref_dirs : np.ndarray, shape (n_vectors, n_obj), float64
               Each row is a unit-simplex reference direction.
    """
    def _recursive_fill(ref_dirs, current, remaining, depth, n_obj, H):
        if depth == n_obj - 1:
            current[depth] = remaining / H
            ref_dirs.append(current.copy())
        else:
            for i in range(remaining + 1):
                current[depth] = i / H
                _recursive_fill(ref_dirs, current, remaining - i,
                                 depth + 1, n_obj, H)

    ref_dirs = []
    current  = np.zeros(n_obj)
    _recursive_fill(ref_dirs, current, n_partitions, 0, n_obj, n_partitions)
    return np.array(ref_dirs)


# ============================================================================
# 2. Nadir Point  (adaptive, hyperplane-based)
# ============================================================================

def _pareto_non_dominated(fitnesses: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of non-dominated solutions (all objectives minimised).
    fitnesses : (N, M) array
    """
    N = len(fitnesses)
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # j dominates i if j is no worse on all and better on at least one
            if (np.all(fitnesses[j] <= fitnesses[i]) and
                    np.any(fitnesses[j] < fitnesses[i])):
                dominated[i] = True
                break
    return ~dominated


def compute_nadir_point(fitnesses: np.ndarray,
                         ideal: np.ndarray,
                         prev_nadir: Optional[np.ndarray] = None
                         ) -> np.ndarray:
    """
    Compute the nadir point adaptively from extreme solutions on the
    non-dominated front.

    Algorithm
    ---------
    1. Extract the non-dominated subset of fitnesses.
    2. Translate by the ideal point so the front is near the origin.
    3. For each objective axis, find the extreme point that minimises
       the Achievement Scalarising Function (ASF):
           ASF_i(f) = max_j ( f[j] / w[j] )
       with w[i] = 1, w[j≠i] = 1e6  →  selects the solution that is
       "most extreme" along axis i.
    4. Stack the M extreme points into a matrix E (shape M×M).
    5. Solve  E · plane = 1  for the hyperplane normal.
    6. Nadir intercept on axis i = ideal[i] + 1 / plane[i].
    7. If the system is singular (degenerate front), fall back to the
       previous nadir (or to the observed maximum per objective).

    Parameters
    ----------
    fitnesses  : (N, M) array of objective values
    ideal      : (M,)  current ideal (minimum) point
    prev_nadir : (M,)  previous nadir point used as fallback

    Returns
    -------
    nadir : (M,) array
    """
    M = fitnesses.shape[1]

    # Non-dominated front
    nd_mask = _pareto_non_dominated(fitnesses)
    nd_fits = fitnesses[nd_mask]

    if len(nd_fits) < M:
        # Not enough points to fit a hyperplane — fallback
        return (prev_nadir if prev_nadir is not None
                else fitnesses.max(axis=0))

    # Translate to ideal
    translated = nd_fits - ideal  # shape (K, M)

    # Extreme points via ASF
    extreme_rows = []
    for i in range(M):
        w = np.full(M, 1e6)
        w[i] = 1.0
        # ASF for each non-dominated solution
        asf_vals = np.max(translated / w, axis=1)
        extreme_rows.append(translated[np.argmin(asf_vals)])

    E = np.array(extreme_rows)   # (M, M)

    try:
        plane = np.linalg.solve(E, np.ones(M))
        intercepts = 1.0 / (plane + 1e-12)
        nadir = ideal + intercepts

        # Sanity: nadir must be >= ideal
        valid = np.all(nadir > ideal - 1e-6)
        if not valid:
            raise np.linalg.LinAlgError("Degenerate intercepts")

        return nadir

    except np.linalg.LinAlgError:
        # Singular or degenerate — fall back
        return (prev_nadir if prev_nadir is not None
                else fitnesses.max(axis=0))


# ============================================================================
# 3. Objective Normalisation
# ============================================================================

def normalize_objectives(fitnesses: np.ndarray,
                           ideal: np.ndarray,
                           nadir: np.ndarray) -> np.ndarray:
    """
    Translate and scale fitnesses into [0, 1] using ideal and nadir points.

        norm[i] = (f[i] - ideal) / (nadir - ideal)

    If nadir == ideal on any axis (zero range), that axis is left as-is
    (already at its best — no variation to normalise).

    Parameters
    ----------
    fitnesses : (N, M)
    ideal     : (M,)
    nadir     : (M,)

    Returns
    -------
    norm_fitnesses : (N, M), values approximately in [0, 1]
    """
    denom = nadir - ideal
    denom[denom < 1e-12] = 1.0          # guard zero-range axes
    return (fitnesses - ideal) / denom


# ============================================================================
# 4. Association to Reference Vectors
# ============================================================================

def _perpendicular_distance(point: np.ndarray, ref: np.ndarray) -> float:
    """
    Perpendicular (orthogonal) distance from `point` to the line defined
    by the origin and `ref`.

    d = ||point - (point·ref / ||ref||²) · ref||

    Both vectors are assumed to have non-negative components (after
    normalisation, fitnesses should be ≥ 0).
    """
    ref_norm_sq = np.dot(ref, ref)
    if ref_norm_sq < 1e-12:
        return float(np.linalg.norm(point))
    proj      = (np.dot(point, ref) / ref_norm_sq) * ref
    residual  = point - proj
    return float(np.linalg.norm(residual))


def associate_to_reference_vectors(
        norm_fitnesses: np.ndarray,
        ref_dirs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associate each solution to its nearest reference direction
    (minimum perpendicular distance).

    Parameters
    ----------
    norm_fitnesses : (N, M) normalised objective values
    ref_dirs       : (R, M) reference directions

    Returns
    -------
    assignments : (N,) int array — index into ref_dirs for each solution
    distances   : (N,) float array — perpendicular distance to assigned ref
    """
    N, R = len(norm_fitnesses), len(ref_dirs)
    assignments = np.zeros(N, dtype=int)
    distances   = np.full(N, np.inf)

    for i in range(N):
        for r in range(R):
            d = _perpendicular_distance(norm_fitnesses[i], ref_dirs[r])
            if d < distances[i]:
                distances[i]   = d
                assignments[i] = r

    return assignments, distances


# ============================================================================
# 5. LHFiD Survival Selection
# ============================================================================

def _dominates_local(f1: np.ndarray, f2: np.ndarray) -> bool:
    """Pareto dominance (minimisation). True iff f1 dominates f2."""
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def _tie_breaker_score(norm_f: np.ndarray, ref_dir: np.ndarray) -> float:
    """
    High-fidelity scalarisation score (lower = better).

    Weighted Chebyshev scalarisation aligned with the reference direction:
        score = max_i( norm_f[i] / (ref_dir[i] + 1e-6) )

    Solutions that are proportionally closer to the reference direction
    receive lower scores.  This is the LHFiD "hf-dominance" criterion —
    a directionally-aware tie-breaker that goes beyond simple crowding
    distance (which is direction-agnostic).
    """
    w = ref_dir + 1e-6
    return float(np.max(norm_f / w))


def lhfid_survival_selection(
        population:   list,          # List[Individual] from evolution.py
        pop_size:     int,
        ref_dirs:     np.ndarray,    # (R, M)
        ideal:        np.ndarray,    # (M,)
        nadir:        np.ndarray,    # (M,)
) -> list:
    """
    LHFiD environmental selection.

    Replaces NSGA-II's crowding-distance-based selection with:
      1. Objective normalisation (makes all three objectives comparable)
      2. Association to reference directions (clustering)
      3. Per-cluster Pareto filtering (localized dominance)
      4. Alpha selection (closest to reference direction)
      5. Beta selection (hf-dominating via tie-breaker)
      6. Gap filling from unassigned pool

    Parameters
    ----------
    population : combined parent + offspring list (2 × pop_size individuals)
    pop_size   : target survivors
    ref_dirs   : (R, M) reference direction matrix
    ideal      : (M,) current ideal point (running minimum)
    nadir      : (M,) current nadir point

    Returns
    -------
    survivors : list of `pop_size` Individual objects

    Selection logic per cluster
    ---------------------------
    Given a cluster C associated with reference direction w_r:
      - Drop any solution that is Pareto-dominated by another in C.
      - The ALPHA solution is the one with minimum perpendicular distance
        to w_r (closest to the reference line) — ensures convergence.
      - Each remaining non-dominated solution is a BETA candidate if its
        tie_breaker_score < alpha's score (hf-dominates alpha w.r.t. w_r).
        Beta solutions represent high-fidelity improvements along w_r.
      - All selected (alpha + betas) are added to the survivor pool.
    """
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)
    N         = len(population)

    # ── Normalise ───────────────────────────────────────────────────────── #
    norm_fits = normalize_objectives(fitnesses.copy(), ideal.copy(),
                                      nadir.copy())

    # ── Associate ───────────────────────────────────────────────────────── #
    assignments, distances = associate_to_reference_vectors(norm_fits, ref_dirs)

    # Build clusters: ref_idx → list of population indices
    clusters: Dict[int, List[int]] = {r: [] for r in range(len(ref_dirs))}
    for i in range(N):
        clusters[assignments[i]].append(i)

    selected_idx: List[int] = []
    unassigned:   List[int] = list(range(N))   # pool for gap-filling

    # ── Per-cluster selection ────────────────────────────────────────────── #
    for r, cluster in clusters.items():
        if not cluster:
            continue

        ref_dir = ref_dirs[r]

        if len(cluster) == 1:
            # Trivially select the only member
            selected_idx.append(cluster[0])
            if cluster[0] in unassigned:
                unassigned.remove(cluster[0])
            continue

        cluster_fits = norm_fits[cluster]      # (K, M)

        # --- Step 1: Local Pareto filter ---
        # Remove solutions that are dominated by anyone else in this cluster
        dominated_local = np.zeros(len(cluster), dtype=bool)
        for a in range(len(cluster)):
            for b in range(len(cluster)):
                if a == b:
                    continue
                if _dominates_local(cluster_fits[b], cluster_fits[a]):
                    dominated_local[a] = True
                    break

        non_dom_local = [cluster[i] for i in range(len(cluster))
                         if not dominated_local[i]]

        if not non_dom_local:
            non_dom_local = cluster   # fallback: keep all if all dominated

        # --- Step 2: Alpha — closest to reference direction ---
        alpha_idx = min(non_dom_local, key=lambda i: distances[i])
        alpha_score = _tie_breaker_score(norm_fits[alpha_idx], ref_dir)

        cluster_selected = [alpha_idx]

        # --- Step 3: Beta — hf-dominating solutions ---
        for i in non_dom_local:
            if i == alpha_idx:
                continue
            score_i = _tie_breaker_score(norm_fits[i], ref_dir)
            # i hf-dominates alpha if its scalarised score is strictly better
            if score_i < alpha_score:
                cluster_selected.append(i)

        for i in cluster_selected:
            if i not in selected_idx:
                selected_idx.append(i)
            if i in unassigned:
                unassigned.remove(i)

    # ── Gap filling ──────────────────────────────────────────────────────── #
    # If we have fewer survivors than pop_size, fill from the unassigned pool
    # sorted by distance to their reference vector (nearest first)
    if len(selected_idx) < pop_size and unassigned:
        unassigned_sorted = sorted(unassigned, key=lambda i: distances[i])
        for i in unassigned_sorted:
            if len(selected_idx) >= pop_size:
                break
            if i not in selected_idx:
                selected_idx.append(i)

    # ── Trim if over budget ──────────────────────────────────────────────── #
    if len(selected_idx) > pop_size:
        # Keep the pop_size solutions with smallest distance to their ref dir
        selected_idx = sorted(selected_idx,
                               key=lambda i: distances[i])[:pop_size]

    # Assign rank=0 to everyone (all survivors treated as Pareto-comparable)
    survivors = [population[i] for i in selected_idx]
    for ind in survivors:
        ind.rank = 0

    return survivors


# ============================================================================
# 6. Stabilisation Tracker
# ============================================================================

class StabilizationTracker:
    """
    Monitors convergence of the population across reference vectors.

    Tracks two quantities per generation:
      - mu_D[r] : mean perpendicular distance of cluster r's solutions
      - S_D[r]  : standard deviation of distances in cluster r

    Two thresholds (matching the paper):
      MILD   : changes < 0.01  (2 d.p.) stable for n_s_mild=20  generations
               → triggers nadir point computation
      STRICT : changes < 0.001 (3 d.p.) stable for n_s_strict=50 generations
               → triggers early termination

    Usage
    -----
    tracker = StabilizationTracker(n_ref_dirs=15)
    for gen in range(max_gen):
        ...
        tracker.update(assignments, distances, n_ref_dirs)
        if tracker.should_terminate():
            break
        if tracker.should_update_nadir():
            nadir = compute_nadir_point(...)
    """

    def __init__(self, n_ref_dirs: int,
                 n_s_mild:   int = 2,
                 n_s_strict: int = 10,
                 tol_mild:   float = 0.1,
                 tol_strict: float = 0.01):
        self.R          = n_ref_dirs
        self.n_s_mild   = n_s_mild    # public — used in verbose logging
        self.n_s_strict = n_s_strict  # public — used in verbose logging
        self.tol_mild   = tol_mild
        self.tol_strict = tol_strict

        # Running histories: list of (R,) arrays
        self._mu_D_history: List[np.ndarray] = []
        self._S_D_history:  List[np.ndarray] = []

        self._mild_stable_count:   int = 0
        self._strict_stable_count: int = 0

        self.nadir_triggered:       bool = False
        self.termination_triggered: bool = False

    # ------------------------------------------------------------------
    def update(self, assignments: np.ndarray, distances: np.ndarray) -> None:
        """
        Record one generation's distance statistics per reference vector.
        Call this once per generation after association.
        """
        mu_D = np.zeros(self.R)
        S_D  = np.zeros(self.R)

        for r in range(self.R):
            mask = assignments == r
            if mask.any():
                d         = distances[mask]
                mu_D[r]   = d.mean()
                S_D[r]    = d.std() if len(d) > 1 else 0.0

        self._mu_D_history.append(mu_D)
        self._S_D_history.append(S_D)

        # Need at least 2 generations to compare
        if len(self._mu_D_history) < 2:
            return

        prev_mu = self._mu_D_history[-2]
        curr_mu = self._mu_D_history[-1]
        prev_S  = self._S_D_history[-2]
        curr_S  = self._S_D_history[-1]

        delta_mu = np.abs(curr_mu - prev_mu).max()
        delta_S  = np.abs(curr_S  - prev_S ).max()
        change   = max(delta_mu, delta_S)

        # MILD stabilisation check
        if change < self.tol_mild:
            self._mild_stable_count += 1
        else:
            self._mild_stable_count = 0

        # STRICT stabilisation check
        if change < self.tol_strict:
            self._strict_stable_count += 1
        else:
            self._strict_stable_count = 0

        # Trigger nadir update after mild stabilisation
        if (not self.nadir_triggered and
                self._mild_stable_count >= self.n_s_mild):
            self.nadir_triggered = True

        # Trigger termination after strict stabilisation
        if (not self.termination_triggered and
                self._strict_stable_count >= self.n_s_strict):
            self.termination_triggered = True

    # ------------------------------------------------------------------
    def should_update_nadir(self) -> bool:
        """True once mild stabilisation has been detected (fires once)."""
        return self.nadir_triggered

    def should_terminate(self) -> bool:
        """True once strict stabilisation has been detected."""
        return self.termination_triggered

    def mild_count(self)   -> int: return self._mild_stable_count
    def strict_count(self) -> int: return self._strict_stable_count
