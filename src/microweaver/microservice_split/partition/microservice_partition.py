"""
Microservice Partition Optimizer (0-1 IQP linearized to MILP via OR-Tools CP-SAT).

Formulation
- Variables: x[i,k] in {0,1} (class i assigned to service k)
- Auxiliary variables: y[i,j,k] = x[i,k] & x[j,k] (linearization)
- Objective function (maximize):
    alpha * sum_k sum_{i<j} S_struc[i,j] * y[i,j,k]
  + beta  * sum_k sum_{i<j} S_sem[i,j]   * y[i,j,k]
  - gamma * sum_k sum_{i<j} C_run[i,j]   * (x[i,k] - y[i,j,k])   # cross-service coupling

Constraints
- Sum_k x[i,k] = 1 (each class assigned to exactly one service)
- Capacity: L_k <= sum_i s_i * x[i,k] <= U_k (if provided)
- Linearization: y[i,j,k] <= x[i,k]; y[i,j,k] <= x[j,k]; y[i,j,k] >= x[i,k] + x[j,k] - 1
- Must-link (hard constraint): for all k, x[i,k] == x[j,k]
- Cannot-link (hard constraint): for all k, x[i,k] + x[j,k] <= 1

Notes
- CP-SAT is an integer solver; we scale all real-valued scores by integer SCALE and round.
- For large N, creating y for all pairs is O(N^2 K). Use pair_threshold to sparsify pairs.
"""
from __future__ import annotations

import os

from microweaver.microservice_split.config import MicroWeaverConfig
from microweaver.util.file_op import save_json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable, Any
import numpy as np
import torch
from ortools.sat.python import cp_model
from microweaver.microservice_split.config import PartitionConfig


@dataclass
class PartitionResult:
    assignments: List[int]  # length N, value in [0, K-1]
    objective_value: float
    solver_status: str
    stats: Dict[str, float]
    iteration: int = 0  # Current iteration count
    total_iterations: int = 3  # Total iterations
    agent_feedback: Optional[Dict] = None  # Agent optimization feedback


def _save_iteration_result(
        result: "PartitionResult",
        iteration: int,
        node_names: Optional[List[str]],
) -> None:
    """
    Save the partition result of current iteration to result directory.

    File named result_iter_{iteration}.json, content is Service-x -> [class names or indices].
    """
    try:
        base_result_path = MicroWeaverConfig.result_path
        result_dir = os.path.dirname(base_result_path)
        os.makedirs(result_dir, exist_ok=True)
        iter_path = os.path.join(result_dir, f"result_iter_{iteration}.json")

        groups: Dict[str, List[Any]] = {}
        for idx, svc in enumerate(result.assignments):
            if svc < 0:
                continue
            key = f"Service-{svc}"
            if key not in groups:
                groups[key] = []
            groups[key].append(node_names[idx] if node_names else idx)

        save_json(groups, iter_path)
        print(f"[Iteration {iteration}] Result saved to {iter_path}")
    except Exception as e:
        print(f"[Iteration {iteration}] Failed to save result to result directory: {e}")


def cosine_similarity(emb: torch.Tensor) -> np.ndarray:
    emb = torch.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1, eps=1e-8)
    sim = emb @ emb.t()
    return sim.cpu().numpy().astype(np.float64)


def _cosine_similarity_01(emb: torch.Tensor) -> np.ndarray:
    """
    Map embedding to cosine similarity matrix in [0,1] range:
    - First do L2 normalization
    - Then compute cosine similarity [-1,1]
    - Finally map to [0,1]
    """
    sim = cosine_similarity(emb)  # [-1,1]
    return (sim + 1.0) / 2.0


def build_structural_similarity(num_nodes: int,
                                edge_index: torch.Tensor,
                                weight: Optional[torch.Tensor] = None,
                                symmetric: bool = True) -> np.ndarray:
    """
    Build S_struc from directed edges. By default, we symmetrize to reward mutual cohesion.
    S_ij is in [0,1] range, normalized by max degree.
    
    Args:
        num_nodes: Number of nodes
        edge_index: [2, num_edges] Edge indices
        weight: [num_edges] Edge weights (optional), use these if provided
        symmetric: Whether to symmetrize similarity matrix
    
    Returns:
        [num_nodes, num_nodes] Structural similarity matrix
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_index.numel() > 0:
        ei = edge_index.cpu().numpy()
        if weight is None:
            for s, d in zip(ei[0], ei[1]):
                A[s, d] += 1.0
        else:
            w = weight.cpu().numpy()
            for idx in range(ei.shape[1]):
                s, d = int(ei[0, idx]), int(ei[1, idx])
                A[s, d] += float(w[idx])
    if symmetric:
        S = (A + A.T) / 2.0
    else:
        S = A
    # normalize to [0,1]
    mx = S.max()
    if mx > 0:
        S = S / mx
    return S


def build_runtime_coupling(num_nodes: int,
                           edge_index: torch.Tensor,
                           weight: Optional[torch.Tensor] = None) -> np.ndarray:
    """
    Build C_run from directed edges; higher values mean higher cost of crossing services.
    Default value: 1 per edge.
    
    Args:
        num_nodes: Number of nodes
        edge_index: [2, num_edges] Edge indices
        weight: [num_edges] Edge weights (optional), use these if provided
    
    Returns:
        [num_nodes, num_nodes] Runtime coupling matrix
    """
    C = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_index.numel() > 0:
        ei = edge_index.cpu().numpy()
        if weight is None:
            for s, d in zip(ei[0], ei[1]):
                C[s, d] += 1.0
        else:
            w = weight.cpu().numpy()
            for idx in range(ei.shape[1]):
                s, d = int(ei[0, idx]), int(ei[1, idx])
                C[s, d] += float(w[idx])
    # normalize to [0,1]
    mx = C.max()
    if mx > 0:
        C = C / mx
    return C


def _sparsify_pairs(weights: np.ndarray, quantile_threshold: float) -> List[Tuple[int, int, float]]:
    """
    Quantile sparsification (alternative to fixed threshold)
    :param quantile_threshold: Quantile (0~1), e.g., 0.9 -> keep top 10% high-affinity class pairs
    """
    print("Sparsification threshold:", quantile_threshold)
    N = weights.shape[0]
    # 1. Collect all i<j class pair weights (exclude diagonal)
    all_weights = []
    for i in range(N):
        for j in range(i + 1, N):
            all_weights.append(float(weights[i, j]))

    # 2. Compute quantile threshold (core: adapt to different data distributions)
    if not all_weights:
        return []
    threshold = np.quantile(all_weights, quantile_threshold)

    # 3. Filter high-affinity class pairs
    pairs: List[Tuple[int, int, float]] = []
    for i in range(N):
        for j in range(i + 1, N):
            weight_val = float(weights[i, j])
            if weight_val >= threshold:  # No need for abs, as weights are non-negative
                pairs.append((i, j, weight_val))

    # Print log for tuning
    total = len(all_weights)
    kept = len(pairs)
    print(f"Sparsification: total pairs {total} -> kept {kept}, compression ratio {kept / total:.2%} (quantile threshold={quantile_threshold})")
    return pairs


def _debug_print_objective_components(
        assignments: List[int],
        S_struc: np.ndarray,
        S_sem: np.ndarray,
        C_run: np.ndarray,
        config: "PartitionConfig",
) -> None:
    """
    Based on final assignments, approximately compute scores for each objective term according to the "original definition formula", and print them for debugging weights.

    Here we don't strictly reproduce the sparsification and linearization details in MILP, but follow the intuitive objectives:
      - Structural cohesion: sum of S_struc[i,j] for all (i<j) within same service
      - Semantic cohesion: sum of S_sem[i,j] for all (i<j) within same service
      - Runtime cross-service penalty: sum of C_run[i,j] for all cross-service directed pairs (i!=j, svc[i]!=svc[j])

    This makes it easier to intuitively compare the relative magnitudes of alpha / beta / gamma.
    """
    N = len(assignments)
    alpha, beta, gamma = config.alpha, config.beta, config.gamma

    intra_struc = 0.0
    intra_sem = 0.0
    cross_run = 0.0

    # Structural/semantic cohesion for same-service pairs
    for i in range(N):
        si = assignments[i]
        if si < 0:
            continue
        for j in range(i + 1, N):
            sj = assignments[j]
            if sj < 0:
                continue
            if si == sj:
                intra_struc += float(S_struc[i, j])
                intra_sem += float(S_sem[i, j])

    # Cross-service runtime penalty (directed)
    for i in range(N):
        si = assignments[i]
        if si < 0:
            continue
        for j in range(N):
            if i == j:
                continue
            sj = assignments[j]
            if sj < 0:
                continue
            if si != sj:
                cross_run += float(C_run[i, j])

    weighted_struc = alpha * intra_struc
    weighted_sem = beta * intra_sem
    weighted_run = -gamma * cross_run
    total_approx = weighted_struc + weighted_sem + weighted_run

    print("\n[Debug] Objective function components (approximate calculation based on assignments)")
    print("------------------------------------------------------------")
    print(f"  Number of nodes N     : {N}")
    print(f"  alpha (structural cohesion weight): {alpha}")
    print(f"  beta  (semantic cohesion weight)  : {beta}")
    print(f"  gamma (cross-service penalty weight): {gamma}")
    print("------------------------------------------------------------")
    print(f"  Structural cohesion raw     : {intra_struc:.6f}")
    print(f"  Structural cohesion weighted: {weighted_struc:.6f}")
    print(f"  Semantic cohesion raw       : {intra_sem:.6f}")
    print(f"  Semantic cohesion weighted  : {weighted_sem:.6f}")
    print(f"  Cross-service penalty raw   : {cross_run:.6f}")
    print(f"  Cross-service penalty weighted: {weighted_run:.6f}")
    print("------------------------------------------------------------")
    print(f"  Approximate total objective (weighted sum): {total_approx:.6f}\n")


def optimize_partition(S_struc: np.ndarray,
                       S_sem: np.ndarray,
                       C_run: np.ndarray,
                       must_link: Optional[List[Tuple[int, int]]] = None,
                       cannot_link: Optional[List[Tuple[int, int]]] = None,
                       config: PartitionConfig = None,
                       edge_index: Optional[torch.Tensor] = None) -> PartitionResult:
    """
    Solve microservice partition MILP via CP-SAT.
    """
    must_link = must_link or []
    cannot_link = cannot_link or []
    K = config.num_communities
    if edge_index is None:
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

    N = S_struc.shape[0]
    assert S_struc.shape == (N, N)
    assert S_sem.shape == (N, N)
    assert C_run.shape == (N, N)

    # Precompute pair weights for y[i,j,k] used in objective function
    # Use symmetric runtime coupling (merge-friendly) to help select informative pairs
    # and keep directional C_run in the separate cross-service penalty below.
    C_sym = (C_run + C_run.T) / 2.0
    pair_weight = config.alpha * S_struc + config.beta * S_sem + config.gamma * C_sym
    pair_list = _sparsify_pairs(pair_weight, config.pair_threshold)
    pair_set = {(i, j) for (i, j, _) in pair_list}

    model = cp_model.CpModel()

    # Variables x[i,k]
    x = [[model.NewBoolVar(f"x_{i}_{k}") for k in range(K)] for i in range(N)]

    # y[i,j,k] only used for selected pairs to reduce size
    # We will index them in dictionary y[(i,j,k)]
    y: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
    for (i, j, _) in pair_list:
        for k in range(K):
            y[(i, j, k)] = model.NewBoolVar(f"y_{i}_{j}_{k}")
            # Linearization
            model.Add(y[(i, j, k)] <= x[i][k])
            model.Add(y[(i, j, k)] <= x[j][k])
            model.Add(y[(i, j, k)] >= x[i][k] + x[j][k] - 1)

    # Each node assigned to exactly one service
    for i in range(N):
        model.Add(sum(x[i][k] for k in range(K)) == 1)

    # Must-link / Cannot-link (default hard constraints)
    for (i, j) in must_link:
        for k in range(K):
            model.Add(x[i][k] == x[j][k])
    for (i, j) in cannot_link:
        for k in range(K):
            model.Add(x[i][k] + x[j][k] <= 1)

    # Capacity constraints
    if config.size_lower is not None:
        for k in range(K):
            Lk = int(config.size_lower[k])
            model.Add(sum(x[i][k] for i in range(N)) >= Lk)
    if config.size_upper is not None:
        for k in range(K):
            Uk = int(config.size_upper[k])
            model.Add(sum(x[i][k] for i in range(N)) <= Uk)

    # Objective function construction (scaled integers)
    SCALE = config.scale
    objective_terms: List[cp_model.LinearExpr] = []

    # Cohesion terms: alpha*S_struc*y + beta*S_sem*y
    for (i, j, _) in pair_list:
        for k in range(K):
            coeff = int(round(SCALE * (config.alpha * S_struc[i, j] + config.beta * S_sem[i, j])))
            if coeff != 0:
                objective_terms.append(coeff * y[(i, j, k)])

    # Cross-service penalty: -gamma*C_run[i,j]*(x_i_k - y_ij_k)
    # Equivalent to -gamma*C_run*x_i_k  + gamma*C_run*y_ij_k
    # First term over ordered pairs (i,j), second term only for pairs we create y for.
    # Use ordered pairs to reflect directionality.
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            coeff_x = int(round(SCALE * (-config.gamma * C_run[i, j])))
            if coeff_x != 0:
                for k in range(K):
                    objective_terms.append(coeff_x * x[i][k])
            # y part (only when exists; use base=(min(i,j),max(i,j)) to avoid duplicates)
            base = (i, j) if i < j else (j, i)
            if base in pair_set:
                coeff_y = int(round(SCALE * (config.gamma * C_run[i, j])))
                if coeff_y != 0:
                    for k in range(K):
                        var = y.get((base[0], base[1], k))
                        if var is not None:
                            objective_terms.append(coeff_y * var)

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(config.time_limit_sec)
    solver.parameters.num_search_workers = int(config.num_cpu)
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    assignments = [-1] * N
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(N):
            for k in range(K):
                if solver.BooleanValue(x[i][k]):
                    assignments[i] = k
                    break

    # Objective value (rescaled)
    obj_value = solver.ObjectiveValue() / SCALE

    # If debugging needed, approximate decomposition of objective components by assignments
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        try:
            _debug_print_objective_components(assignments, S_struc, S_sem, C_run, config)
        except Exception as e:
            print(f"[Debug] Exception occurred while computing objective components: {e}")

    return PartitionResult(
        assignments=assignments,
        objective_value=obj_value,
        solver_status=solver.StatusName(status),
        stats={
            "num_pairs": len(pair_list),
            "N": N,
            "K": K,
        },
        iteration=0,
        total_iterations=1,
        agent_feedback=None,
    )


def _convert_assignments_to_partitions(assignments: List[int], num_nodes: int) -> Dict[int, List[int]]:
    """
    Convert assignment results to partition dictionary format.
    
    Args:
        assignments: [num_nodes] Service assignment for each node
        num_nodes: Total number of nodes
    
    Returns:
        {service_id: [node_ids]} Partition dictionary
    """
    partitions = {}
    for node_id, service_id in enumerate(assignments):
        if service_id not in partitions:
            partitions[service_id] = []
        partitions[service_id].append(node_id)
    return partitions


def _merge_constraints(
        must_link: Optional[List[Tuple[int, int]]],
        cannot_link: Optional[List[Tuple[int, int]]],
        agent_must_link: Optional[List[Tuple[int, int]]],
        agent_cannot_link: Optional[List[Tuple[int, int]]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Merge constraints from Agent with existing constraints.
    
    Args:
        must_link: Original must-link constraints
        cannot_link: Original cannot-link constraints
        agent_must_link: Agent-suggested must-link constraints
        agent_cannot_link: Agent-suggested cannot-link constraints
    
    Returns:
        (merged_must_link, merged_cannot_link)
    """
    merged_must = list(must_link or [])
    merged_cannot = list(cannot_link or [])

    # Add Agent suggestions, avoiding duplicates
    if agent_must_link:
        for constraint in agent_must_link:
            normalized = tuple(sorted(constraint))
            if normalized not in {tuple(sorted(c)) for c in merged_must}:
                merged_must.append(constraint)

    if agent_cannot_link:
        for constraint in agent_cannot_link:
            normalized = tuple(sorted(constraint))
            if normalized not in {tuple(sorted(c)) for c in merged_cannot}:
                merged_cannot.append(constraint)

    return merged_must, merged_cannot


async def _ask_agent_for_initial_constraints(
        node_names: Optional[List[str]],
        agent_optimize_fn: Optional[Callable[[Dict, str], Any]],
        config: PartitionConfig,
        safe_upper: int
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Before formal iterative solving begins, let Agent estimate initial must-link / cannot-link constraints based on node names.

    This does not rely on any existing partition results, only provides node list and description.
    """
    if agent_optimize_fn is None or not node_names:
        return None

    # Capacity hint: tell Agent the upper bound for each service, and require single must-link groups not to exceed safe upper bound
    capacity_hint = ""
    if config is not None and config.size_upper is not None:
        try:
            capacity_hint = (
                f"Please note: the number of nodes in any must-link group should not exceed {safe_upper}. Note that you should consider the connected component size similar to union-find, not simply the node count of a single must-link group. The largest connected component size cannot exceed {safe_upper}.\n"
                f"If a group has too many nodes, please proactively split it into multiple smaller must-link groups.\n" if safe_upper is not None else ""
            )
        except Exception:
            capacity_hint = ""

    advice = (
        "Below is the list of all classes or node names in the current system. Please estimate initial constraints based on semantics/responsibilities:\n"
        f"{node_names}\n"
        f"{capacity_hint}"
        "Please provide:\n"
        "1. Some reasonable must-link groups (field name must_links, list of lists, representing sets of node names that must be in the same service);\n"
        "2. Some reasonable cannot-link constraints (field name cannot_link, elements are pairs [name1, name2], representing nodes that cannot be in the same service).\n"
        "3. Please ensure that the size of each must-link group you provide does not exceed the capacity upper bound safety value mentioned above. If necessary, please split overly large groups into multiple smaller groups.\n"
        "Note: Provide some constraints that you think are reasonable."
    )

    placeholder_partitions: Dict[str, list] = {}

    try:
        optimize_result = await agent_optimize_fn(placeholder_partitions, advice)
    except Exception as e:
        print(f"Exception occurred when calling Agent to get initial must-link / cannot-link constraints: {e}")
        return None

    if optimize_result is None:
        return None

    agent_must_link_groups = getattr(optimize_result, "must_links", []) or []
    agent_cannot_links_raw = getattr(optimize_result, "cannot_link", []) or []

    flat_ml_pairs: List[Tuple[int, int]] = []
    flat_cl_pairs: List[Tuple[int, int]] = []

    name_to_idx = {name: idx for idx, name in enumerate(node_names)}

    # must-link: list of name lists
    for group in agent_must_link_groups:
        if not isinstance(group, (list, tuple)):
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                n1, n2 = group[i], group[j]
                if n1 in name_to_idx and n2 in name_to_idx:
                    i_idx, j_idx = name_to_idx[n1], name_to_idx[n2]
                    if i_idx != j_idx:
                        flat_ml_pairs.append((i_idx, j_idx))

    # cannot-link: list of name pairs
    for item in agent_cannot_links_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        n1, n2 = item
        if n1 in name_to_idx and n2 in name_to_idx:
            i_idx, j_idx = name_to_idx[n1], name_to_idx[n2]
            if i_idx != j_idx:
                flat_cl_pairs.append((i_idx, j_idx))

    unique_ml = list({tuple(sorted(p)) for p in flat_ml_pairs})
    unique_cl = list({tuple(sorted(p)) for p in flat_cl_pairs})

    if not unique_ml and not unique_cl:
        return None
    return unique_ml, unique_cl


async def _ask_agent_for_new_constraints_due_to_infeasible(
        current_must_link: List[Tuple[int, int]],
        node_names: Optional[List[str]],
        agent_optimize_fn: Optional[Callable[[Dict, str], Any]],
        safe_upper: int
) -> Optional[List[Tuple[int, int]]]:
    """
    When the model becomes infeasible due to too many/strong must-link / cannot-link constraints, request Agent to provide a new feasible constraint set.

    Current implementation will:
    - Construct a name-based must-link / cannot-link list
    - Use empty partitions as placeholder, and explain in detail in advice why current constraints cause infeasibility
    - Let Agent return new must_links / cannot_link, then convert back to index form
    """
    if agent_optimize_fn is None or not current_must_link:
        return None

    # Use union-find to merge must-link pairs into connected components, build nested List form constraints
    def _build_must_link_components(pairs: List[Tuple[int, int]]) -> List[List[int]]:
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # Initialize union-find
        for a, b in pairs:
            if a not in parent:
                parent[a] = a
            if b not in parent:
                parent[b] = b
            union(a, b)

        # Collect each connected component
        comp: Dict[int, List[int]] = {}
        for x in parent.keys():
            rx = find(x)
            comp.setdefault(rx, []).append(x)
        return list(comp.values())

    ml_components: List[List[int]] = _build_must_link_components(current_must_link) if current_must_link else []

    # Convert current must-link to name form for easier Agent understanding
    if node_names:
        # must-link: each component after union-find forms a name list
        ml_names: List[List[str]] = [
            [node_names[i] for i in group if 0 <= i < len(node_names)]
            for group in ml_components
            if group
        ]
    else:
        ml_names = [[str(i) for i in group] for group in ml_components if group]

    # Construct capacity info hint to help Agent control must-link group size
    capacity_hint = f"Please pay special attention: the number of nodes in any must-link group should not exceed {safe_upper}, otherwise it may be impossible to find a feasible solution under capacity upper bound constraints.\n" if safe_upper is not None else ""

    advice = (
        "Current MILP solving is infeasible due to too many or unreasonable must-link constraints.\n"
        f"Current must-link constraints (by name or index) are: {ml_names}.\n"
        f"{capacity_hint}"
        "Please give a new must-link constraint suggestion based on this information:\n"
        "1. You can delete some constraints or adjust groups to make overall constraints easier to satisfy;\n"
        "2. Please return a more reasonable and feasible set of must-link (field name is must_links, list of lists format).\n"
        "3. Please ensure that the size of each must-link group you give does not exceed the capacity upper bound safety value mentioned above. If a group exceeds this upper bound, please actively split it into two or more smaller must-link groups.\n"
    )

    # Only need must-link here, so partitions use an empty placeholder
    placeholder_partitions: Dict[str, list] = {}

    try:
        optimize_result = await agent_optimize_fn(placeholder_partitions, advice)
    except Exception as e:
        print(f"Exception occurred when calling Agent to fix must-link constraints: {e}")
        return None

    if optimize_result is None:
        return None

    # Agent returns names, need to map back to indices based on node_names (only adjust must-link)
    agent_must_link_groups = getattr(optimize_result, "must_links", []) or []

    flat_ml_pairs: List[Tuple[int, int]] = []

    if node_names:
        name_to_idx = {name: idx for idx, name in enumerate(node_names)}

        # Process must-link (list of name lists)
        for group in agent_must_link_groups:
            if not isinstance(group, (list, tuple)):
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    n1, n2 = group[i], group[j]
                    if n1 in name_to_idx and n2 in name_to_idx:
                        i_idx, j_idx = name_to_idx[n1], name_to_idx[n2]
                        if i_idx != j_idx:
                            flat_ml_pairs.append((i_idx, j_idx))

    else:
        # If no names, can only try to parse string indices back to integers
        for group in agent_must_link_groups:
            if not isinstance(group, (list, tuple)):
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    try:
                        i_idx = int(group[i])
                        j_idx = int(group[j])
                    except Exception:
                        continue
                    if i_idx != j_idx:
                        flat_ml_pairs.append((i_idx, j_idx))

    # Deduplicate
    unique_ml = list({tuple(sorted(p)) for p in flat_ml_pairs})
    if not unique_ml:
        return None
    return unique_ml


async def iterative_optimize_partition(
        S_struc: np.ndarray,
        S_sem: np.ndarray,
        C_run: np.ndarray,
        config: PartitionConfig,
        edge_index: Optional[torch.Tensor] = None,
        agent_optimize_fn: Optional[Callable[[PartitionResult, str], Any]] = None,
        agent_analyze_fn: Optional[Callable[[Any], Any]] = None,
        node_names: Optional[List[str]] = None,
) -> PartitionResult:
    """
    Iteratively optimize microservice partition: constraint solving → Agent optimization → constraint solving → ...
    
    Args:
        S_struc: Structural similarity matrix
        S_sem: Semantic similarity matrix
        C_run: Runtime coupling matrix
        config: Partition configuration
        edge_index: Edge index
        agent_optimize_fn: Agent optimization function (async), signature is async fn(partitions: Dict) -> Dict
        agent_analyze_fn: Agent evaluation function (async)
        node_names: Node name list (for Agent understanding)
    
    Returns:
        Final partition result
    """
    if edge_index is None:
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

    N = S_struc.shape[0]
    current_must_link = list(config.must_link or [])
    current_cannot_link = list(config.cannot_link or [])

    max_iterations = config.max_iterations
    enable_agent = config.enable_agent_optimization and agent_optimize_fn is not None

    print(f"Starting iterative optimization: max_iterations={max_iterations}, enable_agent_optimization={enable_agent}")

    size_lower_list = config.size_lower
    size_upper_list = config.size_upper
    safe_upper = len(node_names)
    safe_lower = 0
    if size_upper_list:
        safe_upper = min(float(u) for u in size_upper_list)
    if size_lower_list:
        safe_lower = max(float(l) for l in size_lower_list)
    safe_upper = min(safe_upper, len(node_names) - (config.num_communities - 1) * safe_lower)

    # Before formal iteration, if Agent is enabled and node names are provided, let Agent estimate initial constraints (and inform capacity upper bound)
    if enable_agent and node_names:
        print("Calling Agent to estimate initial must-link / cannot-link constraints before iteration starts...")
        initial_constraints = await _ask_agent_for_initial_constraints(
            node_names=node_names,
            agent_optimize_fn=agent_optimize_fn,
            config=config,
            safe_upper=safe_upper
        )
        if initial_constraints:
            init_ml, init_cl = initial_constraints
            print(f"Agent provided {len(init_ml)} initial must-link and {len(init_cl)} initial cannot-link constraints")
            current_must_link, current_cannot_link = _merge_constraints(
                current_must_link,
                current_cannot_link,
                init_ml,
                init_cl,
            )

    for iteration in range(max_iterations):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print(f"{'=' * 60}")

        # Step 1: Constraint solving
        print(f"[Iteration {iteration + 1}] Executing constraint solving...")
        result = optimize_partition(
            S_struc=S_struc,
            S_sem=S_sem,
            C_run=C_run,
            must_link=current_must_link,
            cannot_link=current_cannot_link,
            config=config,
            edge_index=edge_index,
        )

        result.iteration = iteration
        result.total_iterations = max_iterations

        print(f"[Iteration {iteration + 1}] Constraint solving completed")
        print(f"  - Objective value: {result.objective_value:.4f}")
        print(f"  - Solver status: {result.solver_status}")

        # If solving is infeasible (e.g., must-link / cannot-link constraints too strong or capacity configuration unreasonable), try to analyze reasons and let Agent adjust constraints
        if result.solver_status not in ("OPTIMAL", "FEASIBLE"):
            print(f"[Iteration {iteration + 1}] Solving result infeasible, starting to analyze possible reasons...")
            print(f"  - Current must-link constraint count: {len(current_must_link)}")
            print(f"  - Current cannot-link constraint count: {len(current_cannot_link)}")

            # Multiple attempts: let Agent re-give must-link multiple times (no longer adjust cannot-link)
            if enable_agent and current_must_link:
                max_fix_tries = 2
                fix_try = 0
                while result.solver_status not in ("OPTIMAL", "FEASIBLE") and fix_try < max_fix_tries:
                    fix_try += 1
                    print(f"[Iteration {iteration + 1}] Attempt {fix_try} to adjust must-link constraints through Agent...")
                    new_constraints = await _ask_agent_for_new_constraints_due_to_infeasible(
                        current_must_link=current_must_link,
                        node_names=node_names,
                        agent_optimize_fn=agent_optimize_fn,
                        safe_upper=safe_upper
                    )
                    if not new_constraints:
                        print(f"[Iteration {iteration + 1}] Agent failed to provide new constraints, stopping further attempts")
                        break

                    new_must_link = new_constraints
                    print(f"[Iteration {iteration + 1}] Agent returned new must-link constraint count: {len(new_must_link)}")
                    current_must_link = new_must_link

                    # Re-solve using updated constraints (without increasing iteration count)
                    result = optimize_partition(
                        S_struc=S_struc,
                        S_sem=S_sem,
                        C_run=C_run,
                        must_link=current_must_link,
                        cannot_link=current_cannot_link,
                        config=config,
                        edge_index=edge_index,
                    )
                    result.iteration = iteration
                    result.total_iterations = max_iterations
                    print(
                        f"[Iteration {iteration + 1}] Re-solving completed, status: {result.solver_status}, objective value: {result.objective_value:.4f}")

            # If still infeasible after multiple Agent adjustments, try completely removing must-link / cannot-link as fallback
            if result.solver_status not in ("OPTIMAL", "FEASIBLE"):
                print(
                    f"[Iteration {iteration + 1}] Still infeasible after multiple Agent adjustments, will try re-solving without must-link / cannot-link constraints (fallback).")
                unconstrained_result = optimize_partition(
                    S_struc=S_struc,
                    S_sem=S_sem,
                    C_run=C_run,
                    must_link=[],
                    cannot_link=[],
                    config=config,
                    edge_index=edge_index,
                )
                print(
                    f"[Iteration {iteration + 1}] Fallback solving without constraints completed, status: {unconstrained_result.solver_status}, objective value: {unconstrained_result.objective_value:.4f}")
                if unconstrained_result.solver_status in ("OPTIMAL", "FEASIBLE"):
                    print(f"[Iteration {iteration + 1}] Using fallback solution without must-link / cannot-link as current iteration result.")
                    result = unconstrained_result
                    current_must_link = []
                    current_cannot_link = []
                else:
                    print(
                        f"[Iteration {iteration + 1}] Even after removing must-link / cannot-link, problem is still infeasible, ending iteration and returning current result.")
                    return result

        _save_iteration_result(
            result=result,
            iteration=iteration + 1,
            node_names=node_names
        )

        # If this is the last iteration or Agent optimization is not enabled, return directly
        if iteration == max_iterations - 1 or not enable_agent:
            print(f"\nOptimization completed (iteration {iteration + 1}/{max_iterations})")
            return result

        # Step 2: Agent optimization
        print(f"[Iteration {iteration + 1}] Calling Agent for optimization...")
        partitions = _convert_assignments_to_partitions(result.assignments, N)

        # If node names are provided, build a more friendly partition representation
        if node_names:
            partitions_with_names = {
                k: [node_names[i] for i in v]
                for k, v in partitions.items()
            }
        else:
            partitions_with_names = partitions

        # Construct capacity hint for analysis Agent (_capacity_hint), informing safe_upper and other info
        analyze_input = partitions_with_names
        if config is not None and getattr(config, "size_upper", None) is not None:
            try:
                size_upper_list = list(getattr(config, "size_upper", []))
                safe_upper = None
                if size_upper_list:
                    safe_upper = min(float(u) for u in size_upper_list)
                capacity_meta = {
                    "size_upper": size_upper_list,
                    "safe_upper": safe_upper,
                }
                analyze_input = {
                    "_capacity_hint": capacity_meta,
                    "partitions": partitions_with_names,
                }
            except Exception:
                analyze_input = partitions_with_names

        try:
            analyze_result = await agent_analyze_fn(analyze_input, safe_upper)
            if analyze_result is None:
                print(f"[Iteration {iteration + 1}] Agent analysis returned None, ending iteration")
                return result
            if hasattr(analyze_result, 'needs_optimization'):
                if not analyze_result.needs_optimization:
                    print(f"[Iteration {iteration + 1}] Agent analysis indicates no optimization needed, ending iteration")
                    return result
            suggestions = getattr(analyze_result, 'suggestions', '') or ''
            suggestions = suggestions + f"Maximum number of nodes per service is {safe_upper}"
            optimize_result = await agent_optimize_fn(partitions_with_names, suggestions)

            if optimize_result is None:
                print(f"[Iteration {iteration + 1}] Agent returned None, ending iteration")
                return result

            # Extract Agent suggestions
            agent_must_link = getattr(optimize_result, 'must_links', []) or []
            agent_cannot_link = getattr(optimize_result, 'cannot_link', []) or []

            print(f"[Iteration {iteration + 1}] Agent suggestions:")
            print(f"  - Must-link: {len(agent_must_link)} constraints")
            print(f"  - Must-link: {agent_must_link}")
            print(f"  - Cannot-link: {len(agent_cannot_link)} constraints")
            print(f"  - Cannot-link: {agent_cannot_link}")

            agent_must_link_list = agent_must_link
            agent_must_link = []
            for link_list in agent_must_link_list:
                for i in range(len(link_list)):
                    for j in range(i + 1, len(link_list)):
                        if link_list[i] != link_list[j]:
                            agent_must_link.append((link_list[i], link_list[j]))

            # If node names are used, need to convert back to node indices
            if node_names:
                name_to_idx = {name: idx for idx, name in enumerate(node_names)}
                agent_must_link = [
                    (name_to_idx[m[0]], name_to_idx[m[1]])
                    for m in agent_must_link
                    if m[0] in name_to_idx and m[1] in name_to_idx
                ]
                agent_cannot_link = [
                    (name_to_idx[c[0]], name_to_idx[c[1]])
                    for c in agent_cannot_link
                    if c[0] in name_to_idx and c[1] in name_to_idx
                ]

            # Merge constraints
            current_must_link, current_cannot_link = _merge_constraints(
                current_must_link,
                current_cannot_link,
                agent_must_link,
                agent_cannot_link,
            )

            # Save Agent feedback
            result.agent_feedback = {
                "iteration": iteration,
                "must_links": agent_must_link,
                "cannot_link": agent_cannot_link,
            }

            print(f"[Iteration {iteration + 1}] Constraints updated, preparing for next round of solving")

        except Exception as e:
            print(f"[Iteration {iteration + 1}] Agent optimization failed: {e}")
            print(f"[Iteration {iteration + 1}] Returning current best result")
            return result

    return result


async def partition_from_multi_embeddings_iterative(
        emb_struct: torch.Tensor,
        emb_sem: torch.Tensor,
        emb_fused: torch.Tensor,
        edge_index: torch.Tensor,
        config: PartitionConfig,
        edge_weights: Optional[torch.Tensor] = None,
        agent_optimize_fn: Optional[Callable[[PartitionResult, str], Any]] = None,
        agent_analyze_fn: Optional[Callable[[PartitionResult], Any]] = None,
        node_names: Optional[List[str]] = None,
) -> PartitionResult:
    """
    Use three embeddings (structural / semantic / fused) to jointly participate in microservice partitioning (iterative version).

    Idea:
        - S_struc is still built from graph structure (edge_index + edge_weights);
        - Combine cosine similarities of three embeddings (mapped to [0,1]) with weighted sum to get S_sem_combined:
              S_sem = w_s * cos(emb_struct) + w_t * cos(emb_sem) + w_f * cos(emb_fused)
          where weights are controlled by beta_struct / beta_sem / beta_fused.
        - Then reuse the existing iterative MILP process.

    Parameter description:
        emb_struct:
            Structural vector representation (structural embedding), shape [num_nodes, dim],
            usually obtained from structural encoder, emphasizing structural information such as call relationships, dependency graphs, etc.
        emb_sem:
            Semantic vector representation (semantic embedding), shape [num_nodes, dim],
            usually obtained from semantic models such as code text/comments, emphasizing business semantic similarity.
        emb_fused:
            Fused vector representation (fused embedding), shape [num_nodes, dim],
            generally a representation after further fusion of multi-modal features such as structural + semantic.
        edge_index:
            Graph edge index, shape [2, num_edges], each column is a directed edge (src, dst),
            used to build structural similarity matrix S_struc and runtime coupling matrix C_run.
        edge_weights:
            Edge weight tensor, shape [num_edges] (optional),
            if provided, weights are accumulated when building S_struc and C_run instead of simple counting.
        agent_optimize_fn:
            Agent optimization function (async), signature roughly async fn(partitions_or_result, suggestions: str) -> Any,
            used to return new constraint suggestions (must_links / cannot_link) based on current partitioning and hint information.
        agent_analyze_fn:
            Agent analysis function (async), signature roughly async fn(partitions_or_result) -> Any,
            used to evaluate current partitioning quality, give whether optimization is needed and suggestion text, etc.
        node_names:
            Node name list (such as class names, file names, etc.), length num_nodes;
            if provided, will use human-readable names instead of pure indices when interacting with Agent, for easier understanding and giving constraints.
    """
    N = emb_struct.size(0)
    assert emb_sem.size(0) == N and emb_fused.size(0) == N, "Number of nodes in three embeddings must be consistent"

    sims: List[np.ndarray] = []
    weights: List[float] = []

    if config.beta_struct > 0.0:
        sims.append(_cosine_similarity_01(emb_struct))
        weights.append(config.beta_struct)
    if config.beta_sem > 0.0:
        sims.append(_cosine_similarity_01(emb_sem))
        weights.append(config.beta_sem)
    if config.beta_fused > 0.0:
        sims.append(_cosine_similarity_01(emb_fused))
        weights.append(config.beta_fused)

    if not sims:
        sims.append(_cosine_similarity_01(emb_fused))
        weights.append(1.0)

    weights_np = np.array(weights, dtype=np.float64)
    weights_np = weights_np / (weights_np.sum() + 1e-8)

    S_vec = np.zeros((N, N), dtype=np.float64)
    for w, S in zip(weights_np, sims):
        S_vec += w * S

    # Structural similarity is still based on graph structure, can be changed to also fuse structural embedding if needed
    S_struc = build_structural_similarity(N, edge_index, weight=edge_weights, symmetric=config.symmetric_struc)
    C_run = build_runtime_coupling(N, edge_index, weight=edge_weights)

    return await iterative_optimize_partition(
        S_struc=S_struc,
        S_sem=S_vec,
        C_run=C_run,
        config=config,
        edge_index=edge_index,
        agent_optimize_fn=agent_optimize_fn,
        agent_analyze_fn=agent_analyze_fn,
        node_names=node_names,
    )
