"""
微服务分区优化器（0-1 IQP 通过 OR-Tools CP-SAT 线性化为 MILP）。

公式
- 变量：x[i,k] in {0,1}（类 i 分配给服务 k）
- 辅助变量：y[i,j,k] = x[i,k] & x[j,k]（线性化）
- 目标函数（最大化）：
    alpha * sum_k sum_{i<j} S_struc[i,j] * y[i,j,k]
  + beta  * sum_k sum_{i<j} S_sem[i,j]   * y[i,j,k]
  - gamma * sum_k sum_{i<j} C_run[i,j]   * (x[i,k] - y[i,j,k])   # 跨服务耦合

约束条件
- Sum_k x[i,k] = 1（每个类恰好分配到一个服务）
- 容量：L_k <= sum_i s_i * x[i,k] <= U_k（如果提供）
- 线性化：y[i,j,k] <= x[i,k]; y[i,j,k] <= x[j,k]; y[i,j,k] >= x[i,k] + x[j,k] - 1
- 必须链接（硬约束）：对所有 k，x[i,k] == x[j,k]
- 不能链接（硬约束）：对所有 k，x[i,k] + x[j,k] <= 1

注意
- CP-SAT 是整数求解器；我们将所有实数值分数按整数 SCALE 缩放并四舍五入。
- 对于大 N，为所有对创建 y 的复杂度为 O(N^2 K)。使用 pair_threshold 来稀疏化对。
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
    iteration: int = 0  # 当前迭代次数
    total_iterations: int = 3  # 总迭代次数
    agent_feedback: Optional[Dict] = None  # Agent 优化的反馈信息


def _save_iteration_result(
        result: "PartitionResult",
        iteration: int,
        node_names: Optional[List[str]],
) -> None:
    """
    将当前迭代的划分结果保存到 result 目录。

    文件命名为 result_iter_{iteration}.json，内容为 Service-x -> [类名或索引]。
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
        print(f"[迭代 {iteration}] 结果已保存到 {iter_path}")
    except Exception as e:
        print(f"[迭代 {iteration}] 保存结果到 result 目录失败: {e}")


def cosine_similarity(emb: torch.Tensor) -> np.ndarray:
    emb = torch.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1, eps=1e-8)
    sim = emb @ emb.t()
    return sim.cpu().numpy().astype(np.float64)


def _cosine_similarity_01(emb: torch.Tensor) -> np.ndarray:
    """
    将 embedding 映射为 [0,1] 区间的余弦相似度矩阵：
    - 先做 L2 归一化
    - 再计算余弦相似度 [-1,1]
    - 最后映射到 [0,1]
    """
    sim = cosine_similarity(emb)  # [-1,1]
    return (sim + 1.0) / 2.0


def build_structural_similarity(num_nodes: int,
                                edge_index: torch.Tensor,
                                weight: Optional[torch.Tensor] = None,
                                symmetric: bool = True) -> np.ndarray:
    """
    从有向边构建 S_struc。默认情况下，我们对称化以奖励相互内聚度。
    S_ij 在 [0,1] 范围内，按最大度数归一化。
    
    Args:
        num_nodes: 节点数
        edge_index: [2, num_edges] 边索引
        weight: [num_edges] 边权重（可选），如果提供则使用这些权重
        symmetric: 是否对称化相似度矩阵
    
    Returns:
        [num_nodes, num_nodes] 结构相似度矩阵
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
    从有向边构建 C_run；值越高意味着跨越服务的成本越高。
    默认值：每条边 1。
    
    Args:
        num_nodes: 节点数
        edge_index: [2, num_edges] 边索引
        weight: [num_edges] 边权重（可选），如果提供则使用这些权重
    
    Returns:
        [num_nodes, num_nodes] 运行时耦合矩阵
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
    分位数稀疏化（替代固定阈值）
    :param quantile_threshold: 分位数（0~1），例：0.9 → 保留前10%高亲和性类对
    """
    print("稀疏化阈值：", quantile_threshold)
    N = weights.shape[0]
    # 1. 收集所有i<j的类对权重（排除对角线）
    all_weights = []
    for i in range(N):
        for j in range(i + 1, N):
            all_weights.append(float(weights[i, j]))

    # 2. 计算分位数阈值（核心：适配不同数据分布）
    if not all_weights:
        return []
    threshold = np.quantile(all_weights, quantile_threshold)

    # 3. 筛选高亲和性类对
    pairs: List[Tuple[int, int, float]] = []
    for i in range(N):
        for j in range(i + 1, N):
            weight_val = float(weights[i, j])
            if weight_val >= threshold:  # 无需abs，因weights非负
                pairs.append((i, j, weight_val))

    # 打印日志，方便调参
    total = len(all_weights)
    kept = len(pairs)
    print(f"稀疏化：总类对{total} → 保留{kept}，压缩比{kept / total:.2%}（分位数阈值={quantile_threshold}）")
    return pairs


def _debug_print_objective_components(
        assignments: List[int],
        S_struc: np.ndarray,
        S_sem: np.ndarray,
        C_run: np.ndarray,
        config: "PartitionConfig",
) -> None:
    """
    根据最终的 assignments，按「原始定义公式」近似计算各目标项的分值，并打印出来，方便调试权重。

    这里不严格复现 MILP 中的稀疏化和线性化细节，而是按直观的目标：
      - 结构内聚：同一服务内所有 (i<j) 的 S_struc[i,j] 之和
      - 语义内聚：同一服务内所有 (i<j) 的 S_sem[i,j] 之和
      - 运行时跨服务惩罚：所有跨服务有向对 (i!=j, svc[i]!=svc[j]) 的 C_run[i,j] 之和

    这样更便于你直观比较 alpha / beta / gamma 的相对量级。
    """
    N = len(assignments)
    alpha, beta, gamma = config.alpha, config.beta, config.gamma

    intra_struc = 0.0
    intra_sem = 0.0
    cross_run = 0.0

    # 同服务对的结构/语义内聚
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

    # 跨服务运行时惩罚（有向）
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

    print("\n[调试] 目标函数分量（基于 assignments 的近似计算）")
    print("------------------------------------------------------------")
    print(f"  节点数 N              : {N}")
    print(f"  alpha (结构内聚权重) : {alpha}")
    print(f"  beta  (语义内聚权重) : {beta}")
    print(f"  gamma (跨服务惩罚权重): {gamma}")
    print("------------------------------------------------------------")
    print(f"  结构内聚  raw        : {intra_struc:.6f}")
    print(f"  结构内聚  weighted   : {weighted_struc:.6f}")
    print(f"  语义内聚  raw        : {intra_sem:.6f}")
    print(f"  语义内聚  weighted   : {weighted_sem:.6f}")
    print(f"  跨服务惩罚 raw       : {cross_run:.6f}")
    print(f"  跨服务惩罚 weighted  : {weighted_run:.6f}")
    print("------------------------------------------------------------")
    print(f"  近似总目标值（各项加权和）: {total_approx:.6f}\n")


def optimize_partition(S_struc: np.ndarray,
                       S_sem: np.ndarray,
                       C_run: np.ndarray,
                       must_link: Optional[List[Tuple[int, int]]] = None,
                       cannot_link: Optional[List[Tuple[int, int]]] = None,
                       config: PartitionConfig = None,
                       edge_index: Optional[torch.Tensor] = None) -> PartitionResult:
    """
    通过 CP-SAT 求解微服务分区 MILP。
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

    # 预计算目标函数中 y[i,j,k] 使用的对权重
    # 使用对称运行时耦合（合并友好）来帮助选择信息对
    # 并在下面的单独跨服务惩罚中保持方向性 C_run。
    C_sym = (C_run + C_run.T) / 2.0
    pair_weight = config.alpha * S_struc + config.beta * S_sem + config.gamma * C_sym
    pair_list = _sparsify_pairs(pair_weight, config.pair_threshold)
    pair_set = {(i, j) for (i, j, _) in pair_list}

    model = cp_model.CpModel()

    # 变量 x[i,k]
    x = [[model.NewBoolVar(f"x_{i}_{k}") for k in range(K)] for i in range(N)]

    # y[i,j,k] 仅用于选定的对以减少大小
    # 我们将在字典 y[(i,j,k)] 中索引它们
    y: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
    for (i, j, _) in pair_list:
        for k in range(K):
            y[(i, j, k)] = model.NewBoolVar(f"y_{i}_{j}_{k}")
            # 线性化
            model.Add(y[(i, j, k)] <= x[i][k])
            model.Add(y[(i, j, k)] <= x[j][k])
            model.Add(y[(i, j, k)] >= x[i][k] + x[j][k] - 1)

    # 每个节点恰好分配到一个服务
    for i in range(N):
        model.Add(sum(x[i][k] for k in range(K)) == 1)

    # 必须链接 / 不能链接（默认为硬约束）
    for (i, j) in must_link:
        for k in range(K):
            model.Add(x[i][k] == x[j][k])
    for (i, j) in cannot_link:
        for k in range(K):
            model.Add(x[i][k] + x[j][k] <= 1)

    # 容量约束
    if config.size_lower is not None:
        for k in range(K):
            Lk = int(config.size_lower[k])
            model.Add(sum(x[i][k] for i in range(N)) >= Lk)
    if config.size_upper is not None:
        for k in range(K):
            Uk = int(config.size_upper[k])
            model.Add(sum(x[i][k] for i in range(N)) <= Uk)

    # 目标函数构造（缩放整数）
    SCALE = config.scale
    objective_terms: List[cp_model.LinearExpr] = []

    # 内聚度项：alpha*S_struc*y + beta*S_sem*y
    for (i, j, _) in pair_list:
        for k in range(K):
            coeff = int(round(SCALE * (config.alpha * S_struc[i, j] + config.beta * S_sem[i, j])))
            if coeff != 0:
                objective_terms.append(coeff * y[(i, j, k)])

    # 跨服务惩罚：-gamma*C_run[i,j]*(x_i_k - y_ij_k)
    # 等价于 -gamma*C_run*x_i_k  + gamma*C_run*y_ij_k
    # 第一项在有序对 (i,j) 上，第二项仅适用于我们为其创建 y 的对。
    # 使用有序对来反映方向性。
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            coeff_x = int(round(SCALE * (-config.gamma * C_run[i, j])))
            if coeff_x != 0:
                for k in range(K):
                    objective_terms.append(coeff_x * x[i][k])
            # y 部分（仅当存在时；使用 base=(min(i,j),max(i,j)) 避免重复）
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

    # 目标值（重新缩放）
    obj_value = solver.ObjectiveValue() / SCALE

    # 如果需要调试，按 assignments 近似分解各目标分量
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        try:
            _debug_print_objective_components(assignments, S_struc, S_sem, C_run, config)
        except Exception as e:
            print(f"[调试] 计算目标分量时发生异常: {e}")

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
    将分配结果转换为分区字典格式。
    
    Args:
        assignments: [num_nodes] 每个节点的服务分配
        num_nodes: 节点总数
    
    Returns:
        {service_id: [node_ids]} 分区字典
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
    合并来自 Agent 的约束和原有约束。
    
    Args:
        must_link: 原有的必须链接约束
        cannot_link: 原有的不能链接约束
        agent_must_link: Agent 建议的必须链接约束
        agent_cannot_link: Agent 建议的不能链接约束
    
    Returns:
        (merged_must_link, merged_cannot_link)
    """
    merged_must = list(must_link or [])
    merged_cannot = list(cannot_link or [])

    # 添加 Agent 的建议，避免重复
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
    在正式迭代求解开始前，让 Agent 基于节点名称预估一份初始的 must-link / cannot-link 约束。

    这里不依赖任何已有划分结果，仅提供节点列表和说明。
    """
    if agent_optimize_fn is None or not node_names:
        return None

    # 容量提示：告诉 Agent 每个服务的上界，并要求单个 must-link 分组不要超过安全上界
    capacity_hint = ""
    if config is not None and config.size_upper is not None:
        try:
            capacity_hint = (
                f"请特别注意：任意一个 must-link 分组中的节点数量不要超过 {safe_upper}，注意，你要考虑的是类似并查集的连通分量大小，而不是简单的单个 must-link 分组的节点数量，最大的连通分量大小不能够超过 {safe_upper}。\n"
                f"如果一个分组中节点过多，请你主动将其拆分成多个较小的 must-link 分组。\n" if safe_upper is not None else ""
            )
        except Exception:
            capacity_hint = ""

    advice = (
        "下面是当前系统中所有的类或节点名称列表，请你基于语义/职责预估一份初始约束：\n"
        f"{node_names}\n"
        f"{capacity_hint}"
        "请给出：\n"
        "1. 一些合理的 must-link 分组（字段名 must_links，列表的列表，表示必须在同一服务中的节点名称集合）；\n"
        "2. 一些合理的 cannot-link 约束（字段名 cannot_link，元素是二元组，[name1, name2]，表示不能在同一服务中）。\n"
        "3. 请确保你给出的每个 must-link 分组规模都不超过上面提示中的容量上界安全值，如果需要请将过大的分组合并拆分为多个较小分组。\n"
        "注意：给出你认为合理的一些约束。"
    )

    placeholder_partitions: Dict[str, list] = {}

    try:
        optimize_result = await agent_optimize_fn(placeholder_partitions, advice)
    except Exception as e:
        print(f"调用 Agent 获取初始 must-link / cannot-link 约束时发生异常: {e}")
        return None

    if optimize_result is None:
        return None

    agent_must_link_groups = getattr(optimize_result, "must_links", []) or []
    agent_cannot_links_raw = getattr(optimize_result, "cannot_link", []) or []

    flat_ml_pairs: List[Tuple[int, int]] = []
    flat_cl_pairs: List[Tuple[int, int]] = []

    name_to_idx = {name: idx for idx, name in enumerate(node_names)}

    # must-link: 名称列表的列表
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

    # cannot-link: 名称二元组列表
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
    当因为 must-link / cannot-link 约束过多/过强导致模型不可行时，请求 Agent 重新给出一份可行的约束集合。

    当前实现会：
    - 构造一份基于名称的 must-link / cannot-link 列表
    - 使用空的 partitions 作为占位，并在 advice 中详细说明当前约束导致不可行
    - 让 Agent 返回新的 must_links / cannot_link，再转换回索引形式
    """
    if agent_optimize_fn is None or not current_must_link:
        return None

    # 使用并查集把 must-link 对合并成连通分量，构建嵌套 List 形式的约束
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

        # 初始化并查集
        for a, b in pairs:
            if a not in parent:
                parent[a] = a
            if b not in parent:
                parent[b] = b
            union(a, b)

        # 收集每个连通分量
        comp: Dict[int, List[int]] = {}
        for x in parent.keys():
            rx = find(x)
            comp.setdefault(rx, []).append(x)
        return list(comp.values())

    ml_components: List[List[int]] = _build_must_link_components(current_must_link) if current_must_link else []

    # 将当前 must-link 转成名称形式，方便 Agent 理解
    if node_names:
        # must-link：使用并查集后的每个分量，构成一个名称列表
        ml_names: List[List[str]] = [
            [node_names[i] for i in group if 0 <= i < len(node_names)]
            for group in ml_components
            if group
        ]
    else:
        ml_names = [[str(i) for i in group] for group in ml_components if group]

    # 构造容量信息提示，帮助 Agent 控制 must-link 分组规模
    capacity_hint = f"请特别注意：任意一个 must-link 分组中的节点数量不要超过 {safe_upper}，否则在容量上界约束下可能无法找到可行解。\n" if safe_upper is not None else ""

    advice = (
        "当前 MILP 求解因为 must-link 约束过多或不合理导致约束不可满足（模型不可行）。\n"
        f"当前的 must-link 约束（按名称或索引）为：{ml_names}。\n"
        f"{capacity_hint}"
        "请基于这些信息，给出一份新的 must-link 约束建议：\n"
        "1. 可以删除一部分约束，或调整分组，使得整体约束更容易被满足；\n"
        "2. 请返回你认为更合理、可行的一组 must-link（字段名为 must_links，列表的列表形式）。\n"
        "3. 请确保你给出的每个 must-link 分组规模都不超过上面提示中的容量上界安全值，如果某个分组超过该上界，请你主动将其拆分成两个或多个较小的 must-link 分组。\n"
    )

    # 这里只需要 must-link，因此 partitions 用一个空占位即可
    placeholder_partitions: Dict[str, list] = {}

    try:
        optimize_result = await agent_optimize_fn(placeholder_partitions, advice)
    except Exception as e:
        print(f"调用 Agent 以修正 must-link 约束时发生异常: {e}")
        return None

    if optimize_result is None:
        return None

    # Agent 返回的是名称，需要根据 node_names 映射回索引（只调整 must-link）
    agent_must_link_groups = getattr(optimize_result, "must_links", []) or []

    flat_ml_pairs: List[Tuple[int, int]] = []

    if node_names:
        name_to_idx = {name: idx for idx, name in enumerate(node_names)}

        # 处理 must-link（名称列表的列表）
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
        # 如果没有名称，只能尝试将字符串索引解析回整数
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

    # 去重
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
    迭代优化微服务分区：约束求解 → Agent 优化 → 约束求解 → ...
    
    Args:
        S_struc: 结构相似度矩阵
        S_sem: 语义相似度矩阵
        C_run: 运行时耦合矩阵
        config: 分区配置
        edge_index: 边索引
        agent_optimize_fn: Agent 优化函数（异步），签名为 async fn(partitions: Dict) -> Dict
        agent_analyze_fn: Agent 评估函数（异步）
        node_names: 节点名称列表（用于 Agent 理解）
    
    Returns:
        最终的分区结果
    """
    if edge_index is None:
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

    N = S_struc.shape[0]
    current_must_link = list(config.must_link or [])
    current_cannot_link = list(config.cannot_link or [])

    max_iterations = config.max_iterations
    enable_agent = config.enable_agent_optimization and agent_optimize_fn is not None

    print(f"开始迭代优化：最大迭代次数={max_iterations}，启用Agent优化={enable_agent}")

    size_lower_list = config.size_lower
    size_upper_list = config.size_upper
    safe_upper = len(node_names)
    safe_lower = 0
    if size_upper_list:
        safe_upper = min(float(u) for u in size_upper_list)
    if size_lower_list:
        safe_lower = max(float(l) for l in size_lower_list)
    safe_upper = min(safe_upper, len(node_names) - (config.num_communities - 1) * safe_lower)

    # 在正式迭代前，如果启用了 Agent 且提供了节点名称，先让 Agent 预估一份初始约束（并告知容量上界）
    if enable_agent and node_names:
        print("在迭代开始前调用 Agent 预估初始 must-link / cannot-link 约束...")
        initial_constraints = await _ask_agent_for_initial_constraints(
            node_names=node_names,
            agent_optimize_fn=agent_optimize_fn,
            config=config,
            safe_upper=safe_upper
        )
        if initial_constraints:
            init_ml, init_cl = initial_constraints
            print(f"Agent 给出了 {len(init_ml)} 条初始 must-link 和 {len(init_cl)} 条初始 cannot-link 约束")
            current_must_link, current_cannot_link = _merge_constraints(
                current_must_link,
                current_cannot_link,
                init_ml,
                init_cl,
            )

    for iteration in range(max_iterations):
        print(f"\n{'=' * 60}")
        print(f"迭代 {iteration + 1}/{max_iterations}")
        print(f"{'=' * 60}")

        # 第一步：约束求解
        print(f"[迭代 {iteration + 1}] 执行约束求解...")
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

        print(f"[迭代 {iteration + 1}] 约束求解完成")
        print(f"  - 目标值: {result.objective_value:.4f}")
        print(f"  - 求解器状态: {result.solver_status}")

        # 如果求解不可行（例如 must-link / cannot-link 约束过强或容量配置不合理），尝试分析原因并让 Agent 调整约束
        if result.solver_status not in ("OPTIMAL", "FEASIBLE"):
            print(f"[迭代 {iteration + 1}] 求解结果不可行，开始分析可能原因...")
            print(f"  - 当前 must-link 约束数量: {len(current_must_link)}")
            print(f"  - 当前 cannot-link 约束数量: {len(current_cannot_link)}")

            # 多轮尝试：让 Agent 多次重新给 must-link（不再调整 cannot-link）
            if enable_agent and current_must_link:
                max_fix_tries = 2
                fix_try = 0
                while result.solver_status not in ("OPTIMAL", "FEASIBLE") and fix_try < max_fix_tries:
                    fix_try += 1
                    print(f"[迭代 {iteration + 1}] 第 {fix_try} 次尝试通过 Agent 调整 must-link 约束...")
                    new_constraints = await _ask_agent_for_new_constraints_due_to_infeasible(
                        current_must_link=current_must_link,
                        node_names=node_names,
                        agent_optimize_fn=agent_optimize_fn,
                        safe_upper=safe_upper
                    )
                    if not new_constraints:
                        print(f"[迭代 {iteration + 1}] Agent 未能给出新的约束，停止进一步尝试")
                        break

                    new_must_link = new_constraints
                    print(f"[迭代 {iteration + 1}] Agent 返回的新 must-link 约束数量: {len(new_must_link)}")
                    current_must_link = new_must_link

                    # 使用更新后的约束重新求解一次（不增加迭代轮次）
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
                        f"[迭代 {iteration + 1}] 重新求解完成，状态: {result.solver_status}, 目标值: {result.objective_value:.4f}")

            # 如果多次通过 Agent 调整后仍然不可行，则尝试完全移除 must-link / cannot-link 作为兜底
            if result.solver_status not in ("OPTIMAL", "FEASIBLE"):
                print(
                    f"[迭代 {iteration + 1}] 经过 Agent 多次调整后仍然不可行，将尝试在无 must-link / cannot-link 约束下重新求解（兜底）。")
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
                    f"[迭代 {iteration + 1}] 无约束兜底求解完成，状态: {unconstrained_result.solver_status}, 目标值: {unconstrained_result.objective_value:.4f}")
                if unconstrained_result.solver_status in ("OPTIMAL", "FEASIBLE"):
                    print(f"[迭代 {iteration + 1}] 使用无 must-link / cannot-link 的兜底解作为当前迭代结果。")
                    result = unconstrained_result
                    current_must_link = []
                    current_cannot_link = []
                else:
                    print(
                        f"[迭代 {iteration + 1}] 即便移除 must-link / cannot-link，问题仍然不可行，结束迭代并返回当前结果。")
                    return result

        _save_iteration_result(
            result=result,
            iteration=iteration + 1,
            node_names=node_names
        )

        # 如果是最后一次迭代或不启用 Agent 优化，直接返回
        if iteration == max_iterations - 1 or not enable_agent:
            print(f"\n优化完成（迭代 {iteration + 1}/{max_iterations}）")
            return result

        # 第二步：Agent 优化
        print(f"[迭代 {iteration + 1}] 调用 Agent 进行优化...")
        partitions = _convert_assignments_to_partitions(result.assignments, N)

        # 如果提供了节点名称，构建更友好的分区表示
        if node_names:
            partitions_with_names = {
                k: [node_names[i] for i in v]
                for k, v in partitions.items()
            }
        else:
            partitions_with_names = partitions

        # 为分析 Agent 构造容量提示（_capacity_hint），告知 safe_upper 等信息
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
                print(f"[迭代 {iteration + 1}] Agent 分析返回 None，结束迭代")
                return result
            if hasattr(analyze_result, 'needs_optimization'):
                if not analyze_result.needs_optimization:
                    print(f"[迭代 {iteration + 1}] Agent 分析不需要优化，结束迭代")
                    return result
            suggestions = getattr(analyze_result, 'suggestions', '') or ''
            suggestions = suggestions + f"每个服务的最大节点数量为 {safe_upper}"
            optimize_result = await agent_optimize_fn(partitions_with_names, suggestions)

            if optimize_result is None:
                print(f"[迭代 {iteration + 1}] Agent 返回 None，结束迭代")
                return result

            # 提取 Agent 的建议
            agent_must_link = getattr(optimize_result, 'must_links', []) or []
            agent_cannot_link = getattr(optimize_result, 'cannot_link', []) or []

            print(f"[迭代 {iteration + 1}] Agent 建议:")
            print(f"  - 必须链接: {len(agent_must_link)} 个约束")
            print(f"  - 必须链接: {agent_must_link}")
            print(f"  - 不能链接: {len(agent_cannot_link)} 个约束")
            print(f"  - 不能链接: {agent_cannot_link}")

            agent_must_link_list = agent_must_link
            agent_must_link = []
            for link_list in agent_must_link_list:
                for i in range(len(link_list)):
                    for j in range(i + 1, len(link_list)):
                        if link_list[i] != link_list[j]:
                            agent_must_link.append((link_list[i], link_list[j]))

            # 如果节点名称被使用，需要转换回节点索引
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

            # 合并约束
            current_must_link, current_cannot_link = _merge_constraints(
                current_must_link,
                current_cannot_link,
                agent_must_link,
                agent_cannot_link,
            )

            # 保存 Agent 反馈
            result.agent_feedback = {
                "iteration": iteration,
                "must_links": agent_must_link,
                "cannot_link": agent_cannot_link,
            }

            print(f"[迭代 {iteration + 1}] 约束已更新，准备下一轮求解")

        except Exception as e:
            print(f"[迭代 {iteration + 1}] Agent 优化失败: {e}")
            print(f"[迭代 {iteration + 1}] 返回当前最佳结果")
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
    使用三种 embedding（结构 / 语义 / 融合）共同参与微服务划分（迭代版）。

    思路：
        - S_struc 仍然由图结构（edge_index + edge_weights）构建；
        - 将三种 embedding 的余弦相似度（映射到 [0,1]）按权重加权求和得到 S_sem_combined：
              S_sem = w_s * cos(emb_struct) + w_t * cos(emb_sem) + w_f * cos(emb_fused)
          其中权重由 beta_struct / beta_sem / beta_fused 控制。
        - 之后复用原有的迭代 MILP 流程。

    参数说明：
        emb_struct:
            结构向量表示（结构 embedding），形状为 [num_nodes, dim]，
            通常由结构编码器得到，强调调用关系、依赖图等结构信息。
        emb_sem:
            语义向量表示（语义 embedding），形状为 [num_nodes, dim]，
            通常由代码文本/注释等语义模型得到，强调业务语义相似性。
        emb_fused:
            融合向量表示（融合 embedding），形状为 [num_nodes, dim]，
            一般是结构 + 语义等多模态特征进一步融合后的表示。
        edge_index:
            图的边索引，形状为 [2, num_edges]，每一列是一条有向边 (src, dst)，
            用于构建结构相似度矩阵 S_struc 和运行时耦合矩阵 C_run。
        edge_weights:
            边权重张量，形状为 [num_edges]（可选），
            若提供则在构建 S_struc 和 C_run 时按权重累加而非简单计数。
        agent_optimize_fn:
            Agent 优化函数（异步），签名大致为 async fn(partitions_or_result, suggestions: str) -> Any，
            用于根据当前划分和提示信息返回新的约束建议（must_links / cannot_link）。
        agent_analyze_fn:
            Agent 分析函数（异步），签名大致为 async fn(partitions_or_result) -> Any，
            用于评估当前划分质量、给出是否需要优化及建议文本等。
        node_names:
            节点名称列表（如类名、文件名等），长度为 num_nodes；
            若提供，将在与 Agent 交互时使用人类可读的名称而非纯索引，便于理解和给出约束。
    """
    N = emb_struct.size(0)
    assert emb_sem.size(0) == N and emb_fused.size(0) == N, "三种 embedding 的节点数必须一致"

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

    # 结构相似度仍然基于图结构，可在需要时改为也融合结构 embedding
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
