"""
Main entry: Use full encoder to produce structural / semantic / fused three embeddings, and perform microservice partitioning using all three embeddings.
"""

import os
import asyncio
from typing import List

import torch
import torch.nn.functional as F

from microweaver.microservice_split.config import MicroWeaverConfig, get_config_by_graph_size
from microweaver.microservice_split.model import train_structural_encoder, train_full_encoder
from microweaver.microservice_split.model.code_graph_encoder import CodeClass, CodeGraphDataBuilder, \
    CodeGraphEncoder
from microweaver.microservice_split.partition.microservice_partition import \
    partition_from_multi_embeddings_iterative

import warnings

from microweaver.util.file_op import load_json, save_json
from microweaver.microservice_split.partition.agent_optimize import agent_optimize as _agent_optimize, \
    agent_analyze as _agent_analyze

warnings.filterwarnings("ignore")


def load_data(data_path: str) -> List[CodeClass]:
    nodes = load_json(data_path)
    classes = [
        CodeClass(
            id=node["id"],
            name=node["name"],
            description=node["description"],
            methods=node["methods"],
            dependencies=node["dependencies"],
            edge_types=node["edge_types"],
        )
        for node in nodes
    ]
    return classes


def print_debug_data(z_struct, edge_index):
    print("\n[Debug] Computing semantic vector similarity (by connection relationship)...")

    num_nodes = z_struct.size(0)

    # 1. Build undirected adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32, device=z_struct.device)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # Undirected graph
    adj.fill_diagonal_(0.0)  # Exclude self-loops

    # 2. Compute structural similarity matrix
    z_struct_norm = F.normalize(z_struct, p=2, dim=-1)  # [N, D]
    sim_matrix = torch.matmul(z_struct_norm, z_struct_norm.t())  # [N, N]

    # 3. Categorize node pairs
    # Directly connected node pairs (distance=1)
    direct_connected = (adj > 0).float()
    direct_mask = direct_connected.bool()

    # Node pairs at distance 2 (connected through one intermediate node)
    # Compute A^2, then subtract direct connections and self-loops
    adj_squared = torch.matmul(adj, adj)
    adj_squared.fill_diagonal_(0.0)  # Exclude self-loops
    distance_2 = (adj_squared > 0).float() * (1 - direct_connected)  # Distance 2 and not directly connected
    distance_2_mask = distance_2.bool()

    # Not connected node pairs (distance>2 or no path)
    not_connected = (1 - direct_connected - distance_2).float()
    not_connected.fill_diagonal_(0.0)  # Exclude self-loops
    not_connected_mask = not_connected.bool()

    # 4. Compute average similarity for each group
    # Directly connected node pairs
    direct_sims = sim_matrix[direct_mask]
    avg_direct = direct_sims.mean().item() if direct_sims.numel() > 0 else 0.0
    count_direct = direct_sims.numel()

    # Node pairs at distance 2
    distance_2_sims = sim_matrix[distance_2_mask]
    avg_distance_2 = distance_2_sims.mean().item() if distance_2_sims.numel() > 0 else 0.0
    count_distance_2 = distance_2_sims.numel()

    # Not connected node pairs
    not_connected_sims = sim_matrix[not_connected_mask]
    avg_not_connected = not_connected_sims.mean().item() if not_connected_sims.numel() > 0 else 0.0
    count_not_connected = not_connected_sims.numel()

    # 5. Output results
    print(f"\n[Debug] Semantic similarity statistics (by connection relationship):")
    print(f"{'Connection Type':<25} {'Node Pairs':<15} {'Avg Similarity':<15} {'Std Dev':<15}")
    print("-" * 70)
    print(
        f"{'Directly Connected (dist=1)':<25} {count_direct:<15} {avg_direct:<15.6f} {direct_sims.std().item():<15.6f}" if count_direct > 0 else f"{'Directly Connected (dist=1)':<25} {count_direct:<15} {'N/A':<15} {'N/A':<15}")
    print(
        f"{'1-hop Transit (dist=2)':<25} {count_distance_2:<15} {avg_distance_2:<15.6f} {distance_2_sims.std().item():<15.6f}" if count_distance_2 > 0 else f"{'1-hop Transit (dist=2)':<25} {count_distance_2:<15} {'N/A':<15} {'N/A':<15}")
    print(
        f"{'Not Connected (dist>2)':<25} {count_not_connected:<15} {avg_not_connected:<15.6f} {not_connected_sims.std().item():<15.6f}" if count_not_connected > 0 else f"{'Not Connected (dist>2)':<25} {count_not_connected:<15} {'N/A':<15} {'N/A':<15}")

    # Compute total node pairs (for verification)
    total_pairs = num_nodes * (num_nodes - 1) // 2  # Undirected graph, exclude self-loops
    print(f"\n[Debug] Verification:")
    print(f"  Total nodes: {num_nodes}")
    print(f"  Theoretical total pairs (undirected): {total_pairs}")
    print(f"  Actual counted pairs: {count_direct + count_distance_2 + count_not_connected}")
    print(f"  Difference: {total_pairs - (count_direct + count_distance_2 + count_not_connected)}")

    print("\n" + "=" * 70)


def main():
    config = MicroWeaverConfig()
    if not MicroWeaverConfig.skip_model_training:
        train_structural_encoder.main(config)
        train_full_encoder.main(config)

    classes = load_data(config.data_path)
    class_names = [cls.name for cls in classes]

    # Build graph data (using edge type weights)
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, texts, edge_weights = builder.build_graph_data(
        edge_type_weights=config.partition_config.edge_type_weights
    )

    print(f"\n[FullEncoder] Project Info:")
    print(f"  Classes: {len(classes)}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Edge types: {list(builder.edge_type_to_idx.keys())}")

    # Auto-select config based on data scale
    encoder_config = get_config_by_graph_size(len(classes))
    print(f"  Auto-selected config: {encoder_config.__class__.__name__}")

    # Initialize full encoder (structural + semantic + fusion)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = CodeGraphEncoder(
        structural_hidden_dim=encoder_config.structural.hidden_dim,
        structural_output_dim=encoder_config.structural.output_dim,
        semantic_output_dim=encoder_config.semantic.output_dim,
        final_output_dim=encoder_config.fusion.output_dim,
        num_edge_types=len(builder.edge_type_to_idx),
        num_structural_layers=encoder_config.structural.num_layers,
        num_heads=encoder_config.structural.num_heads,
        dropout=encoder_config.structural.dropout,
        code_encoder_model=encoder_config.semantic.model_name,
        freeze_code_encoder=encoder_config.semantic.freeze_encoder,
        structural_only=False,
    ).to(device)

    # Load full encoder pretrained weights
    pretrained_path = config.full_model_path
    if os.path.exists(pretrained_path):
        print(f"\n[FullEncoder] Detected pretrained model: {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            print("✓ full encoder weights loaded successfully")
        except Exception as e:
            print(f"! Failed to load full encoder pretrained weights: {e}, will use random initialization")
    else:
        print(f"\n[FullEncoder] Pretrained model not found ({pretrained_path}), using random initialization")
        print("  Tip: Run 'python -m split.train_full_encoder' to train full encoder")

    # Move data to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)
    pos_encoding = pos_encoding.to(device)
    edge_weights = edge_weights.to(device) if edge_weights is not None and edge_weights.numel() > 0 else None

    # Forward pass, get three embeddings
    print("\n[FullEncoder] Executing encoding (structural + semantic + fusion)...")
    model.eval()
    with torch.no_grad():
        z_struct, z_text, z_fused = model(
            x=x,
            edge_index=edge_index,
            edge_types=edge_types,
            pos_encoding=pos_encoding,
            texts=texts,
            edge_weights=edge_weights,
        )

    print("✓ Encoding completed!")
    print(f"  Structural embedding shape: {z_struct.shape}")
    print(f"  Semantic embedding shape: {z_text.shape}")
    print(f"  Fused embedding shape: {z_fused.shape}")

    print_debug_data(z_struct, edge_index)

    # ========== Microservice partitioning using three embeddings (iterative version) ==========
    print("\n[FullEncoder] Starting microservice partitioning...")

    # Read iteration and Agent switches from config
    max_iterations = config.partition_config.max_iterations
    enable_agent_cfg = config.partition_config.enable_agent_optimization

    # Try to load Agent optimization functions; if environment not configured (e.g., missing DASHSCOPE_API_KEY), fall back to disabling Agent
    agent_optimize_fn = None
    agent_analyze_fn = None
    try:
        agent_optimize_fn = _agent_optimize
        agent_analyze_fn = _agent_analyze
    except Exception as e:
        print(f"[FullEncoder] Agent optimization not enabled (reason: {e}), will only perform iterative solving without calling Agent")

    enable_agent = enable_agent_cfg and (agent_optimize_fn is not None)
    print(
        f"[FullEncoder] Config: max_iterations={max_iterations}, "
        f"enable_agent_optimization={enable_agent_cfg}, AgentEnabled={enable_agent}"
    )

    part_res = asyncio.run(
        partition_from_multi_embeddings_iterative(
            emb_struct=z_struct,
            emb_sem=z_text,
            emb_fused=z_fused,
            edge_index=edge_index,
            edge_weights=edge_weights,
            config=config.partition_config,
            agent_optimize_fn=agent_optimize_fn,
            agent_analyze_fn=agent_analyze_fn,
            node_names=class_names,
        )
    )

    print(f"[FullEncoder] Partition solver status: {part_res.solver_status}")
    print(f"[FullEncoder] Objective value: {part_res.objective_value:.4f}")

    # Print service groups and save
    groups = {k: [] for k in range(config.num_clusters)}
    for i, k in enumerate(part_res.assignments):
        if k >= 0:
            groups[k].append(classes[i].name)

    for k in groups:
        groups[k].sort()

    os.makedirs(os.path.dirname(config.result_path), exist_ok=True)
    save_json(groups, config.result_path)
    print(f"[FullEncoder] ✓ Microservice partition result saved to {config.result_path}")
    print("[FullEncoder] Service group results:")
    for k in range(config.num_clusters):
        print(f"  Service-{k}: {groups[k]}")

    return groups


if __name__ == "__main__":
    main()
