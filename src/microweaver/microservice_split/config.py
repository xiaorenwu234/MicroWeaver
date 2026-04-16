"""
Configuration file: defines various parameters of the system

This module contains the following configuration classes:
1. DataConfig: Data path configuration
2. PartitionConfig: Microservice partitioning configuration
3. StructuralEncoderConfig: Structural encoder configuration
4. SemanticEncoderConfig: Semantic encoder configuration
5. FusionConfig: Fusion module configuration
6. HierarchicalEncoderConfig: Hierarchical encoder configuration
7. CodeGraphEncoderConfig: Complete encoder configuration (contains the above three encoders)

Predefined configuration templates:
- SMALL_GRAPH_CONFIG: Small-scale graph (< 100 nodes)
- MEDIUM_GRAPH_CONFIG: Medium-scale graph (100-1000 nodes)

Usage:
1. Auto-select: config = get_config_by_graph_size(num_nodes)
2. Manual select: config = MEDIUM_GRAPH_CONFIG
3. Custom: config = CodeGraphEncoderConfig(...)
"""
import os
from dataclasses import dataclass

from microweaver.config import BaseConfig
from microweaver.util.env import get_env_numeric, get_env_boolean


@dataclass
class EdgeTypeWeightConfig:
    """Edge type weight configuration"""
    type_weights: dict = None  # Mapping from edge type to weight, e.g., {"call": 1.0, "import": 0.8}

    def __post_init__(self):
        if self.type_weights is None:
            # Default weight configuration
            self.type_weights = {
                "call": 0.8,  # Method call: medium weight
                "extends": 1.0,  # Inheritance: highest weight
            }

    def get_weight(self, edge_type: str) -> float:
        """Get weight for specified edge type, default is 1.0"""
        return self.type_weights.get(edge_type, 1.0)


@dataclass
class StructuralEncoderConfig:
    """Structural encoder configuration"""
    node_feature_dim: int = 1
    hidden_dim: int = 256
    output_dim: int = 256
    num_edge_types: int = 5
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class SemanticEncoderConfig:
    """Semantic encoder configuration"""
    model_name: str = "BAAI/bge-m3"
    output_dim: int = 256
    freeze_encoder: bool = False
    max_length: int = 512


@dataclass
class FusionConfig:
    """Fusion module configuration"""
    structural_dim: int = 256
    semantic_dim: int = 256
    output_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class CodeGraphEncoderConfig:
    """Complete encoder configuration"""
    structural: StructuralEncoderConfig = None
    semantic: SemanticEncoderConfig = None
    fusion: FusionConfig = None

    def __post_init__(self):
        if self.structural is None:
            self.structural = StructuralEncoderConfig()
        if self.semantic is None:
            self.semantic = SemanticEncoderConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()

# Small-scale graph configuration (< 100 nodes)
SMALL_GRAPH_CONFIG = CodeGraphEncoderConfig(
    structural=StructuralEncoderConfig(
        hidden_dim=256,
        output_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.1
    ),
    semantic=SemanticEncoderConfig(
        model_name="BAAI/bge-m3",
        output_dim=256,
        freeze_encoder=False
    ),
    fusion=FusionConfig(
        structural_dim=256,
        semantic_dim=256,
        output_dim=512,
        num_heads=8,
        dropout=0.1
    )
)

# Medium-scale graph configuration (100-1000 nodes)
MEDIUM_GRAPH_CONFIG = CodeGraphEncoderConfig(
    structural=StructuralEncoderConfig(
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ),
    semantic=SemanticEncoderConfig(
        model_name="BAAI/bge-m3",
        output_dim=256,
        freeze_encoder=True
    ),
    fusion=FusionConfig(
        structural_dim=256,
        semantic_dim=256,
        output_dim=512,
        num_heads=4,
        dropout=0.1
    )
)


def get_config_by_graph_size(num_nodes: int) -> CodeGraphEncoderConfig:
    """
    Automatically select configuration based on graph size
    Args:
        num_nodes: Number of nodes in graph
    Returns:
        Recommended configuration object
    """
    if num_nodes < 100:
        return SMALL_GRAPH_CONFIG
    else:
        return MEDIUM_GRAPH_CONFIG


@dataclass
class PartitionConfig:
    """Partition configuration"""
    num_communities: int = BaseConfig.num_clusters
    random_seed: int = 42
    alpha = get_env_numeric("alpha", 5.0)  # Structural cohesion weight (default 5.0)
    beta = get_env_numeric("beta", 1.0)  # Semantic cohesion weight (default 1.0)
    gamma = get_env_numeric("gamma", 3.0)  # Cross-service coupling penalty (default 3.0)
    beta_struct = get_env_numeric("beta_struct", 1.0)
    beta_sem = get_env_numeric("beta_sem", 2.0)
    beta_fused = get_env_numeric("beta_fused", 1.0)
    must_link = None
    cannot_link = None
    symmetric_struc = True
    size_lower = [int(get_env_numeric("min_size", 5)) for _ in range(num_communities)]
    size_upper = [int(get_env_numeric("max_size", 35)) for _ in range(num_communities)]
    pair_threshold = get_env_numeric("pair_threshold", 0.95)
    time_limit_sec = get_env_numeric("time_limit", 1200)
    max_iterations = get_env_numeric("max_iterations", 1)
    scale = 1000
    num_cpu = get_env_numeric("num_cpu", 8)

    enable_agent_optimization = get_env_boolean("ENABLE_AGENT_OPTIMIZATION", True)
    edge_type_weights: EdgeTypeWeightConfig = None  # Edge type weight configuration

    print(f"PartitionConfig initialized with alpha={alpha}, beta={beta}, gamma={gamma}, "
          f"beta_struct={beta_struct}, beta_sem={beta_sem}, beta_fused={beta_fused}, "
          f"size_lower={size_lower}, size_upper={size_upper}, pair_threshold={pair_threshold}, "
          f"time_limit_sec={time_limit_sec}, max_iterations={max_iterations}, num_cpu={num_cpu}, "
          f"enable_agent_optimization={enable_agent_optimization}")

    def __post_init__(self):
        if self.edge_type_weights is None:
            self.edge_type_weights = EdgeTypeWeightConfig()


class MicroWeaverConfig(BaseConfig):
    model_folder_path = os.path.join(BaseConfig.data_folder_path, "model")
    structural_model_path = os.path.join(model_folder_path, "structural_encoder.pt")
    full_model_path = os.path.join(model_folder_path, "full_encoder.pt")
    skip_model_training = get_env_boolean("SKIP_MODEL_TRAINING", False)
    partition_config = PartitionConfig()
