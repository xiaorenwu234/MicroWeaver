"""
配置文件：定义系统的各种参数

该模块包含以下配置类：
1. DataConfig: 数据路径配置
2. PartitionConfig: 微服务划分配置
3. StructuralEncoderConfig: 结构编码器配置
4. SemanticEncoderConfig: 语义编码器配置
5. FusionConfig: 融合模块配置
6. HierarchicalEncoderConfig: 分层编码器配置
7. CodeGraphEncoderConfig: 完整编码器配置（包含上述三个编码器）

预定义配置模板：
- SMALL_GRAPH_CONFIG: 小规模图 (< 100 节点)
- MEDIUM_GRAPH_CONFIG: 中规模图 (100-1000 节点)

使用方式：
1. 自动选择: config = get_config_by_graph_size(num_nodes)
2. 手动选择: config = MEDIUM_GRAPH_CONFIG
3. 自定义: config = CodeGraphEncoderConfig(...)
"""
import os
from dataclasses import dataclass

from microweaver.config import BaseConfig
from microweaver.util.env import get_env_numeric, get_env_boolean


@dataclass
class EdgeTypeWeightConfig:
    """边类型权重配置"""
    type_weights: dict = None  # 边类型到权重的映射，如 {"call": 1.0, "import": 0.8}

    def __post_init__(self):
        if self.type_weights is None:
            # 默认权重配置
            self.type_weights = {
                "call": 0.8,  # 方法调用：中等权重
                "extends": 1.0,  # 继承关系：最高权重
            }

    def get_weight(self, edge_type: str) -> float:
        """获取指定边类型的权重，默认为1.0"""
        return self.type_weights.get(edge_type, 1.0)


@dataclass
class StructuralEncoderConfig:
    """结构编码器配置"""
    node_feature_dim: int = 1
    hidden_dim: int = 256
    output_dim: int = 256
    num_edge_types: int = 5
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class SemanticEncoderConfig:
    """语义编码器配置"""
    model_name: str = "BAAI/bge-m3"
    output_dim: int = 256
    freeze_encoder: bool = False
    max_length: int = 512


@dataclass
class FusionConfig:
    """融合模块配置"""
    structural_dim: int = 256
    semantic_dim: int = 256
    output_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class CodeGraphEncoderConfig:
    """完整编码器配置"""
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

# 小规模图配置（< 100 节点）
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

# 中规模图配置（100-1000 节点）
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
    根据图的大小自动选择配置
    Args:
        num_nodes: 图中的节点数
    Returns:
        推荐的配置对象
    """
    if num_nodes < 100:
        return SMALL_GRAPH_CONFIG
    else:
        return MEDIUM_GRAPH_CONFIG


@dataclass
class PartitionConfig:
    """划分配置"""
    num_communities: int = BaseConfig.num_clusters
    random_seed: int = 42
    alpha = get_env_numeric("alpha", 5.0)  # 结构内聚权重（默认5.0）
    beta = get_env_numeric("beta", 1.0)  # 语义内聚权重（默认1.0）
    gamma = get_env_numeric("gamma", 3.0)  # 跨服务耦合惩罚（默认3.0）
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
    edge_type_weights: EdgeTypeWeightConfig = None  # 边类型权重配置

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
