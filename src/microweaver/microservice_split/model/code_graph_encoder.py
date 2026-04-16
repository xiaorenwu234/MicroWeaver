"""
Encode graph nodes using structural and semantic information, and fuse them using cross-attention.
Use structural vectors as query vectors to emphasize the importance of structural information.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


@dataclass
class CodeClass:
    """Information of code class"""
    id: int
    name: str
    description: str
    methods: List[str]
    dependencies: List[int]  # IDs of other classes depended on
    edge_types: List[str]  # Edge types (e.g., "import", "inherit", "call", etc.)


class PositionalEncoding:
    """Graph positional encoding: based on shortest path and PageRank (supports directed graphs)"""
    def __init__(self, num_nodes: int, max_distance: int = 10, directed: bool = True):
        self.num_nodes = num_nodes
        self.max_distance = max_distance
        self.directed = directed

    def shortest_path_encoding(self, adj_matrix: np.ndarray) -> torch.Tensor:
        """
        Compute shortest path distance matrix as positional encoding
        """
        # Compute shortest paths between all node pairs
        dist_matrix = shortest_path(
            csr_matrix(adj_matrix),
            directed=self.directed,
            return_predecessors=False
        )

        # Replace infinity with max_distance
        dist_matrix[np.isinf(dist_matrix)] = self.max_distance

        # Normalize to [0, 1]
        dist_matrix = dist_matrix / self.max_distance

        return torch.from_numpy(dist_matrix).float()

    def pagerank_encoding(self, adj_matrix: np.ndarray, alpha: float = 0.85) -> torch.Tensor:
        """
        PageRank-based positional encoding
        """
        # Use directed graph, PageRank will reflect directed dependencies
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        pr = nx.pagerank(G, alpha=alpha)

        # Convert to vector
        pr_values = np.array([pr[i] for i in range(self.num_nodes)])

        # Normalize
        pr_values = (pr_values - pr_values.min()) / (pr_values.max() - pr_values.min() + 1e-8)

        return torch.from_numpy(pr_values).float().unsqueeze(1)

    def degree_encoding(self, adj_matrix: np.ndarray) -> torch.Tensor:
        """
        Degree-based positional encoding
        """
        degrees = np.sum(adj_matrix, axis=1)
        degrees = degrees / (degrees.max() + 1e-8)

        return torch.from_numpy(degrees).float().unsqueeze(1)


class EdgeTypeEmbedding(nn.Module):
    """Heterogeneous edge type embedding"""

    def __init__(self, num_edge_types: int, embedding_dim: int):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(num_edge_types, embedding_dim)

    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_types: [num_edges] Edge type indices

        Returns:
            [num_edges, embedding_dim] Edge embeddings
        """
        return self.edge_type_embedding(edge_types)


class GraphormerLayer(nn.Module):
    """
    Fixed Graphormer layer (enhanced structural encoding capability)
    Key improvements:
    - Enhanced attention bias strength and learnability
    - Fixed edge weight bias logic
    - Added self-loop bias and attention temperature coefficient
    - Optimized SPD encoding mapping method
    - Added structural normalization
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_edge_types: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_dist: int = 10,
            max_degree: int = 512,
            bias_scale: float = 10.0,  # Attention bias scaling factor (enhances structural weight)
            attn_temp: float = 0.5,  # Attention temperature coefficient (controls distribution sharpness)
    ):
        super().__init__()
        assert in_channels == out_channels, "GraphormerLayer requires in_channels == out_channels"
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.hidden_dim = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.max_dist = max_dist
        self.max_degree = max_degree
        self.bias_scale = bias_scale  # New: bias intensity scaling
        self.attn_temp = attn_temp  # New: attention temperature

        # QKV projection
        self.q_proj = nn.Linear(out_channels, out_channels)
        self.k_proj = nn.Linear(out_channels, out_channels)
        self.v_proj = nn.Linear(out_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)

        # Attention bias (enhanced version)
        self.spd_bias_table = nn.Embedding(self.max_dist + 2, num_heads)  # +2 reserved for self-loop/ultra-long distance
        self.edge_type_bias = nn.Embedding(num_edge_types, num_heads)
        self.self_loop_bias = nn.Parameter(torch.zeros(num_heads))  # New: self-loop bias

        # Centrality encoding (add learnable scaling)
        self.in_degree_emb = nn.Embedding(self.max_degree + 1, out_channels)
        self.out_degree_emb = nn.Embedding(self.max_degree + 1, out_channels)
        self.degree_scale = nn.Parameter(torch.ones(1))  # New: centrality encoding scaling

        # Pre-normalization + FFN
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialize self-loop bias to positive value (enhance self-attention)
        nn.init.constant_(self.self_loop_bias, 1.0)

    @staticmethod
    def _compute_degrees(num_nodes: int, edge_index: torch.Tensor) -> Tuple[Tensor, Tensor]:
        device = edge_index.device if edge_index.numel() > 0 else torch.device('cpu')
        deg_out = torch.zeros(num_nodes, dtype=torch.long, device=device)
        deg_in = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            deg_out.scatter_add_(0, src, torch.ones_like(src))
            deg_in.scatter_add_(0, dst, torch.ones_like(dst))
        return deg_in, deg_out

    def _build_attention_bias(
            self,
            num_nodes: int,
            edge_index: torch.Tensor,
            edge_types: torch.Tensor,
            pos_encoding: Tensor,
            device: torch.device,
            edge_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Fixed attention bias construction:
        1. Enhanced bias intensity
        2. Fixed edge weight logic
        3. Added self-loop bias
        4. Optimized SPD mapping
        """
        # Initialize bias matrix [H, N, N]
        attn_bias = torch.zeros(self.num_heads, num_nodes, num_nodes, device=device)

        # 1. Self-loop bias (enhance attention of node to itself)
        self_loop_mask = torch.eye(num_nodes, dtype=torch.bool, device=device)  # [N,N]
        for h in range(self.num_heads):
            attn_bias[h][self_loop_mask] += self.self_loop_bias[h]

        # 2. SPD bias (optimized mapping method)
        if pos_encoding is not None and pos_encoding.size(1) >= num_nodes:
            spd_norm = pos_encoding[:, :num_nodes]  # [N, N]
            # Optimization: piecewise mapping of SPD to retain more details
            spd_bucket = torch.where(
                spd_norm == 0,  # Self-loop
                torch.tensor(0, device=device),
                torch.where(
                    spd_norm > 1.0,  # Ultra-long distance
                    torch.tensor(self.max_dist + 1, device=device),
                    (spd_norm * self.max_dist).clamp(1, self.max_dist).long()
                )
            )
            spd_bias = self.spd_bias_table(spd_bucket)  # [N, N, H]
            # Scale bias intensity
            attn_bias = attn_bias + spd_bias.permute(2, 0, 1) * self.bias_scale

        # 3. Edge type + edge weight bias (fixed logic)
        if edge_index.numel() > 0 and edge_types.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            e_bias = self.edge_type_bias(edge_types)  # [E, H]

            # Fix: edge weights should enhance bias (not scale), higher weight means larger bias
            if edge_weights is not None:
                # Map weight from [0,1] to [0, 2*bias_scale] to enhance attention of high-weight edges
                weight_scale = edge_weights.unsqueeze(1) * 2 * self.bias_scale  # [E,1]
                e_bias = e_bias + weight_scale  # Additive enhancement instead of multiplicative scaling

            # Accumulate to bias matrix (with scaling)
            for h in range(self.num_heads):
                attn_bias[h].index_put_((src, dst), e_bias[:, h] * self.bias_scale, accumulate=True)

        return attn_bias

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_types: Tensor,
            pos_encoding: Tensor,
            edge_weights: Optional[Tensor] = None,
    ) -> Tensor:
        N, D = x.size()
        device = x.device

        # 1) Centrality encoding (add learnable scaling)
        deg_in, deg_out = self._compute_degrees(N, edge_index)
        deg_in = deg_in.clamp(max=self.max_degree)
        deg_out = deg_out.clamp(max=self.max_degree)

        degree_emb = (self.in_degree_emb(deg_in) + self.out_degree_emb(deg_out)) * self.degree_scale
        x = x + degree_emb

        # 2) Pre-LN + Multi-head attention
        x_norm = self.ln1(x)

        # QKV projection + reshape
        q = self.q_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]
        k = self.k_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]
        v = self.v_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]

        # Build attention bias
        attn_bias = self._build_attention_bias(N, edge_index, edge_types, pos_encoding, device, edge_weights)

        # Compute attention scores (add temperature coefficient)
        if hasattr(F, 'scaled_dot_product_attention'):
            # Flash Attention path (with temperature and bias)
            q_t = q.unsqueeze(0) / self.attn_temp  # Temperature scaling
            k_t = k.unsqueeze(0) / self.attn_temp
            v_t = v.unsqueeze(0)
            attn_mask = attn_bias.unsqueeze(0)

            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
            attn_out = attn_out.squeeze(0)
        else:
            # Manual computation path (with temperature and bias)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (np.sqrt(self.head_dim) * self.attn_temp)
            scores = scores + attn_bias  # Bias accumulation
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            attn_out = torch.matmul(attn, v)

        # Merge heads + projection
        attn_out = attn_out.transpose(0, 1).contiguous().view(N, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_dropout(attn_out)

        # Residual connection
        x = x + attn_out

        # 3) FFN
        y = self.ln2(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        out = x + y

        return out


class StructuralEncoder(nn.Module):
    """
    Enhanced Structural Encoder: adds layer normalization and output structure constraints
    """

    def __init__(
            self,
            node_feature_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_edge_types: int,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1,
            bias_scale: float = 10.0,
            attn_temp: float = 0.5,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initial node feature projection (adds bias and normalization)
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Graphormer layers (enhanced version)
        self.graphormer_layers = nn.ModuleList([
            GraphormerLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_edge_types=num_edge_types,
                num_heads=num_heads,
                dropout=dropout,
                bias_scale=bias_scale,
                attn_temp=attn_temp
            )
            for _ in range(num_layers)
        ])

        # Output projection (add structural normalization)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)

        )

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_types: Tensor,
            pos_encoding: Tensor,
            edge_weights: Optional[Tensor] = None
    ) -> Tensor:
        # Embed initial features
        x = self.node_embedding(x)

        # Pass through Graphormer layers
        for layer in self.graphormer_layers:
            x = layer(x, edge_index, edge_types, pos_encoding, edge_weights)

        # Output projection + L2 normalization (enhance interpretability of cosine similarity)
        x = self.output_proj(x)
        x = F.normalize(x, p=2, dim=-1)  # New: L2 normalization of output vectors

        return x

    def load_pretrained(self, checkpoint_path: str, device: torch.device | None = None):
        """
        Load pretrained structural encoder weights

        Args:
            checkpoint_path: Pretrained weights file path
            device: Device to load to (default is current model device)
        """
        if device is None:
            device = next(self.parameters()).device

        state_dict = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(state_dict, strict=True)
        print(f"✓ Loaded pretrained structural encoder: {checkpoint_path}")



class SemanticEncoder(nn.Module):
    """
    Semantic encoder: uses BGE-M3 model for semantic encoding
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        output_dim: int = 256,
        freeze_encoder: bool = False,
        batch_size: int = 32
    ):
        super().__init__()

        self.model_name = model_name
        self.batch_size = batch_size

        # Load BGE-M3 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Get encoder dimension
        encoder_dim = self.encoder.config.hidden_size

        # Projection layer: project BGE-M3 output dimension to target dimension
        self.projection = nn.Linear(encoder_dim, output_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: List of texts (class names, comments, method signatures, etc.)

        Returns:
            [num_texts, output_dim] Semantic representation of texts
        """
        device = next(self.encoder.parameters()).device
        all_embeddings = []

        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize and encode
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,  # BGE-M3 supports longer sequences
                return_tensors="pt"
            )

            # Move to same device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

            # Get encoding
            # Critical fix: even if freeze_encoder=True, should not use no_grad()
            # Freezing parameters means parameters don't update, but output can still have gradients so projection layer can train
            # Only when encoder parameters requires_grad=False, gradients won't backprop to encoder parameters, but will backprop to projection layer
            model_output = self.encoder(**encoded_input)

            # BGE-M3 uses mean pooling
            # Check if pooler_output exists
            if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
                # If pooler_output exists, use directly
                embeddings = model_output.pooler_output
            else:
                # Otherwise use mean pooling
                embeddings = model_output.last_hidden_state
                # Mask padding positions
                attention_mask = encoded_input['attention_mask']
                # Expand attention_mask dimensions to match embeddings
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                # Compute average of valid tokens
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            # Project to target dimension
            semantic_repr = self.projection(embeddings)

            all_embeddings.append(semantic_repr)

        # Merge all batches
        return torch.cat(all_embeddings, dim=0)


class CrossAttentionFusion(nn.Module):
    """
    Cross-modal Fusion Module: Uses Cross-Attention to fuse structural and semantic information

    Optimizations:
    1. Adaptive attention scaling: learns attention scale factor for better numerical stability
    2. Attention mask support: handles invalid nodes or padding values
    3. Better activation function: uses GELU instead of ReLU
    4. Compatible with PyTorch 2.0+ Flash Attention: improves performance
    """

    def __init__(
        self,
        structural_dim: int,
        semantic_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_adaptive_scale: bool = True
    ):
        super().__init__()

        self.structural_dim = structural_dim
        self.semantic_dim = semantic_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_adaptive_scale = use_adaptive_scale

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Semantic adapter layer: adapts semantic representation near structural dimension, preserving more semantic information
        # If semantic and structural dimensions differ significantly, first adapt to structural dimension, then project to output_dim
        if semantic_dim != structural_dim:
            self.semantic_adapter = nn.Sequential(
                nn.Linear(semantic_dim, structural_dim),
                nn.LayerNorm(structural_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            adapted_semantic_dim = structural_dim
        else:
            self.semantic_adapter = nn.Identity()
            adapted_semantic_dim = semantic_dim

        # Query comes from structural representation
        self.query_proj = nn.Linear(structural_dim, output_dim)

        # Key and Value come from semantic representation (using adapted dimension)
        self.key_proj = nn.Linear(adapted_semantic_dim, output_dim)
        self.value_proj = nn.Linear(adapted_semantic_dim, output_dim)

        # Output projection
        self.out_proj = nn.Linear(output_dim, output_dim)

        # Residual projection (projects structural representation to output_dim for residual addition)
        self.residual_proj = nn.Linear(structural_dim, output_dim) if structural_dim != output_dim else nn.Identity()

        # Normalization layers
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        # Feed-forward network (uses GELU activation)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),  # Use GELU instead of ReLU for better performance
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim)
        )

        # Adaptive attention scaling factor
        if self.use_adaptive_scale:
            self.attention_scale = nn.Parameter(torch.log(torch.tensor(self.head_dim ** 0.5)))
        else:
            self.attention_scale = torch.tensor(self.head_dim ** 0.5)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Parameter initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize model parameters to ensure training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier uniform initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm weight initialized to 1, bias initialized to 0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        structural_repr: torch.Tensor,
        semantic_repr: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-modal fusion forward pass

        Args:
            structural_repr: [num_nodes, structural_dim] Structural representation of nodes
            semantic_repr: [num_nodes, semantic_dim] Semantic representation of nodes
            attention_mask: Optional, [num_nodes, num_nodes] or [num_heads, num_nodes, num_nodes]
                Attention mask, positions with value True or 1 indicate need to be masked, will be set to negative infinity

        Returns:
            [num_nodes, output_dim] Fused node representation
        """
        num_nodes = structural_repr.size(0)

        # 1. Semantic adaptation: adapt semantic representation near structural dimension
        semantic_repr_adapted = self.semantic_adapter(semantic_repr)  # [num_nodes, adapted_semantic_dim]

        # 2. Linear projection
        Q = self.query_proj(structural_repr)  # [num_nodes, output_dim]
        K = self.key_proj(semantic_repr_adapted)  # [num_nodes, output_dim]
        V = self.value_proj(semantic_repr_adapted)  # [num_nodes, output_dim]

        # 3. Reshape to multi-head format: [num_nodes, num_heads, head_dim]
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)

        # 4. Process attention mask
        if attention_mask is not None:
            # Convert mask to float type, and set True/1 values to negative infinity
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.float().masked_fill(attention_mask, float('-inf'))
            else:
                attention_mask = attention_mask.masked_fill(attention_mask != 0, float('-inf'))

            # Adapt to multi-head dimensions: [num_heads, num_nodes, num_nodes]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).expand(self.num_heads, -1, -1)

        # 5. Compute attention weights
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch requires scale to be float, here convert adaptive scaling to Python float
            if self.use_adaptive_scale:
                scale = float(torch.exp(self.attention_scale).item())
            else:
                scale = float(self.attention_scale)

            # Use Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q.unsqueeze(0).transpose(1, 2),
                K.unsqueeze(0).transpose(1, 2),
                V.unsqueeze(0).transpose(1, 2),
                attn_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
                scale=scale  # Pass adaptive scaling factor (float)
            ).transpose(1, 2).squeeze(0)
        else:
            # Compatible with older PyTorch versions: manual attention computation
            # Compute attention scores: [num_nodes, num_heads, num_nodes]
            Qh = Q.transpose(0, 1)  # [H, N, Hd]
            Kh = K.transpose(0, 1)  # [H, N, Hd]
            Vh = V.transpose(0, 1)  # [H, N, Hd]
            scores = torch.matmul(Qh, Kh.transpose(-2, -1))

            # Apply adaptive scaling factor
            scale = torch.exp(self.attention_scale) if hasattr(self, 'attention_scale') else np.sqrt(self.head_dim)
            scores = scores / scale

            # Apply attention mask
            if attention_mask is not None:
                mask = attention_mask if attention_mask.dim() == 3 else attention_mask.unsqueeze(0)
                scores = scores + mask

            # Softmax normalization
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values: [H, N, Hd]
            attn_output = torch.matmul(attn_weights, Vh)  # [H, N, Hd]
            attn_output = attn_output.transpose(0, 1)  # [N, H, Hd]

        # 6. Merge multi-heads: [num_nodes, output_dim]
        attn_output = attn_output.reshape(num_nodes, self.output_dim)

        # 7. Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # 8. Residual connection and layer normalization
        out = self.norm1(attn_output + self.residual_proj(structural_repr))

        # 9. Feed-forward network
        ffn_output = self.ffn(out)
        ffn_output = self.dropout(ffn_output)

        # 10. Residual connection and layer normalization
        out = self.norm2(ffn_output + out)

        return out


class CodeGraphEncoder(nn.Module):
    """
    Complete code graph encoding system
    """

    def __init__(
        self,
        structural_hidden_dim: int = 256,
        structural_output_dim: int = 256,
        semantic_output_dim: int = 256,
        final_output_dim: int = 512,
        num_edge_types: int = 5,
        num_structural_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        code_encoder_model: str = "BAAI/bge-m3",
        freeze_code_encoder: bool = False,
        structural_only: bool = False
    ):
        super().__init__()

        self.structural_output_dim = structural_output_dim
        self.semantic_output_dim = semantic_output_dim
        self.final_output_dim = final_output_dim
        self.structural_only = structural_only

        # Structural encoder
        self.structural_encoder = StructuralEncoder(
            node_feature_dim=1,  # Initial feature dimension
            hidden_dim=structural_hidden_dim,
            output_dim=structural_output_dim,
            num_edge_types=num_edge_types,
            num_layers=num_structural_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        if not self.structural_only:
            # Semantic encoder
            self.semantic_encoder = SemanticEncoder(
                model_name=code_encoder_model,
                output_dim=semantic_output_dim,
                freeze_encoder=freeze_code_encoder
            )

            # Cross-modal fusion
            self.fusion = CrossAttentionFusion(
                structural_dim=structural_output_dim,
                semantic_dim=semantic_output_dim,
                output_dim=final_output_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.semantic_encoder = None
            self.fusion = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        pos_encoding: torch.Tensor,
        texts: Optional[List[str]] = None,
        edge_weights: Optional[torch.Tensor] = None,
        fusion_attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: [num_nodes, 1] Initial node features
            edge_index: [2, num_edges] Edge indices
            edge_types: [num_edges] Edge types
            pos_encoding: [num_nodes, pos_dim] Positional encoding
            texts: Text corresponding to nodes (class name + comments + method signatures)
            edge_weights: [num_edges] Edge weights (optional), based on edge type weights

        Returns:
            [num_nodes, final_output_dim] Final node vectors
        """
        # Structural encoding (using edge weights)
        structural_repr = self.structural_encoder(x, edge_index, edge_types, pos_encoding, edge_weights)

        if self.structural_only:
            return structural_repr

        assert texts is not None, "texts cannot be empty, text input required when structural_only=False"
        semantic_repr = self.semantic_encoder(texts)

        if fusion_attention_mask is None:
            num_nodes = structural_repr.size(0)
            fusion_attention_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=structural_repr.device)
        fused_repr = self.fusion(structural_repr, semantic_repr, attention_mask=fusion_attention_mask)
        return structural_repr, semantic_repr, fused_repr


class CodeGraphDataBuilder:
    """
    Build graph data from code class information
    """

    def __init__(self, classes: List[CodeClass]):
        self.classes = classes
        self.num_nodes = len(classes)
        self.edge_type_to_idx = {}
        self.build_edge_type_mapping()

    def build_edge_type_mapping(self):
        """Build mapping from edge type to index"""
        edge_types_set = set()
        for cls in self.classes:
            edge_types_set.update(cls.edge_types)

        for i, edge_type in enumerate(sorted(edge_types_set)):
            self.edge_type_to_idx[edge_type] = i

    def build_graph_data(self, edge_type_weights=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
        """
        Build graph data

        Args:
            edge_type_weights: Edge type weight configuration, if provided applies weights to edges

        Returns:
            x: Node features
            edge_index: Edge indices
            edge_types: Edge types
            pos_encoding: Positional encoding
            texts: Node texts
            edge_weights: Edge weights (based on type)
        """
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        edge_list = []
        edge_type_list = []
        edge_weight_list = []

        for src_id, cls in enumerate(self.classes):
            for dst_id, edge_type in zip(cls.dependencies, cls.edge_types):
                # Get edge weight
                weight = 1.0
                if edge_type_weights is not None:
                    weight = edge_type_weights.get_weight(edge_type)

                adj_matrix[src_id, dst_id] = weight
                edge_list.append([src_id, dst_id])
                edge_type_list.append(self.edge_type_to_idx[edge_type])
                edge_weight_list.append(weight)

        # Convert to tensor
        x = torch.ones(self.num_nodes, 1, dtype=torch.float32)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_types = torch.tensor(edge_type_list, dtype=torch.long)
            edge_weights = torch.tensor(edge_weight_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)
            edge_weights = torch.zeros(0, dtype=torch.float32)

        # Compute positional encoding
        pos_encoder = PositionalEncoding(self.num_nodes)

        # Combine multiple positional encodings
        sp_encoding = pos_encoder.shortest_path_encoding(adj_matrix)
        pr_encoding = pos_encoder.pagerank_encoding(adj_matrix)
        deg_encoding = pos_encoder.degree_encoding(adj_matrix)

        pos_encoding = torch.cat([sp_encoding, pr_encoding, deg_encoding], dim=1)

        texts = []
        for cls in self.classes:
            class_name = cls.name
            description = cls.description if cls.description else "No description"
            if cls.methods:
                methods_list = cls.methods[:5]
                methods_text = f"Main methods: {', '.join(methods_list)}"
                if len(cls.methods) > 5:
                    methods_text += f" etc. total {len(cls.methods)} methods"
            else:
                methods_text = ""
            if methods_text:
                text = f"{class_name}. {description}. {methods_text}"
            else:
                text = f"{class_name}. {description}"
            texts.append(text)
        return x, edge_index, edge_types, pos_encoding, texts, edge_weights