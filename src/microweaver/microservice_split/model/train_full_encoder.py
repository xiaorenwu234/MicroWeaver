"""
Joint training of structural encoder + semantic encoder + cross-attention fusion:
- Structural part: uses current Graphormer structural encoding + multi-positive InfoNCE + (optional) Laplacian smoothing
- Semantic part: uses current BGE-M3 semantic encoding
- Cross-attention: uses CrossAttentionFusion to align structure/text, making cross-attn actually effective

Training objective:
    loss = loss_struct + lambda_lap * loss_lap + lambda_align * loss_align

Where:
- loss_struct: multi-positive InfoNCE same as structural specialized training script
- loss_lap:   Laplacian smoothing term (optional)
- loss_align: structure-text alignment loss (aligns structural vector with corresponding text vector for each class)
"""

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from microweaver.microservice_split.config import MicroWeaverConfig, get_config_by_graph_size, PartitionConfig
from microweaver.microservice_split.model.code_graph_encoder import CodeClass, CodeGraphDataBuilder, CodeGraphEncoder
from microweaver.util.file_op import load_json


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


def build_graph(classes: List[CodeClass], partition_config: PartitionConfig):
    """
    Build graph + positive sample mask / weight matrix:
    - adj: undirected 0/1 adjacency matrix, used for Laplacian smoothing
    - pos_mask: same as adj, used for multi-positive InfoNCE
    - pos_weight: positive sample weight matrix weighted by edge type
    """
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, texts, edge_weights = builder.build_graph_data(
        edge_type_weights=partition_config.edge_type_weights
    )

    num_nodes = len(classes)
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    pos_weight = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    if edge_index.numel() > 0:
        src, dst = edge_index
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0

        if edge_weights is not None and edge_weights.numel() > 0:
            pos_weight[src, dst] = edge_weights
            pos_weight[dst, src] = edge_weights
        else:
            pos_weight[src, dst] = 1.0
            pos_weight[dst, src] = 1.0

    adj.fill_diagonal_(0.0)
    pos_weight.fill_diagonal_(0.0)

    # pos_mask: undirected adjacency
    pos_mask = (adj > 0).float()

    return (
        builder,
        x,
        edge_index,
        edge_types,
        pos_encoding,
        texts,
        adj,
        pos_mask,
        pos_weight,
        edge_weights,
    )


def multi_positive_infonce(
        z: torch.Tensor,
        pos_mask: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        temperature: float = 0.2,
) -> torch.Tensor:
    """
    Multi-positive InfoNCE (consistent with structural specialized training version):
        For each node i, positive samples are its neighbor set P(i).
        loss_i = -log( sum_{j in P(i)} w_ij * exp(sim(i,j)/tau) / sum_{k != i} exp(sim(i,k)/tau) )
    """
    z = F.normalize(z, p=2, dim=-1)
    sim = torch.matmul(z, z.t())
    logits = sim / temperature

    N = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    denom_mask = ~eye

    pos_mask = pos_mask.to(z.device).bool()
    weight = pos_weight.to(z.device) if pos_weight is not None else pos_mask.float()
    has_pos = (weight > 0).any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    row_max = logits.max(dim=1, keepdim=True).values
    logits = logits - row_max

    exp_logits = torch.exp(logits)
    denom = (exp_logits * denom_mask.float()).sum(dim=1)
    numer = (exp_logits * weight).sum(dim=1)

    numer = numer[has_pos]
    denom = denom[has_pos] + 1e-12

    loss = -torch.log(numer / denom + 1e-12).mean()
    return loss


def laplacian_smoothing(z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Laplacian smoothing term: sum_{(i,j) in E} ||z_i - z_j||^2
    """
    if adj.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    D = torch.diag(adj.sum(dim=1))
    L = D - adj
    loss = torch.trace(z.t() @ L @ z) / (adj.sum() + 1e-12)
    return loss


def struct_text_alignment_loss(
        z_struct: torch.Tensor,
        z_text: torch.Tensor,
        temperature: float = 0.1,
) -> torch.Tensor:
    """
    Structure-text alignment loss:
    - z_struct[i] should align with z_text[i]
    - Uses InfoNCE / classification-style alignment:
        sim = z_struct @ z_text^T / tau
        labels = arange(N)
        loss = CrossEntropy(sim, labels)
    """
    # Normalize before alignment for better stability
    z_s = F.normalize(z_struct, p=2, dim=-1)
    z_t = F.normalize(z_text, p=2, dim=-1)

    sim = torch.matmul(z_s, z_t.t()) / temperature  # [N, N]
    labels = torch.arange(z_struct.size(0), device=z_struct.device)
    return F.cross_entropy(sim, labels)


def fused_alignment_loss(
        z_fused: torch.Tensor,
        z_target: torch.Tensor,
        proj_layer: nn.Module | None = None,
) -> torch.Tensor:
    """
    Fused representation alignment loss (per-sample alignment, supports different dimensions):
    - z_fused[i] should align with z_target[i] (per-sample alignment, not global matching)
    - Uses negative log-likelihood of per-sample cosine similarity as loss
    - Supports cases where z_fused and z_target have different dimensions (adapted via projection layer)

    Fix: Changed from global CrossEntropy to per-sample alignment, ensuring node i's fused representation aligns with node i's semantic/structural representation
    Uses projection layer instead of truncation to avoid information loss
    """
    # Use projection layer to project target representation to fused representation dimension
    if proj_layer is not None:
        z_t_projected = proj_layer(z_target)  # [N, fused_dim]
    else:
        z_t_projected = z_target

    # Normalize (normalize after projection)
    z_f = F.normalize(z_fused, p=2, dim=-1)  # [N, fused_dim]
    z_t = F.normalize(z_t_projected, p=2, dim=-1)  # [N, fused_dim]

    # Per-sample alignment: compute similarity between each node i's fused representation and corresponding target representation
    # Use dot product similarity (already normalized, equivalent to cosine similarity)
    sim_per_sample = (z_f * z_t).sum(dim=-1)  # [N] per-sample similarity

    # Convert to loss: negative log-likelihood (higher similarity, lower loss)
    loss_per_sample = -torch.log(sim_per_sample.clamp(min=1e-8))  # [N]

    return loss_per_sample.mean()


class FullEncoderTrainer:
    def __init__(
            self,
            encoder: CodeGraphEncoder,
            lr: float = 1e-4,
            weight_decay: float = 1e-4,
            temperature_struct: float = 0.2,
            lambda_lap: float = 0.1,
            lambda_align: float = 0.3,
            device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = encoder.to(self.device)

        self.temperature_struct = temperature_struct
        self.lambda_lap = lambda_lap
        self.lambda_align = lambda_align

        structural_dim = encoder.structural_output_dim
        semantic_dim = encoder.semantic_output_dim
        fused_dim = encoder.final_output_dim

        if structural_dim != fused_dim:
            self.alignment_proj_struct = nn.Sequential(
                nn.Linear(structural_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU()
            ).to(self.device)
        else:
            self.alignment_proj_struct = nn.Identity()

        if semantic_dim != fused_dim:
            self.alignment_proj_semantic = nn.Sequential(
                nn.Linear(semantic_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU()
            ).to(self.device)
        else:
            self.alignment_proj_semantic = nn.Identity()

        # Only optimize parameters that need training (including alignment projection layers)
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        alignment_params = list(self.alignment_proj_struct.parameters()) + list(
            self.alignment_proj_semantic.parameters())
        trainable_params = model_params + alignment_params
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        self.scheduler = None
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.amp.GradScaler("cpu")

    def train(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_types: torch.Tensor,
            pos_encoding: torch.Tensor,
            texts: List[str],
            ckpt_path: str,
            adj: torch.Tensor,
            pos_mask: torch.Tensor,
            pos_weight: torch.Tensor | None = None,
            edge_weights: torch.Tensor | None = None,
            epochs: int = 100,

            use_lr_scheduler: bool = True,
    ):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_types = edge_types.to(self.device)
        pos_encoding = pos_encoding.to(self.device)
        adj = adj.to(self.device)
        pos_mask = pos_mask.to(self.device)
        pos_weight = pos_weight.to(self.device) if pos_weight is not None else None
        edge_weights = (
            edge_weights.to(self.device) if edge_weights is not None and edge_weights.numel() > 0 else None
        )

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # Initialize learning rate scheduler
        if use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=self.optimizer.param_groups[0]['lr'] * 0.01
            )

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            amp_device = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=amp_device, enabled=True):
                z_struct, z_text, z_fused = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_types=edge_types,
                    pos_encoding=pos_encoding,
                    texts=texts,
                    edge_weights=edge_weights,
                )

                loss_struct = multi_positive_infonce(
                    z_fused,
                    pos_mask=pos_mask,
                    pos_weight=pos_weight,
                    temperature=self.temperature_struct,
                )

                loss_lap = laplacian_smoothing(z_fused, adj) if self.lambda_lap > 0 else 0.0

                # Alignment loss: align fused representation to structural and semantic representations
                # Key fix: no detach(), because frozen parameters != no gradient output
                # Frozen parameters mean parameters don't update, but output can still have gradients so alignment loss can propagate to fusion module
                # Use projection layer instead of truncation to avoid information loss
                loss_align_struct = fused_alignment_loss(
                    z_fused, z_struct,
                    proj_layer=self.alignment_proj_struct
                )
                loss_align_text = fused_alignment_loss(
                    z_fused, z_text,
                    proj_layer=self.alignment_proj_semantic
                )
                loss_align = (loss_align_struct + loss_align_text) / 2.0

                loss = loss_struct + self.lambda_lap * loss_lap + self.lambda_align * loss_align

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Learning rate scheduling
            if use_lr_scheduler and hasattr(self, 'scheduler'):
                self.scheduler.step()

            with torch.no_grad():
                cur_loss = float(loss.detach().cpu().item())
                cur_loss_struct = float(loss_struct.detach().cpu().item())
                cur_loss_lap = float(loss_lap.detach().cpu().item()) if self.lambda_lap > 0 else 0.0
                cur_loss_align = float(loss_align.detach().cpu().item())
                cur_loss_align_struct = float(loss_align_struct.detach().cpu().item())
                cur_loss_align_text = float(loss_align_text.detach().cpu().item())
                current_lr = self.optimizer.param_groups[0]['lr']

                if epoch % 20 == 0:
                    print(
                        f"[Epoch {epoch:03d}] "
                        f"loss={cur_loss:.6f} "
                        f"(struct={cur_loss_struct:.6f}"
                        f"{f', lap={cur_loss_lap:.6f}' if self.lambda_lap > 0 else ''}"
                        f", align={cur_loss_align:.6f} [struct={cur_loss_align_struct:.6f}, text={cur_loss_align_text:.6f}])"
                        f" | lr={current_lr:.2e}"
                    )
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            torch.save(best_state, ckpt_path)
            print(f"✓ Best full encoder model saved: {ckpt_path}  | best_loss={best_loss:.6f}")
        else:
            print("! Model not saved (no better state found)")


def main(config: MicroWeaverConfig):
    classes = load_data(config.data_path)

    (
        builder,
        x,
        edge_index,
        edge_types,
        pos_encoding,
        texts,
        adj,
        pos_mask,
        pos_weight,
        edge_weights,
    ) = build_graph(classes, config.partition_config)

    encoder_cfg = get_config_by_graph_size(len(classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CodeGraphEncoder(
        structural_hidden_dim=encoder_cfg.structural.hidden_dim,
        structural_output_dim=encoder_cfg.structural.output_dim,
        semantic_output_dim=encoder_cfg.semantic.output_dim,
        final_output_dim=encoder_cfg.fusion.output_dim,
        num_edge_types=len(builder.edge_type_to_idx) if len(
            builder.edge_type_to_idx) > 0 else encoder_cfg.structural.num_edge_types,
        num_structural_layers=encoder_cfg.structural.num_layers,
        num_heads=encoder_cfg.structural.num_heads,
        dropout=encoder_cfg.structural.dropout,
        code_encoder_model=encoder_cfg.semantic.model_name,
        freeze_code_encoder=encoder_cfg.semantic.freeze_encoder,
        structural_only=False,
    ).to(device)

    struct_ckpt = config.structural_model_path
    if os.path.exists(struct_ckpt):
        try:
            print(f"\n[FullEncoder] Structural pre-trained model detected: {struct_ckpt}")
            model.structural_encoder.load_pretrained(struct_ckpt, device=device)
            print("✓ Structural encoder initialized with structural_best.pt")
        except Exception as e:
            print(f"! Failed to load structural pre-trained model: {e}, will use randomly initialized structural encoder")
    else:
        print(f"\n[FullEncoder] Structural pre-trained model not found ({struct_ckpt}), will use randomly initialized structural encoder")

    # 4.2 Freeze structural encoder and semantic encoder, only train fusion module
    for p in model.structural_encoder.parameters():
        p.requires_grad = False

    # Freeze semantic encoder
    for p in model.semantic_encoder.parameters():
        p.requires_grad = False

    # Print trainable parameter information
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[FullEncoder] Parameter statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,} (fusion module only)")
    print(f"  - Frozen parameters: {total_params - trainable_params:,} (structural + semantic encoders)")

    trainer = FullEncoderTrainer(
        encoder=model,
        lr=1e-4,
        weight_decay=1e-4,
        temperature_struct=0.2,
        lambda_lap=0,
        lambda_align=0.3,
        device=device,
    )

    trainer.train(
        x=x,
        edge_index=edge_index,
        edge_types=edge_types,
        pos_encoding=pos_encoding,
        texts=texts,
        adj=adj,
        pos_mask=pos_mask,
        pos_weight=pos_weight,
        edge_weights=edge_weights,
        epochs=180,
        ckpt_path=config.full_model_path,
    )

    # 5. Post-training debugging: detailed node pair similarity statistical analysis
    model.eval()
    with torch.no_grad():
        z_struct, z_text, z_fused = model(
            x=x.to(device),
            edge_index=edge_index.to(device),
            edge_types=edge_types.to(device),
            pos_encoding=pos_encoding.to(device),
            texts=texts,
            edge_weights=edge_weights.to(device) if edge_weights is not None and edge_weights.numel() > 0 else None,
        )

        # Normalize and compute similarity matrix
        z_fused_norm = F.normalize(z_fused, p=2, dim=-1)
        z_text_norm = F.normalize(z_text, p=2, dim=-1)
        z_struct_norm = F.normalize(z_struct, p=2, dim=-1)

        sim_fused = z_fused_norm @ z_fused_norm.t()  # [N, N] fused vector similarity
        sim_text = z_text_norm @ z_text_norm.t()  # [N, N] semantic vector similarity
        sim_struct = z_struct_norm @ z_struct_norm.t()  # [N, N] structural vector similarity

        N = sim_fused.size(0)

        # ========== 2. Statistics for directly connected, two-hop connected, and other node pairs ==========
        print("\n" + "=" * 80)
        print("[Debug 1] Fused vector similarity statistics grouped by graph structure distance:")
        print("=" * 80)

        # Build adjacency matrix (undirected graph)
        adj_matrix = torch.zeros(N, N, dtype=torch.bool, device=device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            adj_matrix[src, dst] = True
            adj_matrix[dst, src] = True

        # Compute two-hop adjacency matrix (via matrix multiplication)
        adj_2hop = (adj_matrix.float() @ adj_matrix.float()) > 0
        adj_2hop = adj_2hop & (~adj_matrix)  # Exclude one-hop edges
        adj_2hop.fill_diagonal_(False)  # Exclude self-loops

        # Categorize node pairs
        pairs_1hop = []  # Directly connected
        pairs_2hop = []  # Two-hop connected
        pairs_other = []  # Others (three-hop or more, or disconnected)

        for i in range(N):
            for j in range(i + 1, N):
                score = float(sim_fused[i, j].detach().cpu().item())
                if adj_matrix[i, j]:
                    pairs_1hop.append(score)
                elif adj_2hop[i, j]:
                    pairs_2hop.append(score)
                else:
                    pairs_other.append(score)

        def compute_stats(scores):
            if len(scores) == 0:
                return 0.0, 0.0, 0
            mean_val = sum(scores) / len(scores)
            variance = sum((x - mean_val) ** 2 for x in scores) / len(scores)
            return mean_val, variance, len(scores)

        mean_1hop, var_1hop, count_1hop = compute_stats(pairs_1hop)
        mean_2hop, var_2hop, count_2hop = compute_stats(pairs_2hop)
        mean_other, var_other, count_other = compute_stats(pairs_other)

        print(f"  Directly connected (1-hop):")
        print(f"    Node pairs: {count_1hop}")
        print(f"    Mean similarity: {mean_1hop:.6f}")
        print(f"    Variance: {var_1hop:.6f}")
        print(f"    Std dev: {var_1hop ** 0.5:.6f}")

        print(f"\n  Two-hop connected:")
        print(f"    Node pairs: {count_2hop}")
        print(f"    Mean similarity: {mean_2hop:.6f}")
        print(f"    Variance: {var_2hop:.6f}")
        print(f"    Std dev: {var_2hop ** 0.5:.6f}")

        print(f"\n  Others (3+ hops or disconnected):")
        print(f"    Node pairs: {count_other}")
        print(f"    Mean similarity: {mean_other:.6f}")
        print(f"    Variance: {var_other:.6f}")
        print(f"    Std dev: {var_other ** 0.5:.6f}")

        # ========== 3. Fused vector similarity statistics grouped by semantic similarity ==========
        print("\n" + "=" * 80)
        print("[Debug 2] Fused vector similarity statistics grouped by semantic similarity:")
        print("=" * 80)

        # Group by semantic similarity
        pairs_sem_high = []  # Semantic similarity >= 0.7
        pairs_sem_med_high = []  # 0.5 <= Semantic similarity < 0.7
        pairs_sem_med_low = []  # 0.3 <= Semantic similarity < 0.5
        pairs_sem_low = []  # Semantic similarity < 0.3

        for i in range(N):
            for j in range(i + 1, N):
                sem_score = float(sim_text[i, j].detach().cpu().item())
                fused_score = float(sim_fused[i, j].detach().cpu().item())

                if sem_score >= 0.7:
                    pairs_sem_high.append(fused_score)
                elif sem_score >= 0.5:
                    pairs_sem_med_high.append(fused_score)
                elif sem_score >= 0.3:
                    pairs_sem_med_low.append(fused_score)
                else:
                    pairs_sem_low.append(fused_score)

        mean_high, var_high, count_high = compute_stats(pairs_sem_high)
        mean_med_high, var_med_high, count_med_high = compute_stats(pairs_sem_med_high)
        mean_med_low, var_med_low, count_med_low = compute_stats(pairs_sem_med_low)
        mean_low, var_low, count_low = compute_stats(pairs_sem_low)

        print(f"  Semantic similarity >= 0.7:")
        print(f"    Node pairs: {count_high}")
        print(f"    Fused vector mean similarity: {mean_high:.6f}")
        print(f"    Variance: {var_high:.6f}")
        print(f"    Std dev: {var_high ** 0.5:.6f}")

        print(f"\n  Semantic similarity [0.5, 0.7):")
        print(f"    Node pairs: {count_med_high}")
        print(f"    Fused vector mean similarity: {mean_med_high:.6f}")
        print(f"    Variance: {var_med_high:.6f}")
        print(f"    Std dev: {var_med_high ** 0.5:.6f}")

        print(f"\n  Semantic similarity [0.3, 0.5):")
        print(f"    Node pairs: {count_med_low}")
        print(f"    Fused vector mean similarity: {mean_med_low:.6f}")
        print(f"    Variance: {var_med_low:.6f}")
        print(f"    Std dev: {var_med_low ** 0.5:.6f}")

        print(f"\n  Semantic similarity < 0.3:")
        print(f"    Node pairs: {count_low}")
        print(f"    Fused vector mean similarity: {mean_low:.6f}")
        print(f"    Variance: {var_low:.6f}")
        print(f"    Std dev: {var_low ** 0.5:.6f}")

        print("\n" + "=" * 80)

if __name__ == "__main__":
    main(MicroWeaverConfig())