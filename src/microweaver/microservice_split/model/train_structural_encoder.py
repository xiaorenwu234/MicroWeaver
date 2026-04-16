"""
Train structural-only encoder to make directly connected nodes closer in embedding space.
- Training objective: Graph-based multi-positive InfoNCE + Laplacian smoothing (optional)
- Only uses structural information (no text, no cross-attention)
- After training, save weights to split/result/structural_best.pt
"""

import os
import torch
import torch.nn.functional as F
from typing import List
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_

from microweaver.microservice_split.config import MicroWeaverConfig, get_config_by_graph_size
from microweaver.microservice_split.model.code_graph_encoder import CodeGraphDataBuilder, CodeClass, StructuralEncoder
from microweaver.microservice_split.config import PartitionConfig
from microweaver.util.file_op import load_json


def load_data(data_path: str) -> List[CodeClass]:
    nodes = load_json(data_path)
    classes = [
        CodeClass(
            id=node['id'],
            name=node['name'],
            description=node['description'],
            methods=node['methods'],
            dependencies=node['dependencies'],
            edge_types=node['edge_types']
        )
        for node in nodes
    ]
    return classes


def build_graph(classes: List[CodeClass], partition_config: PartitionConfig):
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, _, edge_weights = builder.build_graph_data(
        edge_type_weights=partition_config.edge_type_weights
    )

    # Build undirected adjacency (positive sample mask) and corresponding weight matrix, ignoring self-loops
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

    return builder, x, edge_index, edge_types, pos_encoding, adj, pos_weight, edge_weights


class StructuralTrainer:
    def __init__(
            self,
            node_feature_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_edge_types: int,
            num_layers: int,
            num_heads: int,
            dropout: float,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            temperature: float = 0.2,
            lambda_lap: float = 0.1,  # Laplacian smoothing coefficient (>0 enables)
            warmup_epochs: int = 10,  # Warmup epochs
            max_grad_norm: float = 1.0,  # Maximum gradient norm for gradient clipping
            device: torch.device | None = None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StructuralEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_edge_types=num_edge_types,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        ).to(self.device)

        self.temperature = temperature
        self.lambda_lap = lambda_lap
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.amp.GradScaler('cpu')

        self.scheduler = None

    def multi_positive_infonce(self, z: torch.Tensor, pos_mask: torch.Tensor,
                               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        Multi-positive InfoNCE (supports weighted positive samples):
        For each node i, positive samples are its neighbor set P(i).
        loss_i = -log( sum_{j in P(i)} w_ij * exp(sim(i,j)/tau) / sum_{k != i} exp(sim(i,k)/tau) )
        Only nodes with non-empty P(i) are counted in loss.
        """
        z = F.normalize(z, p=2, dim=-1)
        sim = torch.matmul(z, z.t())  # [N,N], cosine similarity
        logits = sim / self.temperature

        # Exclude self
        N = z.size(0)
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        denom_mask = ~eye  # k != i

        # Only compute for rows with positive samples
        pos_mask = pos_mask.to(z.device).bool()
        weight = pos_weight.to(z.device) if pos_weight is not None else pos_mask.float()
        has_pos = (weight > 0).any(dim=1)
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # Numerical stability: subtract max per row
        row_max = logits.max(dim=1, keepdim=True).values
        logits = logits - row_max

        exp_logits = torch.exp(logits)
        denom = (exp_logits * denom_mask.float()).sum(dim=1)  # [N]
        numer = (exp_logits * weight).sum(dim=1)  # [N]

        # Only keep items with positive samples
        numer = numer[has_pos]
        denom = denom[has_pos] + 1e-12

        loss = -torch.log(numer / denom + 1e-12).mean()
        return loss

    def laplacian_smoothing(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothing term: sum_{(i,j) in E} ||z_i - z_j||^2
        """
        if adj.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        D = torch.diag(adj.sum(dim=1))
        L = D - adj
        loss = torch.trace(z.t() @ L @ z) / (adj.sum() + 1e-12)
        return loss

    def train(self,
              x: torch.Tensor,
              edge_index: torch.Tensor,
              edge_types: torch.Tensor,
              pos_encoding: torch.Tensor,
              pos_mask: torch.Tensor,
              pos_weight: torch.Tensor | None = None,
              edge_weights: torch.Tensor | None = None,
              epochs: int = 50,
              ckpt_path: str = "models/structural_best.pt"):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_types = edge_types.to(self.device)
        pos_encoding = pos_encoding.to(self.device)
        pos_mask = pos_mask.to(self.device)
        pos_weight = pos_weight.to(self.device) if pos_weight is not None else None
        edge_weights = edge_weights.to(self.device) if edge_weights is not None and edge_weights.numel() > 0 else None

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # Setup learning rate scheduler: warmup + cosine annealing
        warmup_epochs = min(self.warmup_epochs, epochs // 4)
        cosine_epochs = epochs - warmup_epochs

        if warmup_epochs > 0 and cosine_epochs > 0:
            # Warmup phase: linearly increase learning rate
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            # Cosine annealing phase: anneal from initial learning rate to minimum
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=self.base_lr * 0.01
            )
            # Combined scheduler: warmup first, then cosine annealing
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        elif warmup_epochs > 0:
            # If total epochs too few, only use warmup
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
        else:
            # If not using warmup, only use cosine annealing
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.base_lr * 0.01
            )

        best_loss = float('inf')
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                z = self.model(x, edge_index, edge_types, pos_encoding, edge_weights)
                loss_contrast = self.multi_positive_infonce(z, pos_mask, pos_weight=pos_weight)
                loss_lap = self.laplacian_smoothing(z, pos_mask) if self.lambda_lap > 0 else 0.0
                loss = loss_contrast + self.lambda_lap * loss_lap

            self.scaler.scale(loss).backward()

            # Gradient clipping: unscale first, then clip gradient norm
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            with torch.no_grad():
                cur_loss = float(loss.detach().cpu().item())
                current_lr = self.optimizer.param_groups[0]['lr']
                grad_norm_val = float(grad_norm.detach().cpu().item())
                if epoch % 100 == 0:
                    print(
                        f"[Epoch {epoch:03d}] loss={cur_loss:.6f} (contrast={float(loss_contrast):.6f}{f', lap={float(loss_lap):.6f}' if self.lambda_lap > 0 else ''}), lr={current_lr:.2e}, grad_norm={grad_norm_val:.4f}")
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            torch.save(best_state, ckpt_path)
            print(f"✓ Best model saved: {ckpt_path}  | best_loss={best_loss:.6f}")
        else:
            print("! Model not saved (no better state found)")


def main(config: MicroWeaverConfig):
    classes = load_data(config.data_path)

    enc_cfg = get_config_by_graph_size(len(classes)).structural

    builder, x, edge_index, edge_types, pos_encoding, pos_mask, pos_weight, edge_weights = build_graph(classes,
                                                                                                       config.partition_config)

    trainer = StructuralTrainer(
        node_feature_dim=enc_cfg.node_feature_dim,
        hidden_dim=enc_cfg.hidden_dim,
        output_dim=enc_cfg.output_dim,
        num_edge_types=len(builder.edge_type_to_idx) if len(builder.edge_type_to_idx) > 0 else enc_cfg.num_edge_types,
        num_layers=enc_cfg.num_layers,
        num_heads=enc_cfg.num_heads,
        dropout=enc_cfg.dropout,
        lr=1e-3,
        weight_decay=1e-4,
        temperature=0.2,
        lambda_lap=0.1,
        warmup_epochs=10,
        max_grad_norm=1.0,
    )

    trainer.train(
        x=x,
        edge_index=edge_index,
        edge_types=edge_types,
        pos_encoding=pos_encoding,
        pos_mask=pos_mask,
        pos_weight=pos_weight,
        edge_weights=edge_weights,
        epochs=400,
        ckpt_path=config.structural_model_path,
    )

if __name__ == "__main__":
    config = MicroWeaverConfig()
    main(config)