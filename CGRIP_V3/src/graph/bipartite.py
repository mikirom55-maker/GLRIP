import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.utils.geo import haversine_distance


@dataclass
class BipartiteGraphNode:
    id: int
    node_type: str
    embedding: np.ndarray
    lat: float
    lon: float
    timestamp: float
    attributes: dict


@dataclass
class BipartiteGraphEdge:
    source_id: int
    target_id: int
    weight: float
    data_uncertainty: float
    model_uncertainty: float


class SparseMax(nn.Module):
    def __init__(self, dim: int = -1, init_temperature: float = 1.0, min_temperature: float = 0.1):
        super().__init__()
        self.dim = dim
        self.min_temperature = min_temperature
        self._log_temp = nn.Parameter(torch.tensor(np.log(init_temperature)))

    @property
    def temperature(self):
        return torch.clamp(torch.exp(self._log_temp), min=self.min_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.temperature
        sorted_x, _ = torch.sort(x, dim=self.dim, descending=True)
        cumsum = torch.cumsum(sorted_x, dim=self.dim)
        k = torch.arange(1, x.size(self.dim) + 1, device=x.device).float()
        k = k.view([1] * (x.dim() - 1) + [-1])
        condition = 1 + k * sorted_x > cumsum
        k_max = condition.sum(dim=self.dim, keepdim=True)
        tau = (cumsum.gather(self.dim, k_max - 1) - 1) / k_max.float()
        return torch.clamp(x - tau, min=0)


class AlignmentModule(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.W = nn.Parameter(torch.eye(embedding_dim).unsqueeze(0).repeat(num_heads, 1, 1))
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, physical_emb: torch.Tensor, social_emb: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(physical_emb.shape[0], social_emb.shape[0],
                             device=physical_emb.device)
        for h in range(self.num_heads):
            transformed = physical_emb @ self.W[h]
            head_score = transformed @ social_emb.t() / (self.embedding_dim ** 0.5)
            head_score = self.attn_dropout(head_score)
            scores = scores + head_score
        return scores / self.num_heads

    def compute_single_score(self, p_emb: np.ndarray, s_emb: np.ndarray) -> float:
        W_np = self.W.detach().cpu().numpy()
        d = len(p_emb)
        score = 0.0
        for h in range(self.num_heads):
            score += np.dot(p_emb, np.dot(W_np[h], s_emb)) / np.sqrt(d)
        return float(score / self.num_heads)


class UncertaintyQuantifier(nn.Module):
    def __init__(self, social_feature_dim: int = 4, physical_feature_dim: int = 3,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.social_proj = nn.Sequential(
            nn.Linear(social_feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.physical_proj = nn.Sequential(
            nn.Linear(physical_feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def compute_data_uncertainty_social(self, feat: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigmoid(self.social_proj(feat))).squeeze(-1)

    def compute_data_uncertainty_physical(self, feat: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigmoid(self.physical_proj(feat))).squeeze(-1)

    def compute_model_uncertainty(self, p_emb, s_emb, alignment_module, num_passes=10):
        was_training = alignment_module.training
        alignment_module.train()
        scores_list = []
        with torch.no_grad():
            for _ in range(num_passes):
                scores = alignment_module(p_emb, s_emb)
                scores_list.append(scores)
        if not was_training:
            alignment_module.eval()
        return torch.var(torch.stack(scores_list, dim=0), dim=0)


class BipartiteGraphBuilder:
    def __init__(
        self,
        spatial_radius_km: float = ,
        temporal_window_hours: float = ,
        min_edge_weight: float = 0.01,
        lambda_model: float = 1.0,
        alignment_module: Optional[AlignmentModule] = None,
        sparsemax: Optional[SparseMax] = None,
        min_edges_per_node: int = 3,
    ):
        self.spatial_radius = spatial_radius_km
        self.temporal_window = temporal_window_hours
        self.min_edge_weight = min_edge_weight
        self.lambda_model = lambda_model
        self.alignment_module = alignment_module
        self.sparsemax = sparsemax or SparseMax()
        self.min_edges_per_node = min_edges_per_node

    def find_candidates(self, p_node, social_nodes):
        candidates = []
        for s_node in social_nodes:
            dist = haversine_distance(p_node.lat, p_node.lon, s_node.lat, s_node.lon)
            tdiff = abs(p_node.timestamp - s_node.timestamp)
            if dist <= self.spatial_radius and tdiff <= self.temporal_window:
                candidates.append(s_node)
        return candidates

    def compute_alignment_score(self, p_emb, s_emb):
        if self.alignment_module is not None:
            return self.alignment_module.compute_single_score(p_emb, s_emb)
        d = len(p_emb)
        return float(np.dot(p_emb, s_emb) / np.sqrt(d))

    def compute_edge_weight(self, alignment, eps_p, eps_s, eps_model):
        w = alignment * (1 - eps_p) * (1 - eps_s) * np.exp(-self.lambda_model * eps_model)
        return max(0.0, w)

    def build_graph(
        self, physical_nodes, social_nodes,
        data_unc_p, data_unc_s, model_unc_matrix,
    ) -> Tuple[List[BipartiteGraphNode], List[BipartiteGraphEdge], dict]:
        all_nodes = physical_nodes + social_nodes
        edges = []
        edge_counts = []

        for p_idx, p_node in enumerate(physical_nodes):
            candidates = self.find_candidates(p_node, social_nodes)
            if not candidates:
                edge_counts.append(0)
                continue

            weights = []
            infos = []
            for s_node in candidates:
                s_idx = social_nodes.index(s_node)
                alignment = self.compute_alignment_score(p_node.embedding, s_node.embedding)
                eps_p = data_unc_p.get(p_node.id, 0.5)
                eps_s = data_unc_s.get(s_node.id, 0.5)
                eps_m = float(model_unc_matrix[p_idx, s_idx])
                w = self.compute_edge_weight(alignment, eps_p, eps_s, eps_m)
                weights.append(w)
                infos.append({"s_node": s_node, "eps_p": eps_p, "eps_s": eps_s, "eps_m": eps_m})

            if len(weights) > 0:
                wt = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)
                norm_w = self.sparsemax(wt).squeeze(0).detach().numpy()
                nonzero = np.count_nonzero(norm_w)
                if nonzero < self.min_edges_per_node and len(norm_w) >= self.min_edges_per_node:
                    top_k_idx = np.argsort(weights)[-self.min_edges_per_node:]
                    for ki in top_k_idx:
                        if norm_w[ki] == 0:
                            norm_w[ki] = self.min_edge_weight
                    total = norm_w.sum()
                    if total > 0:
                        norm_w = norm_w / total
            else:
                norm_w = np.array(weights)

            node_edge_count = 0
            for w, info in zip(norm_w, infos):
                if w >= self.min_edge_weight:
                    edges.append(BipartiteGraphEdge(
                        source_id=info["s_node"].id,
                        target_id=p_node.id,
                        weight=float(w),
                        data_uncertainty=(info["eps_p"] + info["eps_s"]) / 2,
                        model_uncertainty=info["eps_m"],
                    ))
                    node_edge_count += 1
            edge_counts.append(node_edge_count)

        stats = self._compute_stats(all_nodes, edges, edge_counts)
        return all_nodes, edges, stats

    def _compute_stats(self, nodes, edges, edge_counts):
        p_nodes = [n for n in nodes if n.node_type == "physical"]
        s_nodes = [n for n in nodes if n.node_type == "social"]
        np_, ns_ = len(p_nodes), len(s_nodes)
        ne = len(edges)
        max_e = np_ * ns_
        avg_edges = np.mean(edge_counts) if edge_counts else 0
        return {
            "num_physical_nodes": np_,
            "num_social_nodes": ns_,
            "num_edges": ne,
            "density": ne / max_e if max_e > 0 else 0,
            "avg_edges_per_physical": avg_edges,
            "min_edges": int(np.min(edge_counts)) if edge_counts else 0,
            "max_edges": int(np.max(edge_counts)) if edge_counts else 0,
            "connected_physical": len(set(e.target_id for e in edges)),
            "gcc_ratio": len(set(e.target_id for e in edges)) / np_ if np_ > 0 else 0,
            "sparsemax_temperature": float(self.sparsemax.temperature.item())
                                     if hasattr(self.sparsemax, "temperature") else 1.0,
        }
