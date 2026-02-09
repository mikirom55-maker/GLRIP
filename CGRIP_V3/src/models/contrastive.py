import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.models.encoders import (
    DualEncoder,
    extract_physical_features,
    tokenize_social_texts,
)
from src.graph.bipartite import AlignmentModule, UncertaintyQuantifier
from src.utils.geo import haversine_distance, spatiotemporal_distance


def build_contrastive_pairs(
    sat_df: pd.DataFrame,
    soc_df: pd.DataFrame,
    spatial_radius_km: float = 50.0,
    temporal_window_hours: float = 3.0,
    max_pairs_per_physical: int = 3,
) -> List[Tuple[pd.Series, pd.Series]]:
    pairs = []
    s_lats = soc_df["lat"].values
    s_lons = soc_df["lon"].values
    s_times = pd.to_datetime(soc_df["timestamp"])

    for _, p_row in sat_df.iterrows():
        p_lat, p_lon = p_row["Lat"], p_row["Lon"]
        p_time = p_row["timestamp"]

        dists = np.array([
            haversine_distance(p_lat, p_lon, sl, sn)
            for sl, sn in zip(s_lats, s_lons)
        ])
        time_diffs = np.abs((s_times - p_time).dt.total_seconds() / 3600).values

        mask = (dists <= spatial_radius_km) & (time_diffs <= temporal_window_hours)
        idxs = np.where(mask)[0]

        if len(idxs) == 0:
            continue

        st_dists = dists[idxs] + time_diffs[idxs] * 10
        order = np.argsort(st_dists)[:max_pairs_per_physical]

        for o in order:
            pairs.append((p_row, soc_df.iloc[idxs[o]]))

    return pairs


class SpatiotemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, p_emb, s_emb, alignment_module=None):
        N = p_emb.shape[0]
        if alignment_module is not None:
            scores = alignment_module(p_emb, s_emb)
            logits = scores * (alignment_module.embedding_dim ** 0.5) / self.temperature
        else:
            logits = torch.mm(p_emb, s_emb.t()) / self.temperature
        labels = torch.arange(N, device=p_emb.device)
        loss_p2s = F.cross_entropy(logits, labels)
        loss_s2p = F.cross_entropy(logits.t(), labels)
        return (loss_p2s + loss_s2p) / 2


class ContrastiveTrainer:
    def __init__(
        self,
        model: DualEncoder,
        alignment_module: AlignmentModule,
        uncertainty_quantifier: UncertaintyQuantifier,
        train_pairs: List,
        val_pairs: List,
        sat_df: pd.DataFrame,
        soc_df: pd.DataFrame,
        lr: float = 2e-5,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5,
        unc_loss_weight: float = 0.1,
        device: str = "cuda",
        checkpoint_dir: str = "outputs",
    ):
        self.device = device
        self.model = model.to(device)
        self.alignment = alignment_module.to(device)
        self.uq = uncertainty_quantifier.to(device)
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.unc_loss_weight = unc_loss_weight
        self.checkpoint_dir = checkpoint_dir
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.criterion = SpatiotemporalContrastiveLoss(temperature=0.07)
        all_params = (
            list(model.parameters()) +
            list(alignment_module.parameters()) +
            list(uncertainty_quantifier.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def _make_batch(self, pairs, indices):
        phys_feats = []
        texts = []
        verified_list = []

        for i in indices:
            p_row, s_row = pairs[i]
            ts = p_row["timestamp"]
            hour = ts.hour + ts.minute / 60.0
            doy = ts.timetuple().tm_yday
            phys_feats.append([
                p_row["Lat"] / 90.0,
                p_row["Lon"] / 180.0,
                p_row["Area"] / 10.0,
                p_row["FRP"] / 500.0,
                p_row["Level"] / 5.0,
                p_row["Reliability"] / 5.0,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * doy / 365),
                np.cos(2 * np.pi * doy / 365),
            ])
            texts.append(str(s_row["text"]))
            verified_list.append(float(s_row["verified"]))

        phys = torch.tensor(phys_feats, dtype=torch.float32).to(self.device)
        input_ids, attention_mask = tokenize_social_texts(texts)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        verified = torch.tensor(verified_list, dtype=torch.float32).to(self.device)
        return phys, input_ids, attention_mask, verified

    def _compute_unc_loss(self, p_emb, s_emb, phys, verified):
        scores_list = []
        for _ in range(5):
            scores = self.alignment(p_emb, s_emb)
            scores_list.append(torch.diagonal(scores))
        consistency_var = torch.stack(scores_list).var(dim=0).detach()
        std = consistency_var.std()
        target = torch.sigmoid(
            (consistency_var - consistency_var.mean()) / (std + 1e-8)
        )
        phys_quality = phys[:, [4, 5, 2]]
        soc_unc_feat = torch.stack([
            verified,
            torch.zeros_like(verified),
            phys[:, 0],
            phys[:, 1],
        ], dim=-1)
        pred_p = self.uq.compute_data_uncertainty_physical(phys_quality)
        pred_s = self.uq.compute_data_uncertainty_social(soc_unc_feat)
        pred_combined = (pred_p + pred_s) / 2
        return F.mse_loss(pred_combined, target)

    def _run_epoch(self, pairs, train=True):
        if train:
            self.model.train()
            self.alignment.train()
            self.uq.train()
        else:
            self.model.eval()
            self.alignment.eval()

        indices = np.arange(len(pairs))
        if train:
            np.random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if len(batch_idx) < 4:
                continue
            phys, input_ids, attn_mask, verified = self._make_batch(pairs, batch_idx)
            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                p_emb, s_emb = self.model(phys, input_ids, attn_mask)
                contrastive_loss = self.criterion(p_emb, s_emb, self.alignment)
                if train and self.unc_loss_weight > 0:
                    unc_loss = self._compute_unc_loss(p_emb, s_emb, phys, verified)
                    loss = contrastive_loss + self.unc_loss_weight * unc_loss
                else:
                    loss = contrastive_loss
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        print(f"Stage 1 training (device={self.device}, pairs: train={len(self.train_pairs)} val={len(self.val_pairs)})")
        print(f"  epochs={self.epochs} batch_size={self.batch_size} patience={self.patience}")
        print("-" * 70)

        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss = self._run_epoch(self.train_pairs, train=True)
            val_loss = self._run_epoch(self.val_pairs, train=False)
            self.scheduler.step()
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} | "
                  f"lr={lr:.2e} | {elapsed:.1f}s")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                self._save_checkpoint(
                    os.path.join(self.checkpoint_dir, "best_checkpoint.pt"),
                    epoch, val_loss,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stop! best epoch={best_epoch} val_loss={best_val_loss:.4f}")
                    break

        print("-" * 70)
        print(f"Training done! best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}")
        return best_val_loss

    def _save_checkpoint(self, path, epoch, val_loss):
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "alignment_module_state_dict": self.alignment.state_dict(),
            "uncertainty_quantifier_state_dict": self.uq.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)


def run_stage1_training(
    sat_df: pd.DataFrame,
    soc_df: pd.DataFrame,
    train_windows,
    val_windows,
    embedding_dim: int = 256,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 5,
    lr: float = 2e-5,
    device: str = "cuda",
    checkpoint_dir: str = "outputs",
):
    print("[Stage 1] Building contrastive pairs ...")
    train_sat = pd.concat([
        sat_df[(sat_df["timestamp"] >= w.start) & (sat_df["timestamp"] < w.end)]
        for w in train_windows
    ]).drop_duplicates(subset=["ID"]).reset_index(drop=True)
    train_soc = pd.concat([
        soc_df[(soc_df["timestamp"] >= w.start) & (soc_df["timestamp"] < w.end)]
        for w in train_windows
    ]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    val_sat = pd.concat([
        sat_df[(sat_df["timestamp"] >= w.start) & (sat_df["timestamp"] < w.end)]
        for w in val_windows
    ]).drop_duplicates(subset=["ID"]).reset_index(drop=True)
    val_soc = pd.concat([
        soc_df[(soc_df["timestamp"] >= w.start) & (soc_df["timestamp"] < w.end)]
        for w in val_windows
    ]).drop_duplicates(subset=["id"]).reset_index(drop=True)

    print(f"  Train records: sat={len(train_sat)} soc={len(train_soc)}")
    print(f"  Val   records: sat={len(val_sat)} soc={len(val_soc)}")

    train_pairs = build_contrastive_pairs(train_sat, train_soc)
    val_pairs = build_contrastive_pairs(val_sat, val_soc)
    print(f"  Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    model = DualEncoder(
        physical_input_dim=10,
        embedding_dim=embedding_dim,
        dropout=0.1,
        finetune_last_n=2,
    )
    alignment = AlignmentModule(embedding_dim=embedding_dim, num_heads=4)
    uq = UncertaintyQuantifier(social_feature_dim=4, physical_feature_dim=3)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in alignment.parameters())
    n_params += sum(p.numel() for p in uq.parameters())
    print(f"  Trainable params: {n_params:,}")

    trainer = ContrastiveTrainer(
        model=model,
        alignment_module=alignment,
        uncertainty_quantifier=uq,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        sat_df=sat_df,
        soc_df=soc_df,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    trainer.train()
    return model, alignment, uq
