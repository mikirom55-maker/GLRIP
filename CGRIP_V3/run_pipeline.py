import os
import sys
import json
import time
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, List

from src.data.dataset import (
    build_datasets, load_satellite, load_social,
    CgripWindowDataset, CANDIDATE_LOCATIONS,
)
from src.models.encoders import (
    DualEncoder, extract_physical_features, tokenize_social_texts,
)
from src.models.contrastive import run_stage1_training
from src.graph.bipartite import (
    AlignmentModule, UncertaintyQuantifier, SparseMax,
    BipartiteGraphBuilder, BipartiteGraphNode, BipartiteGraphEdge,
)
from src.llm.reasoning import run_llm_reasoning
from src.utils.geo import haversine_distance, compute_bearing

CHECKPOINT_DIR = "outputs"
RESULTS_DIR = "results"
EMBEDDING_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_trained_components(
    checkpoint_path: str = "outputs/best_checkpoint.pt",
    device: str = "cuda",
):
    model = DualEncoder(physical_input_dim=10, embedding_dim=EMBEDDING_DIM,
                        dropout=0.1, finetune_last_n=2)
    alignment = AlignmentModule(embedding_dim=EMBEDDING_DIM, num_heads=4)
    uq = UncertaintyQuantifier(social_feature_dim=4, physical_feature_dim=3)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    alignment.load_state_dict(ckpt["alignment_module_state_dict"])
    uq.load_state_dict(ckpt["uncertainty_quantifier_state_dict"])

    model = model.to(device).eval()
    alignment = alignment.to(device).eval()
    uq = uq.to(device)

    print(f"  Checkpoint loaded: epoch={ckpt.get('epoch', '?')} val_loss={ckpt.get('val_loss', '?')}")
    return model, alignment, uq


@torch.no_grad()
def generate_embeddings(model, sat_df, soc_df, device="cuda"):
    if len(sat_df) > 0:
        phys_feat = extract_physical_features(sat_df).to(device)
        p_emb = model.encode_physical(phys_feat).cpu().numpy()
    else:
        p_emb = np.zeros((0, EMBEDDING_DIM))

    if len(soc_df) > 0:
        texts = soc_df["text"].tolist()
        s_embs = []
        for i in range(0, len(texts), 64):
            batch_texts = texts[i:i+64]
            ids, mask = tokenize_social_texts(batch_texts)
            ids, mask = ids.to(device), mask.to(device)
            emb = model.encode_social(ids, mask).cpu().numpy()
            s_embs.append(emb)
        s_emb = np.concatenate(s_embs, axis=0)
    else:
        s_emb = np.zeros((0, EMBEDDING_DIM))

    return p_emb, s_emb


def build_graph_nodes(df, embeddings, node_type, t_min):
    nodes = []
    is_physical = (node_type == "physical")
    for i, (_, row) in enumerate(df.iterrows()):
        if is_physical:
            attrs = {"area": row["Area"], "frp": row["FRP"],
                     "level": row["Level"], "reliability": row["Reliability"]}
            lat, lon = row["Lat"], row["Lon"]
            node_id = int(row["ID"])
        else:
            attrs = {"text": row["text"], "verified": row["verified"],
                     "engagement": row["engagement"]}
            lat, lon = row["lat"], row["lon"]
            node_id = int(row["id"]) + 100000

        nodes.append(BipartiteGraphNode(
            id=node_id, node_type=node_type,
            embedding=embeddings[i],
            lat=lat, lon=lon,
            timestamp=(row["timestamp"] - t_min).total_seconds() / 3600,
            attributes=attrs,
        ))
    return nodes


def compute_uncertainties(uq, alignment, physical_nodes, social_nodes, p_emb, s_emb, device="cuda"):
    phys_quality = torch.tensor([
        [n.attributes["level"] / 5.0, n.attributes["reliability"] / 5.0, n.attributes["area"] / 10.0]
        for n in physical_nodes
    ], dtype=torch.float32).to(device)
    with torch.no_grad():
        pu = uq.compute_data_uncertainty_physical(phys_quality)
    data_unc_p = {n.id: float(pu[i]) for i, n in enumerate(physical_nodes)}

    soc_feat = torch.tensor([
        [float(n.attributes.get("verified", False)),
         min(n.attributes.get("engagement", 0), 5000) / 5000.0,
         n.lat / 90.0, n.lon / 180.0]
        for n in social_nodes
    ], dtype=torch.float32).to(device)
    with torch.no_grad():
        su = uq.compute_data_uncertainty_social(soc_feat)
    data_unc_s = {n.id: float(su[i]) for i, n in enumerate(social_nodes)}

    model_unc = uq.compute_model_uncertainty(
        torch.tensor(p_emb, dtype=torch.float32).to(device),
        torch.tensor(s_emb, dtype=torch.float32).to(device),
        alignment_module=alignment,
        num_passes=10,
    ).cpu().numpy()

    return data_unc_p, data_unc_s, model_unc


def compute_dynamics(sat_curr, sat_prev, time_delta_hours=1.0):
    def centroid(df, lat_col="Lat", lon_col="Lon"):
        if len(df) == 0:
            return (0, 0)
        return (df[lat_col].mean(), df[lon_col].mean())

    c_curr = centroid(sat_curr)
    c_prev = centroid(sat_prev) if len(sat_prev) > 0 else c_curr

    dist = haversine_distance(c_prev[0], c_prev[1], c_curr[0], c_curr[1])
    speed = dist / time_delta_hours if time_delta_hours > 0 else 0
    direction = compute_bearing(c_prev[0], c_prev[1], c_curr[0], c_curr[1])

    prev_count = len(sat_prev)
    curr_count = len(sat_curr)
    growth = (curr_count / prev_count - 1) if prev_count > 0 else 0

    return {
        "centroid_lat": c_curr[0],
        "centroid_lon": c_curr[1],
        "speed_kmh": speed,
        "direction": direction,
        "growth_rate": growth,
        "fire_count_curr": curr_count,
        "fire_count_prev": prev_count,
    }


def evaluate_window(
    window_data: dict,
    model, alignment, uq, sparsemax,
    sat_df, soc_df, device="cuda",
):
    w = window_data["window"]
    t_min = sat_df["timestamp"].min()

    sat_hist = window_data["sat_hist"]
    soc_hist = window_data["soc_hist"]
    label = window_data["label"]

    timings = {}

    t0 = time.time()
    p_emb, s_emb = generate_embeddings(model, sat_hist, soc_hist, device)
    timings["embedding_s"] = time.time() - t0

    if len(sat_hist) == 0 or len(soc_hist) == 0:
        return None, timings

    t0 = time.time()
    p_nodes = build_graph_nodes(sat_hist, p_emb, "physical", t_min)
    s_nodes = build_graph_nodes(soc_hist, s_emb, "social", t_min)

    data_unc_p, data_unc_s, model_unc = compute_uncertainties(
        uq, alignment, p_nodes, s_nodes, p_emb, s_emb, device
    )

    builder = BipartiteGraphBuilder(
        spatial_radius_km=50, temporal_window_hours=12,
        alignment_module=alignment, sparsemax=sparsemax,
        min_edges_per_node=3,
    )
    nodes, edges, stats = builder.build_graph(
        p_nodes, s_nodes, data_unc_p, data_unc_s, model_unc
    )
    timings["graph_s"] = time.time() - t0

    prev_start = w.start - timedelta(hours=1)
    sat_prev = sat_df[
        (sat_df["timestamp"] >= prev_start) & (sat_df["timestamp"] < w.start)
    ]
    dynamics = compute_dynamics(sat_hist, sat_prev)

    social_texts = soc_hist["text"].tolist()

    t0 = time.time()
    prediction = run_llm_reasoning(
        stats, edges, social_texts, dynamics, str(w.end)
    )
    timings["llm_s"] = time.time() - t0

    return {
        "prediction": prediction,
        "label": label,
        "stats": stats,
        "window": w,
    }, timings


def compute_metrics(results: List[dict]) -> dict:
    from sklearn.metrics import f1_score

    y_true_risk = []
    y_pred_risk = []
    direction_correct = 0
    direction_total = 0
    area_ious = []

    risk_map = {"low": 0, "medium": 1, "high": 2}

    for r in results:
        if r is None:
            continue
        gt = r["label"]
        pred = r["prediction"]

        gt_risk = risk_map.get(gt["risk_level"], 1)
        pred_risk = risk_map.get(pred.risk_level, 1)
        y_true_risk.append(gt_risk)
        y_pred_risk.append(pred_risk)

        gt_dir = gt["direction"]
        pred_dir = pred.predicted_direction
        diff = abs(gt_dir - pred_dir)
        diff = min(diff, 360 - diff)
        direction_total += 1
        if diff <= 45:
            direction_correct += 1

        gt_areas = _nearest_locations(gt.get("next_centroid", (0, 0)))
        pred_areas = set(pred.affected_areas)
        if len(gt_areas | pred_areas) > 0:
            iou = len(gt_areas & pred_areas) / len(gt_areas | pred_areas)
            area_ious.append(iou)

    macro_f1 = f1_score(y_true_risk, y_pred_risk, average="macro", zero_division=0)
    dir_acc = direction_correct / max(direction_total, 1)
    avg_iou = np.mean(area_ious) if area_ious else 0.0

    return {
        "macro_f1": macro_f1,
        "direction_accuracy": dir_acc,
        "area_iou": avg_iou,
        "n_samples": len(results),
    }


_LOCATION_COORDS = {
    "Batemans Bay": (-35.71, 150.18),
    "Nowra": (-34.88, 150.60),
    "Shoalhaven": (-34.90, 150.75),
    "Mogo": (-35.79, 150.08),
    "Lake Conjola": (-35.27, 150.50),
    "Mallacoota": (-37.56, 149.76),
    "East Gippsland": (-37.40, 148.80),
}

def _nearest_locations(centroid, top_k=2):
    if centroid is None or np.isnan(centroid[0]):
        return set()
    dists = {}
    for name, (lat, lon) in _LOCATION_COORDS.items():
        dists[name] = haversine_distance(centroid[0], centroid[1], lat, lon)
    sorted_locs = sorted(dists.items(), key=lambda x: x[1])
    return set(name for name, _ in sorted_locs[:top_k])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 70)
    print("  CGRIP Pipeline")
    print("=" * 70)

    print("\n[0] Loading data ...")
    train_ds, val_ds, test_ds, sat_df, soc_df = build_datasets()

    if not args.eval:
        print("\n" + "=" * 70)
        print("  Stage 1: Contrastive Learning")
        print("=" * 70)
        t0 = time.time()

        model, alignment, uq = run_stage1_training(
            sat_df, soc_df,
            train_windows=[d["window"] for d in train_ds.window_data],
            val_windows=[d["window"] for d in val_ds.window_data],
            embedding_dim=EMBEDDING_DIM,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=5,
            lr=args.lr,
            device=DEVICE,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        stage1_time = time.time() - t0
        print(f"  Stage 1 time: {stage1_time:.1f}s")
    else:
        stage1_time = 0

    print("\n[Load] Loading best checkpoint ...")
    model, alignment, uq = load_trained_components(
        os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt"), DEVICE
    )
    sparsemax = SparseMax(init_temperature=1.0, min_temperature=0.1)

    print("\n" + "=" * 70)
    print(f"  Stage 2+3: Test set evaluation ({len(test_ds)} windows)")
    print("=" * 70)

    results = []
    total_timings = {"embedding_s": 0, "graph_s": 0, "llm_s": 0}
    all_edge_counts = []

    for i in range(len(test_ds)):
        window_data = test_ds[i]
        w = window_data["window"]

        if args.no_llm:
            t_min = sat_df["timestamp"].min()
            sat_hist = window_data["sat_hist"]
            soc_hist = window_data["soc_hist"]
            if len(sat_hist) == 0 or len(soc_hist) == 0:
                continue
            p_emb, s_emb = generate_embeddings(model, sat_hist, soc_hist, DEVICE)
            p_nodes = build_graph_nodes(sat_hist, p_emb, "physical", t_min)
            s_nodes = build_graph_nodes(soc_hist, s_emb, "social", t_min)
            data_unc_p, data_unc_s, model_unc = compute_uncertainties(
                uq, alignment, p_nodes, s_nodes, p_emb, s_emb, DEVICE)
            builder = BipartiteGraphBuilder(
                spatial_radius_km=50, temporal_window_hours=12,
                alignment_module=alignment, sparsemax=sparsemax,
                min_edges_per_node=3)
            nodes, edges, stats = builder.build_graph(
                p_nodes, s_nodes, data_unc_p, data_unc_s, model_unc)
            all_edge_counts.append(stats["avg_edges_per_physical"])
            print(f"  [{i+1}/{len(test_ds)}] {w.end} | "
                  f"nodes={stats['num_physical_nodes']}+{stats['num_social_nodes']} "
                  f"edges={stats['num_edges']} avg_edges={stats['avg_edges_per_physical']:.1f} "
                  f"temp={stats['sparsemax_temperature']:.3f}")
            continue

        result, timings = evaluate_window(
            window_data, model, alignment, uq, sparsemax, sat_df, soc_df, DEVICE
        )

        if result is not None:
            results.append(result)
            for k in total_timings:
                total_timings[k] += timings.get(k, 0)

            stats = result["stats"]
            all_edge_counts.append(stats["avg_edges_per_physical"])
            pred = result["prediction"]
            gt = result["label"]
            print(f"  [{i+1}/{len(test_ds)}] {w.end} | "
                  f"GT: {gt['risk_level']:<6} dir={gt['direction']:3.0f} | "
                  f"Pred: {pred.risk_level:<6} dir={pred.predicted_direction:3.0f} "
                  f"areas={pred.affected_areas} | "
                  f"edges={stats['num_edges']} avg={stats['avg_edges_per_physical']:.1f}")

    if all_edge_counts:
        avg_all = np.mean(all_edge_counts)
        print(f"\n  Edge stats: avg={avg_all:.1f} min={np.min(all_edge_counts):.1f} max={np.max(all_edge_counts):.1f}")

    if args.no_llm:
        print("\n  --no-llm mode: skipped LLM and metrics")
        return

    if results:
        metrics = compute_metrics(results)
        print("\n" + "=" * 70)
        print("  EVALUATION RESULTS")
        print("=" * 70)
        print(f"  Macro-F1:           {metrics['macro_f1']:.4f}")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        print(f"  Area IoU:           {metrics['area_iou']:.4f}")
        print(f"  Samples:            {metrics['n_samples']}")
        print()
        print("  Timing:")
        print(f"    Stage 1 (train):     {stage1_time:.1f}s")
        print(f"    Stage 2 (embedding): {total_timings['embedding_s']:.1f}s total")
        print(f"    Stage 2 (graph):     {total_timings['graph_s']:.1f}s total")
        print(f"    Stage 3 (LLM):       {total_timings['llm_s']:.1f}s total")
        print("=" * 70)

        output = {
            "metrics": metrics,
            "timings": {
                "stage1_total_s": stage1_time,
                "stage2_embedding_total_s": total_timings["embedding_s"],
                "stage2_graph_total_s": total_timings["graph_s"],
                "stage3_llm_total_s": total_timings["llm_s"],
            },
            "per_window": [
                {
                    "timestamp": str(r["window"].end),
                    "gt_risk": r["label"]["risk_level"],
                    "gt_direction": r["label"]["direction"],
                    "pred_risk": r["prediction"].risk_level,
                    "pred_direction": r["prediction"].predicted_direction,
                    "pred_areas": r["prediction"].affected_areas,
                    "explanation": r["prediction"].explanation[:200],
                }
                for r in results
            ],
        }
        out_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved: {out_path}")
    else:
        print("  No valid results")


if __name__ == "__main__":
    main()
