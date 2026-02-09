import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class GroundTruth:
    date: str
    centroid: Tuple[float, float]
    direction: float
    risk_level: str
    affected_areas: List[str] = field(default_factory=list)
    speed_kmh: float = 0.0


@dataclass
class Prediction:
    date: str
    risk_score: float
    risk_level: str
    predicted_direction: float
    affected_areas: List[str] = field(default_factory=list)


RISK_LABELS = ("low", "medium", "high")


def _f1_single_class(
    y_true: List[str], y_pred: List[str], cls: str,
) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}


def macro_f1(
    y_true: List[str],
    y_pred: List[str],
    labels: Tuple[str, ...] = RISK_LABELS,
) -> Dict:
    per_class = {cls: _f1_single_class(y_true, y_pred, cls) for cls in labels}
    active = [per_class[c]["f1"] for c in labels if per_class[c]["support"] > 0]
    return {
        "macro_f1": float(np.mean(active)) if active else 0.0,
        "per_class": per_class,
    }


def angular_error(pred_deg: float, true_deg: float) -> float:
    diff = abs(pred_deg - true_deg) % 360
    return min(diff, 360 - diff)


def direction_accuracy(
    pred_dirs: List[float],
    true_dirs: List[float],
    tolerance: float = 45.0,
) -> Dict:
    errors = [angular_error(p, t) for p, t in zip(pred_dirs, true_dirs)]
    correct = [e <= tolerance for e in errors]
    return {
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "mean_error_deg": float(np.mean(errors)) if errors else 0.0,
        "per_window": [
            {"error_deg": round(e, 1), "correct": c}
            for e, c in zip(errors, correct)
        ],
    }


KNOWN_AREAS = {
    "atlas peak", "calistoga", "santa rosa", "coffey park",
    "fountaingrove", "mark west springs", "napa", "sebastopol",
    "sonoma", "wikiup", "larkfield", "kenwood", "highway 101",
    "bennett valley", "sonoma county", "containment zone",
}


def _normalize_area(name: str) -> str:
    low = name.lower().strip()
    for known in KNOWN_AREAS:
        if known in low or low in known:
            return known
    return low


def area_iou(pred_areas: List[str], true_areas: List[str]) -> float:
    ps = {_normalize_area(a) for a in pred_areas}
    ts = {_normalize_area(a) for a in true_areas}
    if not ps and not ts:
        return 1.0
    if not ps or not ts:
        return 0.0
    return len(ps & ts) / len(ps | ts)


def mean_area_iou(
    pred_list: List[List[str]], true_list: List[List[str]],
) -> Dict:
    ious = [area_iou(p, t) for p, t in zip(pred_list, true_list)]
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "per_window": [round(v, 4) for v in ious],
    }


def evaluate(
    predictions: List[Prediction],
    ground_truth: List[GroundTruth] = None,
) -> Dict:
    gt_map = {g.date: g for g in ground_truth}
    aligned = [(p, gt_map[p.date]) for p in predictions if p.date in gt_map]
    if not aligned:
        return {"error": "no matching dates between predictions and ground truth"}
    preds, gts = zip(*aligned)
    dates = [p.date for p in preds]
    f1_res = macro_f1(
        [g.risk_level for g in gts],
        [p.risk_level for p in preds],
    )
    dir_res = direction_accuracy(
        [p.predicted_direction for p in preds],
        [g.direction for g in gts],
    )
    iou_res = mean_area_iou(
        [p.affected_areas for p in preds],
        [g.affected_areas for g in gts],
    )
    return {
        "n_windows": len(aligned),
        "dates": dates,
        "macro_f1": f1_res,
        "direction_accuracy": dir_res,
        "area_iou": iou_res,
    }


def print_report(results: Dict):
    print()
    print("=" * 65)
    print("  CGRIP EVALUATION REPORT")
    print("=" * 65)
    print(f"  Windows evaluated: {results['n_windows']}")
    print(f"  Dates: {results['dates']}")
    f1 = results["macro_f1"]
    print(f"\n  {'Macro-F1':─<40s} {f1['macro_f1']:.4f}")
    for cls in RISK_LABELS:
        c = f1["per_class"][cls]
        print(f"    {cls:>7s}:  P={c['precision']:.3f}  R={c['recall']:.3f}  "
              f"F1={c['f1']:.3f}  (n={c['support']})")
    d = results["direction_accuracy"]
    print(f"\n  {'Direction Acc (±45°)':─<40s} {d['accuracy']:.4f}  "
          f"(mean err {d['mean_error_deg']:.1f}°)")
    for i, w in enumerate(d["per_window"]):
        tag = "OK" if w["correct"] else "MISS"
        date = results["dates"][i]
        print(f"    {date}:  err={w['error_deg']:5.1f}°  [{tag}]")
    iou = results["area_iou"]
    print(f"\n  {'Area IoU':─<40s} {iou['mean_iou']:.4f}")
    for i, v in enumerate(iou["per_window"]):
        date = results["dates"][i]
        print(f"    {date}:  IoU={v:.3f}")
    print()
    print("=" * 65)
