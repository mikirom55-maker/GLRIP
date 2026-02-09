import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass



@dataclass
class TimeWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    label_start: pd.Timestamp
    label_end: pd.Timestamp


def load_satellite(path: str = "") -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", header=None,
                     names=["ID", "Year", "Month", "Day", "Time_UTC",
                            "Lat", "Lon", "Area", "Volcano", "Level",
                            "Reliability", "FRP", "QF", "HotID"])
    df["timestamp"] = pd.to_datetime(
        df["Year"].astype(str) + "-" +
        df["Month"].astype(str).str.zfill(2) + "-" +
        df["Day"].astype(str).str.zfill(2) + " " +
        df["Time_UTC"].astype(str).str.zfill(4).str[:2] + ":00"
    )
    return df


def load_social(path: str = "") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    mask = (
        (df["is_relevant"] == True) &
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX) &
        (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    )
    return df[mask].reset_index(drop=True)


def build_sliding_windows(
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    lookback_hours: int = 3,
    step_hours: int = 1,
    predict_hours: int = 1,
) -> List[TimeWindow]:
    windows = []
    t = t_start + timedelta(hours=lookback_hours)
    while t + timedelta(hours=predict_hours) <= t_end:
        windows.append(TimeWindow(
            start=t - timedelta(hours=lookback_hours),
            end=t,
            label_start=t,
            label_end=t + timedelta(hours=predict_hours),
        ))
        t += timedelta(hours=step_hours)
    return windows


def split_windows(
    windows: List[TimeWindow],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[List[TimeWindow], List[TimeWindow], List[TimeWindow]]:
    n = len(windows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return windows[:n_train], windows[n_train:n_train + n_val], windows[n_train + n_val:]


def compute_window_label(
    sat_curr: pd.DataFrame,
    sat_next: pd.DataFrame,
) -> Dict:
    def centroid(df):
        if len(df) == 0:
            return (np.nan, np.nan)
        return (df["Lat"].mean(), df["Lon"].mean())

    c_curr = centroid(sat_curr)
    c_next = centroid(sat_next)

    n_next = len(sat_next)
    if n_next == 0:
        risk = "low"
    elif n_next < 30:
        risk = "medium"
    else:
        risk = "high"

    direction = 0.0
    if not (np.isnan(c_curr[0]) or np.isnan(c_next[0])):
        dlat = c_next[0] - c_curr[0]
        dlon = c_next[1] - c_curr[1]
        direction = (np.degrees(np.arctan2(dlon, dlat)) + 360) % 360

    return {
        "risk_level": risk,
        "direction": direction,
        "centroid": c_curr,
        "next_centroid": c_next,
        "fire_count_curr": len(sat_curr),
        "fire_count_next": n_next,
        "frp_mean_curr": sat_curr["FRP"].mean() if len(sat_curr) > 0 else 0,
        "frp_mean_next": sat_next["FRP"].mean() if len(sat_next) > 0 else 0,
    }


class CgripWindowDataset(Dataset):
    def __init__(
        self,
        windows: List[TimeWindow],
        satellite_df: pd.DataFrame,
        social_df: pd.DataFrame,
    ):
        self.windows = windows
        self.sat = satellite_df
        self.soc = social_df
        self.window_data: List[Dict] = []
        for w in windows:
            sat_hist = self.sat[
                (self.sat["timestamp"] >= w.start) &
                (self.sat["timestamp"] < w.end)
            ]
            soc_hist = self.soc[
                (self.soc["timestamp"] >= w.start) &
                (self.soc["timestamp"] < w.end)
            ]
            sat_next = self.sat[
                (self.sat["timestamp"] >= w.label_start) &
                (self.sat["timestamp"] < w.label_end)
            ]
            label = compute_window_label(sat_hist, sat_next)
            self.window_data.append({
                "sat_hist": sat_hist.reset_index(drop=True),
                "soc_hist": soc_hist.reset_index(drop=True),
                "sat_next": sat_next.reset_index(drop=True),
                "label": label,
                "window": w,
            })

    def __len__(self):
        return len(self.window_data)

    def __getitem__(self, idx):
        return self.window_data[idx]


def build_datasets(
    sat_path: str = "data/processed/australia_satellite.csv",
    soc_path: str = "data/processed/australia_social.csv",
    lookback_hours: int = 3,
    step_hours: int = 1,
    predict_hours: int = 1,
) -> Tuple[CgripWindowDataset, CgripWindowDataset, CgripWindowDataset,
           pd.DataFrame, pd.DataFrame]:
    sat = load_satellite(sat_path)
    soc = load_social(soc_path)
    t_start = max(sat["timestamp"].min(), soc["timestamp"].min())
    t_end = min(sat["timestamp"].max(), soc["timestamp"].max())
    print(f"Data time range: {t_start} -> {t_end}")
    windows = build_sliding_windows(t_start, t_end, lookback_hours, step_hours, predict_hours)
    print(f"Total windows: {len(windows)}")
    train_w, val_w, test_w = split_windows(windows)
    print(f"Split: train={len(train_w)} val={len(val_w)} test={len(test_w)}")
    train_ds = CgripWindowDataset(train_w, sat, soc)
    val_ds = CgripWindowDataset(val_w, sat, soc)
    test_ds = CgripWindowDataset(test_w, sat, soc)
    return train_ds, val_ds, test_ds, sat, soc
