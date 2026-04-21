"""Data loading, AAMI mapping, DS1/DS2 split, multi-dataset support."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S", "SVEB": "S",
    "V": "V", "E": "V", "VEB": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


LEAD_FEATURES = [
    "pre-RR", "post-RR", "pPeak", "tPeak", "rPeak", "sPeak", "qPeak",
    "qrs_interval", "pq_interval", "qt_interval", "st_interval",
    "qrs_morph0", "qrs_morph1", "qrs_morph2", "qrs_morph3", "qrs_morph4",
]
FEATURE_COLS = [f"{lead}_{f}" for lead in (0, 1) for f in LEAD_FEATURES]

TEMPORAL_COLS = [c for c in FEATURE_COLS if any(t in c for t in
                  ("pre-RR", "post-RR", "qrs_interval", "pq_interval",
                   "qt_interval", "st_interval"))]
AMPLITUDE_COLS = [c for c in FEATURE_COLS if any(t in c for t in
                   ("pPeak", "tPeak", "rPeak", "sPeak", "qPeak"))]
MORPH_COLS = [c for c in FEATURE_COLS if "qrs_morph" in c]


# de Chazal et al. 2004 patient split — the de-facto AAMI benchmark
# (paced / excluded recordings 102, 104, 107, 217 are absent from the CSV)
DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
               201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
               213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def map_to_aami(label):
    return AAMI_MAP.get(str(label), "Q")


def load_dataset(csv_path, drop_q: bool = False,
                 bad_lines: str = "skip") -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns X, y (AAMI), groups (record id), df."""
    df = pd.read_csv(csv_path, on_bad_lines=bad_lines)
    df["aami"] = df["type"].map(map_to_aami)
    if drop_q:
        df = df[df["aami"] != "Q"].reset_index(drop=True)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["aami"].to_numpy()
    groups = df["record"].to_numpy()
    return X, y, groups, df


def ds1_ds2_split(X, y, groups, ds1=DS1_RECORDS, ds2=DS2_RECORDS):
    """Return (X_train, y_train, X_test, y_test) per the de Chazal split."""
    tr = np.isin(groups, ds1)
    te = np.isin(groups, ds2)
    return X[tr], y[tr], X[te], y[te], groups[tr], groups[te]


def class_distribution(y):
    counts = pd.Series(y).value_counts().reindex(AAMI_CLASSES, fill_value=0)
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "percent": pct})


def feature_group_mask(groups: Iterable[str]) -> list[int]:
    """Map a list of group names ('temporal','amplitude','morph','lead0','lead1') -> col indices."""
    idx = []
    for g in groups:
        if g == "temporal":
            idx += [FEATURE_COLS.index(c) for c in TEMPORAL_COLS]
        elif g == "amplitude":
            idx += [FEATURE_COLS.index(c) for c in AMPLITUDE_COLS]
        elif g == "morph":
            idx += [FEATURE_COLS.index(c) for c in MORPH_COLS]
        elif g in ("lead0", "lead1"):
            prefix = "0_" if g == "lead0" else "1_"
            idx += [i for i, c in enumerate(FEATURE_COLS) if c.startswith(prefix)]
    return sorted(set(idx))
