"""Data loading + AAMI 5-class label mapping for the MIT-BIH CSV exports."""
from pathlib import Path
import numpy as np
import pandas as pd


# Map raw annotation strings -> AAMI super-class
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S", "SVEB": "S",
    "V": "V", "E": "V", "VEB": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


FEATURE_COLS = [
    # Lead 0
    "0_pre-RR", "0_post-RR", "0_pPeak", "0_tPeak", "0_rPeak",
    "0_sPeak", "0_qPeak", "0_qrs_interval", "0_pq_interval",
    "0_qt_interval", "0_st_interval",
    "0_qrs_morph0", "0_qrs_morph1", "0_qrs_morph2", "0_qrs_morph3", "0_qrs_morph4",
    # Lead 1
    "1_pre-RR", "1_post-RR", "1_pPeak", "1_tPeak", "1_rPeak",
    "1_sPeak", "1_qPeak", "1_qrs_interval", "1_pq_interval",
    "1_qt_interval", "1_st_interval",
    "1_qrs_morph0", "1_qrs_morph1", "1_qrs_morph2", "1_qrs_morph3", "1_qrs_morph4",
]


def map_to_aami(label):
    return AAMI_MAP.get(str(label), "Q")


def load_dataset(csv_path):
    """Returns X (features), y (AAMI label string), groups (patient/record id)."""
    df = pd.read_csv(csv_path)
    df["aami"] = df["type"].map(map_to_aami)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["aami"].to_numpy()
    groups = df["record"].to_numpy()
    return X, y, groups, df


def class_distribution(y):
    counts = pd.Series(y).value_counts().reindex(AAMI_CLASSES, fill_value=0)
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "percent": pct})


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    X, y, g, _ = load_dataset(here / "MIT-BIH Arrhythmia Database.csv")
    print("X:", X.shape, "y unique:", np.unique(y, return_counts=True))
    print("Patients:", len(np.unique(g)))
