"""Extended per-beat feature engineering on top of the published 32-d vector.

Adds three families derived purely from columns already in the CSV:
    1. **Local HRV** over a sliding window of pre-/post-RR values
       (approximated from neighbouring beats within the same record).
    2. **RR ratios / normalized RR** — post-RR / pre-RR, local mean ratio,
       and z-score vs. the surrounding 32-beat window.
    3. **Cross-lead** — absolute difference and ratio of matched features
       between lead 0 and lead 1.

Every feature here is engineered from the existing feature CSV — no raw
waveform needed.  Can be applied to every record independently so it works
cross-dataset without retraining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import FEATURE_COLS, LEAD_FEATURES


def _window_stats(series: pd.Series, window: int = 32) -> pd.DataFrame:
    """Centered rolling mean / std / median / IQR."""
    roll = series.rolling(window=window, min_periods=4, center=True)
    stats = pd.DataFrame({
        f"{series.name}_mean": roll.mean(),
        f"{series.name}_std":  roll.std().fillna(0.0),
        f"{series.name}_med":  roll.median(),
        f"{series.name}_q25":  roll.quantile(0.25),
        f"{series.name}_q75":  roll.quantile(0.75),
    })
    return stats


def hrv_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HRV/RR-derivative features per record.

    Assumes df has columns ``0_pre-RR`` / ``0_post-RR`` (lead 0) — the timing
    columns are duplicated across leads so using lead 0 is sufficient.
    """
    out = pd.DataFrame(index=df.index)
    for prefix in ("0", "1"):
        pre  = df[f"{prefix}_pre-RR"].astype(float)
        post = df[f"{prefix}_post-RR"].astype(float)

        # Basic ratios
        out[f"{prefix}_rr_ratio"]     = post / pre.replace(0, np.nan)
        out[f"{prefix}_rr_diff"]      = post - pre
        out[f"{prefix}_rr_abs_diff"]  = out[f"{prefix}_rr_diff"].abs()

        # Window statistics on pre-RR (proxy for local NN-interval series)
        for g, grp in df.groupby("record", sort=False):
            w = _window_stats(grp[f"{prefix}_pre-RR"].astype(float), window=32)
            out.loc[grp.index, [c.replace("pre-RR", f"{prefix}_preRR_win")
                                  for c in w.columns]] = w.values

        # SDNN approx (rolling std), RMSSD approx (rolling std of successive diffs)
        local_std = pre.rolling(32, min_periods=4, center=True).std().fillna(0.0)
        successive_diff = pre.diff().abs()
        rmssd = np.sqrt((successive_diff ** 2)
                         .rolling(32, min_periods=4, center=True).mean()).fillna(0.0)
        out[f"{prefix}_sdnn_local"]  = local_std
        out[f"{prefix}_rmssd_local"] = rmssd

        # pNN50-style: fraction of |ΔRR| > 50 ms. Since the CSV's RR is in
        # seconds, threshold = 0.05.
        p_n50 = (successive_diff > 0.05).rolling(32, min_periods=4,
                                                   center=True).mean().fillna(0.0)
        out[f"{prefix}_pNN50_local"] = p_n50

    # Cross-lead differences for matched features
    for f in LEAD_FEATURES:
        a = df[f"0_{f}"].astype(float)
        b = df[f"1_{f}"].astype(float)
        out[f"xlead_{f}_absdiff"] = (a - b).abs()
        out[f"xlead_{f}_ratio"]   = a / b.replace(0, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def build_extended_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Stack original FEATURE_COLS + HRV + cross-lead features.

    Returns (X_extended, all_column_names).
    """
    base = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    ext = hrv_features_from_df(df)
    X = np.hstack([base, ext.to_numpy(dtype=np.float32)])
    cols = list(FEATURE_COLS) + list(ext.columns)
    return X, cols
