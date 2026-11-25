"""
detectors.py - anomaly detectors for the M7 filler data and helpers to evaluate them.

All detectors work per operating state (column `state` from copa_data.build_wide):
    running  -> fill_weight, compair, power (+ rolling features)
    stopped  -> compair, power   (fill_weight is frozen at the last bottle while stopped)
    glitch / invalid rows are never scored (glitch = one-sample transition at order end that also
    occurs in the clean window, so it is not one of the anomalies COPA-DATA means).

D1  frequency ranking  (unsupervised): how often does this value occur in this state? Values that
                         occur in fewer than `tau` of the rows (default 1 in 10,000) are flagged.
D2  clean envelope     (semi-supervised): value or rolling feature outside the min-max range seen in
                         the clean window 04.11.2026 10:50-15:00 (Andreas Gruen, mail 13.11.2026), widened by
                         `margin_sd` clean standard deviations (default 1). The plain min-max of a 4 h window
                         is too tight for Gaussian-like features and produced false alarms out of sample.
D3  Isolation Forest   (unsupervised, multivariate): sklearn IsolationForest per state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from copa_data import CLEAN_END, CLEAN_START

RAW = {"running": ["fill_weight", "compair", "power"], "stopped": ["compair", "power"]}
ROLLING = {"fill_weight_std": ("fill_weight", "std"), "compair_mean": ("compair", "mean"),
           "power_mean": ("power", "mean")}          # computed for running rows only
UPPER_ONLY = {"fill_weight_std"}                      # a low spread is not an anomaly of interest
FEATURE_TYPE = {"compair": "A", "compair_mean": "A", "power": "B", "power_mean": "B",
                "fill_weight": "C", "fill_weight_std": "C"}
TYPE_NAMES = {"A": "Compressed-air drift", "B": "Power drift", "C": "Fill-weight disturbance"}


# --------------------------------------------------------------------------- features
def window_seconds(step_s: float) -> int:
    """Rolling window: 60 s, but at least 5 samples for coarse sampling."""
    return int(max(60, 5 * step_s))


def add_rolling(wide: pd.DataFrame, step_s: float, center: bool = False) -> pd.DataFrame:
    """Rolling mean/std inside contiguous running segments (never across a stop).
    center=False is causal (online); center=True uses future samples (batch only)."""
    w = wide.copy()
    seg = (w["state"] != w["state"].shift()).cumsum()
    run = w["state"] == "running"
    win_s = window_seconds(step_s)
    minp = max(3, int(round(0.5 * win_s / step_s)))
    for name, (col, how) in ROLLING.items():
        s = w[col].where(run)
        r = s.groupby(seg).rolling(f"{win_s}s", center=center, min_periods=minp)
        r = r.std() if how == "std" else r.mean()
        w[name] = r.reset_index(level=0, drop=True).reindex(w.index).where(run)
    return w


# --------------------------------------------------------------------------- D1 frequency ranking
def frequency_table(values: pd.Series, tau: float = 1e-4) -> pd.DataFrame:
    """The 'descending list': every observed value with its share of rows, most frequent first."""
    vc = values.dropna().round(1).value_counts()
    t = pd.DataFrame({"value": vc.index.to_numpy(), "rows": vc.to_numpy()})
    t["share_pct"] = 100 * t["rows"] / t["rows"].sum()
    t["cum_share_pct"] = t["share_pct"].cumsum()
    t["rank"] = np.arange(1, len(t) + 1)
    t["rare"] = t["share_pct"] < 100 * tau
    return t


def fit_frequency(wide: pd.DataFrame) -> dict:
    model = {}
    for state, sigs in RAW.items():
        sub = wide[wide["state"] == state]
        for sig in sigs:
            model[(state, sig)] = sub[sig].round(1).value_counts(normalize=True)
    return model


def frequency_shares(wide: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Share (0..1) of the row's value within its state; values never seen in training get 0."""
    out = pd.DataFrame(np.nan, index=wide.index, columns=sorted({s for _, s in model}))
    for (state, sig), vc in model.items():
        m = (wide["state"] == state) & wide[sig].notna()
        out.loc[m, sig] = wide.loc[m, sig].round(1).map(vc).fillna(0.0).to_numpy()
    return out


def frequency_flags(shares: pd.DataFrame, tau: float = 1e-4) -> pd.DataFrame:
    return shares.lt(tau) & shares.notna()


# --------------------------------------------------------------------------- D2 clean envelope
def fit_envelope(ref: pd.DataFrame, raw_from: pd.DataFrame | None = None, margin_sd: float = 1.0) -> dict:
    """(low, high, mean, std) per (state, feature): clean min-max widened by margin_sd * clean std.
    `raw_from` lets raw features use another reference (e.g. the full-resolution clean window)
    while rolling features come from `ref`."""
    env = {}
    raw_ref = ref if raw_from is None else raw_from

    def lim(x):
        sd = x.std()
        return (x.min() - margin_sd * sd, x.max() + margin_sd * sd, x.mean(), sd)

    for state, sigs in RAW.items():
        for f in sigs:
            env[(state, f)] = lim(raw_ref.loc[raw_ref["state"] == state, f].dropna())
    for f in ROLLING:
        if f in ref:
            x = ref.loc[ref["state"] == "running", f].dropna()
            if len(x):
                env[("running", f)] = lim(x)
    return env


def envelope_excess(wide: pd.DataFrame, env: dict) -> pd.DataFrame:
    """Distance outside the clean range in units of the clean standard deviation (0 = inside)."""
    cols = sorted({f for _, f in env})
    out = pd.DataFrame(0.0, index=wide.index, columns=cols)
    for (state, f), (lo, hi, mu, sd) in env.items():
        if f not in wide:
            continue
        m = (wide["state"] == state) & wide[f].notna()
        x = wide.loc[m, f]
        sd = sd if sd and sd > 0 else 1.0
        up = (x - hi).clip(lower=0) / sd
        down = 0 if f in UPPER_ONLY else (lo - x).clip(lower=0) / sd
        out.loc[m, f] = np.maximum(out.loc[m, f], up + down)
    return out


def envelope_flags(wide: pd.DataFrame, env: dict) -> pd.DataFrame:
    return envelope_excess(wide, env) > 0


# --------------------------------------------------------------------------- D3 Isolation Forest
def isolation_forest_flags(wide: pd.DataFrame, contamination: float = 0.01, seed: int = 0) -> pd.Series:
    flags = pd.Series(False, index=wide.index)
    feats = {"running": RAW["running"] + list(ROLLING), "stopped": RAW["stopped"]}
    for state, cols in feats.items():
        X = wide.loc[wide["state"] == state, cols].dropna()
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(X)
        flags.loc[X.index] = iso.predict(X) == -1
    return flags


# --------------------------------------------------------------------------- events & metrics
def to_events(flag: pd.Series, gap_s: float) -> pd.DataFrame:
    """Merge flagged rows that are closer than gap_s into alarm events."""
    t = flag.index[flag.fillna(False).to_numpy(bool)]
    if len(t) == 0:
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"),
                             "end": pd.Series(dtype="datetime64[ns]"), "n": pd.Series(dtype=int)})
    s = pd.Series(t)
    grp = (s.diff().dt.total_seconds() > gap_s).cumsum()
    return s.groupby(grp).agg(start="min", end="max", n="size").reset_index(drop=True)


def period_hours(wide: pd.DataFrame) -> tuple[float, float]:
    t0, t1 = wide.index.min(), min(wide.index.max(), CLEAN_END)
    return (CLEAN_START - t0).total_seconds() / 3600, (t1 - CLEAN_START).total_seconds() / 3600


def period_metrics(events: pd.DataFrame, wide: pd.DataFrame) -> dict:
    h_anom, h_clean = period_hours(wide)
    n_anom = int((events["start"] < CLEAN_START).sum())
    n_clean = int((events["start"] >= CLEAN_START).sum())
    return {"events_anomaly_period": n_anom, "events_clean_window": n_clean,
            "events_per_h_anomaly_period": n_anom / h_anom, "events_per_h_clean_window": n_clean / h_clean,
            "false_alarms_per_day": n_clean / h_clean * 24, "clean_hours": h_clean, "anomaly_hours": h_anom}


def typed_episodes(flags: pd.DataFrame, excess: pd.DataFrame, wide: pd.DataFrame, gap_s: float = 300) -> pd.DataFrame:
    """Group envelope flags into episodes per anomaly type (A/B/C) with peak and severity."""
    rows = []
    for typ, name in TYPE_NAMES.items():
        cols = [c for c in flags.columns if FEATURE_TYPE.get(c) == typ]
        if not cols:
            continue
        ev = to_events(flags[cols].any(axis=1), gap_s)
        for e in ev.itertuples(index=False):
            ex = excess.loc[e.start:e.end, cols]
            feat = ex.max().idxmax()
            t_peak = ex[feat].idxmax()
            rows.append({"type": typ, "type_name": name, "start": e.start, "end": e.end,
                         "duration_s": (e.end - e.start).total_seconds(), "flagged_rows": int(e.n),
                         "peak_feature": feat, "peak_value": float(wide.at[t_peak, feat]),
                         "peak_time": t_peak, "severity_sigma": float(ex[feat].max()),
                         "state_at_peak": wide.at[t_peak, "state"], "order_id": int(wide.at[e.start, "order_id"]),
                         "period": wide.at[e.start, "period"]})
    out = pd.DataFrame(rows)
    return out.sort_values("start").reset_index(drop=True) if len(out) else out


def match_episodes(episodes: pd.DataFrame, flag: pd.Series, tol_after_s: float = 0,
                   tol_before_s: float = 0) -> pd.DataFrame:
    """For each reference episode: was there a flag inside [start - tol_before, end + tol_after]?
    delay = time of the first such flag minus the episode start."""
    t = flag.index[flag.fillna(False).to_numpy(bool)].to_numpy()
    starts = (episodes["start"] - pd.Timedelta(seconds=tol_before_s)).to_numpy()
    ends = (episodes["end"] + pd.Timedelta(seconds=tol_after_s)).to_numpy()
    i0 = np.searchsorted(t, starts, side="left")
    i1 = np.searchsorted(t, ends, side="right")
    detected = i1 > i0
    first = np.where(detected, t[np.minimum(i0, max(len(t) - 1, 0))] if len(t) else starts, starts)
    delay = (first - episodes["start"].to_numpy()) / np.timedelta64(1, "s")
    return pd.DataFrame({"detected": detected, "delay_s": np.where(detected, delay, np.nan)}, index=episodes.index)


def deadband_mask(x: np.ndarray, d: float) -> np.ndarray:
    """Which samples an 'on change' archive with hysteresis d would store (d=0: every change)."""
    rec = np.zeros(len(x), dtype=bool)
    last = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        if np.isnan(last) or (abs(v - last) > d + 1e-9 if d > 0 else round(v, 1) != round(last, 1)):
            rec[i] = True
            last = v
    return rec


def cycle_phase_min(ts: pd.Series, cycle_h: int = 4) -> pd.Series:
    """Minute within a `cycle_h`-hour clock cycle (00:00, 04:00, 08:00, ...), 0 .. 60*cycle_h-1."""
    return (ts.dt.hour % cycle_h) * 60 + ts.dt.minute


def best_window_share(phase: pd.Series, window_min: int = 30, cycle_min: int = 240) -> tuple[float, int]:
    """Largest share of onsets inside any window of `window_min` minutes (circular); uniform = window/cycle."""
    if len(phase) == 0:
        return float("nan"), 0
    counts = np.bincount(phase.astype(int).to_numpy(), minlength=cycle_min)
    ext = np.concatenate([counts, counts[:window_min]])
    sums = np.convolve(ext, np.ones(window_min, dtype=int), mode="valid")[:cycle_min]
    start = int(np.argmax(sums))
    return float(sums[start] / len(phase)), start
