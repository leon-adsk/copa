"""
detectors.py - three methods to flag unusual values in the M7 data, plus helpers to evaluate them.
All columns are real zenon variable names; derived columns say how they are derived.

Every method looks at the three process variables per machine condition (column `machine (from CurMachSpeed)`):
    running: CurMachSpeed 990 or 1089  -> Sim/Global/M7_FillWeight, Sim/Global/M7_CompAir, Sim/Global/M7_Power
    stopped: CurMachSpeed 0            -> Sim/Global/M7_CompAir, Sim/Global/M7_Power
                                          (M7_FillWeight keeps the value of the last bottle while stopped)
Rows "order end: CurMachSpeed 1100 with MachSpeed 550" (one sample at the end of a run, also seen in the clean
window) and INVALID rows are not scored.

Method "rare values"            : how often does this value of the variable occur (same machine condition)?
                                  Values seen in fewer than 1 of 10,000 rows are flagged. Unsupervised.
Method "outside clean range"    : value outside the range seen in Andreas' clean window (04.11.2026 10:50-15:00),
                                  widened by one standard deviation of that window. Also for rolling mean / rolling
                                  standard deviation over 60 s while running.
Method "Isolation Forest"       : sklearn IsolationForest per machine condition. Unsupervised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from copa_data import (CLEAN_END, CLEAN_START, LABEL, MACHINE, RUN_ID, RUNNING, STOPPED, M7_CompAir,
                       M7_FillWeight, M7_Power)

VARIABLES = [M7_CompAir, M7_Power, M7_FillWeight]
SCORED = {RUNNING: [M7_FillWeight, M7_CompAir, M7_Power], STOPPED: [M7_CompAir, M7_Power]}


def rolling_name(variable: str, how: str) -> str:
    return f"{variable} [rolling {how}]"


ROLLING = {rolling_name(M7_FillWeight, "std"): (M7_FillWeight, "std"),
           rolling_name(M7_CompAir, "mean"): (M7_CompAir, "mean"),
           rolling_name(M7_Power, "mean"): (M7_Power, "mean")}           # computed for running rows only
UPPER_ONLY = {rolling_name(M7_FillWeight, "std")}                       # a low spread is not of interest
VARIABLE_OF = {**{v: v for v in VARIABLES}, **{k: v[0] for k, v in ROLLING.items()}}


# --------------------------------------------------------------------------- features
def window_seconds(step_s: float) -> int:
    """Rolling window: 60 s, but at least 5 samples for coarse sampling."""
    return int(max(60, 5 * step_s))


def add_rolling(wide: pd.DataFrame, step_s: float, center: bool = False) -> pd.DataFrame:
    """Rolling mean / std inside contiguous running stretches (never across a stop).
    center=False uses only past values (online); center=True also future values (batch only)."""
    w = wide.copy()
    seg = (w[MACHINE] != w[MACHINE].shift()).cumsum()
    run = w[MACHINE] == RUNNING
    win_s = window_seconds(step_s)
    minp = max(3, int(round(0.5 * win_s / step_s)))
    for name, (col, how) in ROLLING.items():
        s = w[col].where(run)
        r = s.groupby(seg).rolling(f"{win_s}s", center=center, min_periods=minp)
        r = r.std() if how == "std" else r.mean()
        w[name] = r.reset_index(level=0, drop=True).reindex(w.index).where(run)
    return w


# --------------------------------------------------------------------------- method: rare values
def frequency_table(values: pd.Series, tau: float = 1e-4) -> pd.DataFrame:
    """Every observed value with its number of rows and share, most frequent first."""
    vc = values.dropna().round(1).value_counts()
    t = pd.DataFrame({"value": vc.index.to_numpy(), "rows": vc.to_numpy()})
    t["share of rows %"] = 100 * t["rows"] / t["rows"].sum()
    t["cumulative share %"] = t["share of rows %"].cumsum()
    t["rank"] = np.arange(1, len(t) + 1)
    t["rare (< 1 in 10,000 rows)"] = t["share of rows %"] < 100 * tau
    return t


def fit_frequency(wide: pd.DataFrame) -> dict:
    return {(cond, var): wide.loc[wide[MACHINE] == cond, var].round(1).value_counts(normalize=True)
            for cond, variables in SCORED.items() for var in variables}


def frequency_shares(wide: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Share (0..1) of each row's value within its machine condition; values never seen in training get 0."""
    out = pd.DataFrame(np.nan, index=wide.index, columns=VARIABLES)
    for (cond, var), vc in model.items():
        m = (wide[MACHINE] == cond) & wide[var].notna()
        out.loc[m, var] = wide.loc[m, var].round(1).map(vc).fillna(0.0).to_numpy()
    return out


def frequency_flags(shares: pd.DataFrame, tau: float = 1e-4) -> pd.DataFrame:
    return shares.lt(tau) & shares.notna()


# --------------------------------------------------------------------------- method: outside clean range
def fit_envelope(ref: pd.DataFrame, raw_from: pd.DataFrame | None = None, margin_sd: float = 1.0) -> dict:
    """(low, high, mean, std) per (machine condition, column): min-max of `ref` widened by margin_sd * std.
    `raw_from` lets the raw variables use another reference than the rolling columns."""
    raw_ref = ref if raw_from is None else raw_from

    def lim(x):
        sd = x.std()
        return (x.min() - margin_sd * sd, x.max() + margin_sd * sd, x.mean(), sd)

    env = {(cond, var): lim(raw_ref.loc[raw_ref[MACHINE] == cond, var].dropna())
           for cond, variables in SCORED.items() for var in variables}
    for col in ROLLING:
        if col in ref:
            x = ref.loc[ref[MACHINE] == RUNNING, col].dropna()
            if len(x):
                env[(RUNNING, col)] = lim(x)
    return env


def envelope_excess(wide: pd.DataFrame, env: dict) -> pd.DataFrame:
    """Distance outside the clean range in units of the clean standard deviation (0 = inside)."""
    cols = sorted({c for _, c in env})
    out = pd.DataFrame(0.0, index=wide.index, columns=cols)
    for (cond, col), (lo, hi, mu, sd) in env.items():
        if col not in wide:
            continue
        m = (wide[MACHINE] == cond) & wide[col].notna()
        x = wide.loc[m, col]
        sd = sd if sd and sd > 0 else 1.0
        up = (x - hi).clip(lower=0) / sd
        down = 0 if col in UPPER_ONLY else (lo - x).clip(lower=0) / sd
        out.loc[m, col] = np.maximum(out.loc[m, col], up + down)
    return out


def envelope_flags(wide: pd.DataFrame, env: dict) -> pd.DataFrame:
    return envelope_excess(wide, env) > 0


# --------------------------------------------------------------------------- method: Isolation Forest
def isolation_forest_flags(wide: pd.DataFrame, contamination: float = 0.01, seed: int = 0) -> pd.Series:
    flags = pd.Series(False, index=wide.index)
    feats = {RUNNING: SCORED[RUNNING] + list(ROLLING), STOPPED: SCORED[STOPPED]}
    for cond, cols in feats.items():
        X = wide.loc[wide[MACHINE] == cond, cols].dropna()
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(X)
        flags.loc[X.index] = iso.predict(X) == -1
    return flags


# --------------------------------------------------------------------------- events, episodes, metrics
def to_events(flag: pd.Series, gap_s: float) -> pd.DataFrame:
    """Merge flagged rows closer than gap_s into one alarm event."""
    t = flag.index[flag.fillna(False).to_numpy(bool)]
    if len(t) == 0:
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"),
                             "end": pd.Series(dtype="datetime64[ns]"), "rows": pd.Series(dtype=int)})
    s = pd.Series(t)
    grp = (s.diff().dt.total_seconds() > gap_s).cumsum()
    return s.groupby(grp).agg(start="min", end="max", rows="size").reset_index(drop=True)


def period_hours(wide: pd.DataFrame) -> tuple[float, float]:
    t0, t1 = wide.index.min(), min(wide.index.max(), CLEAN_END)
    return (CLEAN_START - t0).total_seconds() / 3600, (t1 - CLEAN_START).total_seconds() / 3600


def period_metrics(events: pd.DataFrame, wide: pd.DataFrame) -> dict:
    h_anom, h_clean = period_hours(wide)
    n_anom = int((events["start"] < CLEAN_START).sum())
    n_clean = int((events["start"] >= CLEAN_START).sum())
    return {"alarm events until 04.11. 10:50": n_anom, "alarm events 04.11. 10:50-15:00": n_clean,
            "alarm events per hour until 10:50": n_anom / h_anom, "alarm events per hour 10:50-15:00": n_clean / h_clean,
            "false alarms per day": n_clean / h_clean * 24, "clean hours": h_clean, "anomaly hours": h_anom}


def episodes(flags: pd.DataFrame, excess: pd.DataFrame, wide: pd.DataFrame, gap_s: float = 300) -> pd.DataFrame:
    """Flags of one variable (raw value or its rolling column) closer than gap_s -> one episode of that variable."""
    rows = []
    for var in VARIABLES:
        cols = [c for c in flags.columns if VARIABLE_OF.get(c) == var]
        for e in to_events(flags[cols].any(axis=1), gap_s).itertuples(index=False):
            ex = excess.loc[e.start:e.end, cols]
            col = ex.max().idxmax()
            t_peak = ex[col].idxmax()
            rows.append({"variable": var, "start": e.start, "end": e.end,
                         "duration_s": (e.end - e.start).total_seconds(), "flagged rows": int(e.rows),
                         "column with largest excess": col, "value at peak": float(wide.at[t_peak, col]),
                         "peak time": t_peak, "excess in clean SD": float(ex[col].max()),
                         "machine at peak": wide.at[t_peak, MACHINE], RUN_ID: int(wide.at[e.start, RUN_ID]),
                         LABEL: wide.at[e.start, LABEL]})
    out = pd.DataFrame(rows)
    return out.sort_values("start").reset_index(drop=True) if len(out) else out


def match_episodes(eps: pd.DataFrame, flag: pd.Series, tol_after_s: float = 0, tol_before_s: float = 0) -> pd.DataFrame:
    """For each episode: is there a flag in [start - tol_before, end + tol_after]? delay of the first one."""
    t = flag.index[flag.fillna(False).to_numpy(bool)].to_numpy()
    starts = (eps["start"] - pd.Timedelta(seconds=tol_before_s)).to_numpy()
    ends = (eps["end"] + pd.Timedelta(seconds=tol_after_s)).to_numpy()
    i0 = np.searchsorted(t, starts, side="left")
    i1 = np.searchsorted(t, ends, side="right")
    detected = i1 > i0
    first = np.where(detected, t[np.minimum(i0, max(len(t) - 1, 0))] if len(t) else starts, starts)
    delay = (first - eps["start"].to_numpy()) / np.timedelta64(1, "s")
    return pd.DataFrame({"detected": detected, "delay_s": np.where(detected, delay, np.nan)}, index=eps.index)


def deadband_mask(x: np.ndarray, d: float) -> np.ndarray:
    """Which samples an archive 'recording on change' with hysteresis d would store (d=0: every change)."""
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
    """Minute within a cycle_h-hour clock cycle (00:00, 04:00, 08:00 ...)."""
    return (ts.dt.hour % cycle_h) * 60 + ts.dt.minute


def best_window_share(phase: pd.Series, window_min: int = 30, cycle_min: int = 240) -> tuple[float, int]:
    """Largest share of start times inside any window of window_min minutes (circular); uniform = window/cycle."""
    if len(phase) == 0:
        return float("nan"), 0
    counts = np.bincount(phase.astype(int).to_numpy(), minlength=cycle_min)
    ext = np.concatenate([counts, counts[:window_min]])
    sums = np.convolve(ext, np.ones(window_min, dtype=int), mode="valid")[:cycle_min]
    start = int(np.argmax(sums))
    return float(sums[start] / len(phase)), start
