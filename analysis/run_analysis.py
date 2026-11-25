"""
run_analysis.py - reproducible pipeline: raw zenon exports -> tables, figures, insights.
All tables and figures use the real zenon variable names (e.g. Sim/Global/M7_CompAir).

Run from the repo root (uses the project environment from pyproject.toml / uv.lock):
    uv run python analysis/run_analysis.py
Options:
    --out DIR           results folder                               (default: results/)
    --write-processed   also write the large analysis-ready CSVs     (opt-in, ~57 MB)
    --processed DIR     where those CSVs go                          (default: data/processed/)

Every number in results/metrics.json and results/INSIGHTS.md is computed here from data/txt/*.TXT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import copa_data as CD  # noqa: E402
import detectors as D  # noqa: E402
import plot_style as S  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from copa_data import (CLEAN_END, CLEAN_START, LABEL, LBL_ANOM, LBL_CLEAN, MACHINE, RUN_ID, RUNNING,  # noqa: E402
                       STOPPED, ORDER_END, INVALID, CurMachSpeed, DefectiveCount, M7_CompAir, M7_FillWeight,
                       M7_Power, ProcessedCount, WS_Cur_State)

E1, E2 = "export_2026-11-04", "export_2026-11-20"
TAU = 1e-4               # rare value = occurs in fewer than 1 of 10,000 rows (same variable, same machine condition)
EVENT_GAP_S = 120        # flagged rows closer than 2 min = one alarm event
EPISODE_GAP_S = 300      # flags of one variable closer than 5 min = one episode
IF_CONTAMINATION = 0.01  # Isolation Forest alarm budget: 1 % of the rows of each machine condition
ENVELOPE_MARGIN_SD = 1.0  # clean-window min-max widened by 1 standard deviation of the clean window
CYCLE_H = 4              # episode start times repeat in a 4-hour clock cycle (see fig10)
SAMPLING_FACTORS = [1, 2, 3, 5, 10, 20]   # x 3 s -> 3, 6, 9, 15, 30, 60 s
MAX_OFFSETS = 5
DEADBANDS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
VARS = [M7_CompAir, M7_Power, M7_FillWeight]
METHOD = {"rare": "rare values (< 1 in 10,000 rows)", "range": "outside clean-window range", "iforest": "Isolation Forest"}
RULES = {   # clean-window range of each variable, widened a little; conditions on real variables only
    f"{M7_CompAir} > 33 (running)": lambda w: (w[MACHINE] == RUNNING) & (w[M7_CompAir] > 33),
    f"{M7_CompAir} > 2.3 (stopped)": lambda w: (w[MACHINE] == STOPPED) & (w[M7_CompAir] > 2.3),
    f"{M7_Power} > 11.4 (running)": lambda w: (w[MACHINE] == RUNNING) & (w[M7_Power] > 11.4),
    f"{M7_FillWeight} < 493.6 or > 506.1 (running)":
        lambda w: (w[MACHINE] == RUNNING) & ((w[M7_FillWeight] < 493.6) | (w[M7_FillWeight] > 506.1)),
}

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    CD.atomic_write(path, lambda tmp: df.to_csv(tmp, index=index))


def sha256_prefix(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def fmt_ts(t) -> str:
    return pd.Timestamp(t).strftime("%d.%m.%Y %H:%M:%S")


def cond_short(cond: str) -> str:
    return "running" if cond == RUNNING else "stopped"


def valid_days(w: pd.DataFrame) -> float:
    return (w[MACHINE] != INVALID).sum() * w.attrs["step_s"] / 86400


# =============================================================================== data inventory
def inventory(raws: dict, wides: dict) -> tuple[pd.DataFrame, dict]:
    rows, info = [], {}
    for exp, raw in raws.items():
        w = wides[exp]
        days = (w.index.max() - w.index.min()).total_seconds() / 86400
        info[exp] = {"label": CD.EXPORTS[exp]["label"], "start": fmt_ts(w.index.min()), "end": fmt_ts(w.index.max()),
                     "days": days, "rows per cyclic variable": int(len(w)),
                     "timestamps without CurMachSpeed value": int((w[MACHINE] == INVALID).sum()), "runs (ProcessedCount resets + 1)": int(w[RUN_ID].nunique())}
        for arch, df in raw.items():
            p = CD.TXT_DIR / CD.EXPORTS[exp]["files"][arch]
            dts = df.sort_values(["Variable name", "timestamp"]).groupby("Variable name")["timestamp"].diff().dt.total_seconds()
            rows.append({"export": CD.EXPORTS[exp]["label"], "file": p.name, "bytes": p.stat().st_size,
                         "sha256 (first 12)": sha256_prefix(p), "rows": len(df), "INVALID rows": int(df["INVALID"].sum()),
                         "variables": ", ".join(sorted(df["Variable name"].unique())),
                         "first": fmt_ts(df["timestamp"].min()), "last": fmt_ts(df["timestamp"].max()),
                         "recording": "on change" if arch == "07" else f"cyclic {CD.EXPORTS[exp]['cyclic_s']} s",
                         "median interval s": float(dts.median()), "bytes per day": p.stat().st_size / days})
    return pd.DataFrame(rows), info


# =============================================================================== rules per day + explainer figures
def rule_hits_per_day(w: pd.DataFrame) -> pd.DataFrame:
    hits = pd.DataFrame({k: f(w) for k, f in RULES.items()})
    day = pd.Series(w.index.strftime("%Y-%m-%d"), index=w.index)
    if w.attrs["export"] == CD.LABELLED:
        day[w.index >= CLEAN_START] = "2026-11-04 10:50-15:00 (Andreas: no anomalies)"
        day[(w.index < CLEAN_START) & (w.index >= CLEAN_START.normalize())] = "2026-11-04 until 10:50"
    out = hits.groupby(day).sum()
    out["hours of data"] = (day.value_counts() * w.attrs["step_s"] / 3600).reindex(out.index).round(2)
    return out.reset_index(names="day")


NORMAL_BAND = {M7_CompAir: (25.8, 31.5), M7_Power: (8.8, 10.8), M7_FillWeight: (495.1, 504.6)}   # running, clean window
YLIM = {M7_CompAir: (0, 41), M7_Power: (3, 13), M7_FillWeight: (480, 520)}


def fig_normal_vs_anomaly(w: pd.DataFrame, path: Path) -> None:
    examples = {M7_CompAir: ("2026-10-30 20:05", "2026-10-30 20:25"), M7_Power: ("2026-11-03 06:38", "2026-11-03 06:58"),
                M7_FillWeight: ("2026-11-04 10:00", "2026-11-04 10:20")}
    normal = ("2026-11-04 12:00", "2026-11-04 12:20")
    fig, axes = plt.subplots(3, 2, figsize=(13, 8.4), sharey="row")
    for i, var in enumerate(VARS):
        band = NORMAL_BAND[var]
        for j, (a, b) in enumerate([normal, examples[var]]):
            ax = axes[i, j]
            s = w.loc[a:b]
            run = s[MACHINE] == RUNNING
            ax.axhspan(*band, color=S.GOOD, alpha=0.14, lw=0)
            ax.plot(s.index, s[var].where(run), color=S.BLUE, lw=1.0)
            if var != M7_FillWeight:
                ax.plot(s.index, s[var].where(~run), color=S.MUTED, lw=1.0)
            out = run & ((s[var] < band[0] - 0.05) | (s[var] > band[1] + 0.05))
            ax.scatter(s.index[out.to_numpy(bool)], s.loc[out, var], s=12, color=S.CRITICAL, zorder=3, linewidths=0)
            ax.set_ylim(*YLIM[var])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.set_xlabel(f"time of day on {pd.Timestamp(a):%d.%m.%Y}", fontsize=9.5)
            when = pd.Timestamp(a).strftime("%d.%m.%Y %H:%M") + "-" + pd.Timestamp(b).strftime("%H:%M")
            if j == 0:
                ax.set_ylabel(f"value of {var}\n(unit not given by COPA-DATA)", fontsize=9)
                ax.set_title(f"{when}  (Andreas: no anomalies)", fontsize=11.5, color="#006300")
            else:
                ax.set_title(f"{when}  (Andreas: anomalies): {int(out.sum())} values outside", fontsize=11.5,
                             color=S.CRITICAL)
    fig.legend(handles=[Patch(color=S.GOOD, alpha=0.3, label="range while running, 04.11. 10:50-15:00"),
                        Line2D([], [], color=S.BLUE, label=f"{CurMachSpeed} = 990/1089"),
                        Line2D([], [], color=S.MUTED, label=f"{CurMachSpeed} = 0"),
                        Line2D([], [], marker="o", ls="", color=S.CRITICAL, label="outside that range")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    fig.suptitle("Same variable, same scale, 20 minutes each: left without, right with anomalies", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    S.save(fig, path)


def fig_overview_10min(w1: pd.DataFrame, w2: pd.DataFrame, path: Path) -> None:
    w = pd.concat([w1, w2])
    w = w[~w.index.duplicated(keep="first")]
    run, stp = w[MACHINE] == RUNNING, w[MACHINE] == STOPPED
    # title, values, rule line(s) as in RULES, range seen in 10:50-15:00
    series = [(f"{M7_CompAir}   while {CurMachSpeed} = 990 or 1089", w[M7_CompAir].where(run), [33], "max 10:50-15:00: 31.5"),
              (f"{M7_CompAir}   while {CurMachSpeed} = 0", w[M7_CompAir].where(stp), [2.3], "max 10:50-15:00: 2.2"),
              (f"{M7_Power}   while {CurMachSpeed} = 990 or 1089", w[M7_Power].where(run), [11.4], "max 10:50-15:00: 10.8"),
              (f"{M7_FillWeight}   while {CurMachSpeed} = 990 or 1089", w[M7_FillWeight].where(run), [493.6, 506.1],
               "10:50-15:00: 495.1-504.6")]
    fig, axes = plt.subplots(len(series), 1, figsize=(13, 9.6), sharex=True)
    for ax, (title, x, rules, clean_range) in zip(axes, series):
        blocks = x.resample("10min")
        ax.axvspan(w1.index.min(), CLEAN_START, color=S.CRITICAL, alpha=0.06, lw=0)
        ax.axvspan(CLEAN_START, CLEAN_END, color=S.GOOD, alpha=0.25, lw=0)
        highest = blocks.max()                                  # highest value in each 10-minute block
        ax.plot(highest.index, highest.values, color=S.NORMAL, lw=0.8)
        red = (highest > max(rules)).to_numpy(bool)
        ax.scatter(highest.index[red], highest[red], s=6, color=S.CRITICAL, zorder=3, linewidths=0)
        if len(rules) == 2:                                     # fill weight: too low is a rule hit as well
            lowest = blocks.min()
            ax.plot(lowest.index, lowest.values, color=S.NORMAL, lw=0.8)
            red = (lowest < min(rules)).to_numpy(bool)
            ax.scatter(lowest.index[red], lowest[red], s=6, color=S.CRITICAL, zorder=3, linewidths=0)
        for rule in rules:
            ax.axhline(rule, color=S.INK2, lw=0.8)
        rule_text = f"> {rules[0]:g}" if len(rules) == 1 else f"< {rules[0]:g} or > {rules[1]:g}"
        ax.text(1.0, rules[0] if len(rules) == 1 else 0.5, f" {rule_text}\n ({clean_range})", va="center", ha="left",
                transform=ax.get_yaxis_transform() if len(rules) == 1 else ax.transAxes, fontsize=9, color=S.INK2)
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="normal")
        ax.set_ylabel("highest and lowest\nvalue in each 10 min" if len(rules) == 2 else "highest value\nin each 10 min",
                      fontsize=9)
    axes[-1].set_xlabel("date (2026); one point = one 10-minute block")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.legend(handles=[Patch(color=S.CRITICAL, alpha=0.15, label="export 04.11.2026 until 04.11. 10:50 (Andreas: anomalies)"),
                        Patch(color=S.GOOD, alpha=0.5, label="04.11. 10:50-15:00 (Andreas: no anomalies)"),
                        Patch(facecolor="white", edgecolor=S.AXIS, label="export 20.11.2026 (no statement from Andreas)"),
                        Line2D([], [], color=S.NORMAL, label="highest value in each 10 minutes (fill weight: also the lowest)"),
                        Line2D([], [], marker="o", ls="", color=S.CRITICAL, label="beyond the rule line")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=9.5)
    fig.suptitle("Highest value per 10 minutes (a value, not a count), all days of both exports", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, path)


# =============================================================================== characterisation (export 04.11.)
def characterise(w: pd.DataFrame, raw: dict) -> tuple[dict, dict]:
    h_anom, h_clean = D.period_hours(w)
    ev = raw["07"][~raw["07"]["INVALID"]]
    st = ev[ev["Variable name"] == WS_Cur_State]
    rate_rows = []
    for v in sorted(st["value"].unique()):
        t = st.loc[st["value"] == v, "timestamp"]
        a, c = int((t < CLEAN_START).sum()), int((t >= CLEAN_START).sum())
        rate_rows.append({WS_Cur_State: int(v), "meaning (Andreas)": CD.WS_STATES.get(int(v), str(v)),
                          "entries until 10:50": a, "per hour until 10:50": a / h_anom,
                          "entries 10:50-15:00": c, "per hour 10:50-15:00": c / h_clean})
    rng_rows = []
    for cond, variables in D.SCORED.items():
        for var in variables:
            for lbl in [LBL_ANOM, LBL_CLEAN]:
                x = w.loc[(w[MACHINE] == cond) & (w[LABEL] == lbl), var].dropna()
                rng_rows.append({MACHINE: cond, "variable": var, LABEL: lbl, "rows": len(x), "min": x.min(),
                                 "p0.1": x.quantile(.001), "median": x.median(), "p99.9": x.quantile(.999),
                                 "max": x.max(), "std": x.std()})
    tables = {"ws_cur_state_entries_per_hour": pd.DataFrame(rate_rows), "ranges_until_vs_after_1050": pd.DataFrame(rng_rows)}
    g = w[w[MACHINE] == ORDER_END]
    facts = {"anomaly hours": h_anom, "clean hours": h_clean,
             "order-end samples until 10:50": int((g[LABEL] == LBL_ANOM).sum()),
             "order-end samples 10:50-15:00": int((g[LABEL] == LBL_CLEAN).sum())}
    return tables, facts


def runs_table(w: pd.DataFrame) -> pd.DataFrame:
    g = w.dropna(subset=[ProcessedCount]).groupby(RUN_ID)
    o = pd.DataFrame({"start": g.apply(lambda d: d.index.min()), "end": g.apply(lambda d: d.index.max()),
                      f"max {ProcessedCount}": g[ProcessedCount].max(), f"max {DefectiveCount}": g[DefectiveCount].max()})
    o["duration h"] = (o["end"] - o["start"]).dt.total_seconds() / 3600
    o["DefectiveCount / ProcessedCount %"] = 100 * o[f"max {DefectiveCount}"] / o[f"max {ProcessedCount}"]
    o[LABEL] = w.groupby(RUN_ID)[LABEL].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    o["complete run"] = True
    o.iloc[[0, -1], o.columns.get_loc("complete run")] = False   # first and last run are cut by the export
    return o.reset_index()


# =============================================================================== method: rare values
def rare_events(w: pd.DataFrame, shares: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in D.to_events(flags.any(axis=1), EVENT_GAP_S).itertuples(index=False):
        f = flags.loc[e.start:e.end]
        sh = shares.loc[e.start:e.end].where(f)
        var_min = sh.min().idxmin()
        t_min = sh[var_min].idxmin()
        rows.append({"start": e.start, "end": e.end, "duration_s": (e.end - e.start).total_seconds(),
                     "rare rows": int(e.rows), "variables": " + ".join(c for c in f.columns if f[c].any()),
                     "rarest variable": var_min, "rarest value": float(w.at[t_min, var_min]),
                     "share of that value %": 100 * float(np.nanmin(sh.to_numpy())),
                     MACHINE: w.at[t_min, MACHINE], LABEL: w.at[e.start, LABEL]})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["share of that value %", "start"]).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def fig_frequency_ranking(w: pd.DataFrame, path: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    facts = {}
    for ax, var in zip(axes, [M7_FillWeight, M7_CompAir, M7_Power]):
        t = D.frequency_table(w.loc[w[MACHINE] == RUNNING, var], TAU)
        rare = t["rare (< 1 in 10,000 rows)"]
        ax.bar(t["rank"], t["share of rows %"], width=1.0, color=np.where(rare, S.CRITICAL, S.NORMAL), linewidth=0)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(S.pct_formatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.axhline(100 * TAU, color=S.INK2, lw=0.8)
        ax.text(t["rank"].max(), 100 * TAU * 1.25, "1 in 10,000 rows", ha="right", va="bottom", fontsize=9, color=S.INK2)
        top = t.iloc[0]
        n_rare, rows_rare = int(rare.sum()), int(t.loc[rare, "rows"].sum())
        ax.set_title(var, fontsize=11.5)
        ax.set_xlabel("rank of the value (1 = the value that occurs most often)", fontsize=9.5)
        ax.text(0.98, 0.97, f"most frequent: {top['value']:.1f} ({top['share of rows %']:.1f} % of rows)\n"
                f"{n_rare} rare values = {rows_rare} rows", transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.grid(axis="x", visible=False)
        facts[var] = {"distinct values": int(len(t)), "most frequent value": float(top["value"]),
                      "its share %": float(top["share of rows %"]), "rare values": n_rare, "rare rows": rows_rare,
                      "rows": int(t["rows"].sum())}
    axes[0].set_ylabel("share of all rows while running\n(%, log scale)")
    fig.suptitle(f"Values while running ({CurMachSpeed} = 990/1089), export 04.11.2026", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.legend(handles=[Patch(color=S.NORMAL, label="common value"),
                        Patch(color=S.CRITICAL, label="rare value: fewer than 1 in 10,000 rows")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    S.save(fig, path)
    return facts


def fig_rare_where(w: pd.DataFrame, flags: pd.DataFrame, path: Path, title: str) -> None:
    labelled = w.attrs["export"] == CD.LABELLED
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.4), sharex=True)
    for ax, var in zip(axes, [M7_FillWeight, M7_CompAir, M7_Power]):
        ok = w[MACHINE].isin([RUNNING] if var == M7_FillWeight else [RUNNING, STOPPED]) & w[var].notna()
        ax.scatter(w.index[ok.to_numpy(bool)], w.loc[ok, var], s=0.6, color=S.NORMAL, linewidths=0, rasterized=True)
        f = flags[var].reindex(w.index, fill_value=False)
        ax.scatter(w.index[f.to_numpy(bool)], w.loc[f, var], s=9, color=S.CRITICAL, linewidths=0, zorder=3)
        ax.set_ylabel(f"value of\n{var}", fontsize=9)
        if labelled:
            ax.axvspan(CLEAN_START, CLEAN_END, color=S.GOOD, alpha=0.13, lw=0)
            n_a, n_c = int((f & (w[LABEL] == LBL_ANOM)).sum()), int((f & (w[LABEL] == LBL_CLEAN)).sum())
            ax.text(1.005, 0.5, f"rare rows\nuntil 04.11. 10:50: {n_a}\n10:50-15:00: {n_c}", transform=ax.transAxes,
                    fontsize=9, va="center")
        else:
            ax.text(1.005, 0.5, f"rare rows: {int(f.sum())}", transform=ax.transAxes, fontsize=9, va="center")
    axes[0].set_ylim(478, 522)
    axes[-1].set_xlabel("date (2026)")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axes[0].set_title(title)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=S.NORMAL, label="all values"),
                        Line2D([], [], marker="o", ls="", color=S.CRITICAL, label="rare value"),
                        Patch(color=S.GOOD, alpha=0.3, label="04.11. 10:50-15:00 (Andreas: no anomalies)")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    S.save(fig, path)


# =============================================================================== method: outside clean range
def fig_examples(w: pd.DataFrame, eps: pd.DataFrame, env: dict, flags: pd.DataFrame, path: Path) -> list:
    picks = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, var in zip(axes, VARS):
        cand = eps[(eps["variable"] == var) & (eps[LABEL] == LBL_ANOM) & (eps["machine at peak"] == RUNNING)]
        cand = cand[cand["duration_s"] >= 120] if (cand["duration_s"] >= 120).any() else cand
        e = cand.sort_values("flagged rows", ascending=False).iloc[0]
        picks.append({"variable": var, "start": fmt_ts(e["start"]), "end": fmt_ts(e["end"])})
        a, b = e["start"] - pd.Timedelta(minutes=12), e["end"] + pd.Timedelta(minutes=8)
        s = w.loc[a:b]
        run = s[MACHINE] == RUNNING
        lo, hi = env[(RUNNING, var)][:2]
        ax.axhspan(lo, hi, color=S.GOOD, alpha=0.12, lw=0)
        ax.plot(s.index, s[var].where(run), color=S.VAR_COLOR[var], lw=1.0)
        ax.plot(s.index, s[var].where(~run), color=S.MUTED, lw=1.0)
        cols = [c for c in flags.columns if D.VARIABLE_OF.get(c) == var]
        for x in D.to_events(flags.loc[a:b, cols].any(axis=1), EPISODE_GAP_S).itertuples(index=False):
            ax.axvspan(x.start, x.end, color=S.CRITICAL, alpha=0.13, lw=0)
        ax.set_title(f"{var}, {pd.Timestamp(e['start']):%d.%m.%Y}", fontsize=11)
        ax.set_ylabel(f"value of {var}", fontsize=9)
        ax.set_xlabel(f"time of day on {pd.Timestamp(e['start']):%d.%m.%Y}", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_ylim(*YLIM[var])
    fig.legend(handles=[Patch(color=S.GOOD, alpha=0.3, label="range while running 04.11. 10:50-15:00 (+/- 1 SD)"),
                        Line2D([], [], color=S.MUTED, label=f"{CurMachSpeed} = 0"),
                        Patch(color=S.CRITICAL, alpha=0.3, label="episode outside that range")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, path)
    return picks


def fig_calendar(eps1: pd.DataFrame, eps2: pd.DataFrame, w1: pd.DataFrame, w2: pd.DataFrame, path: Path, rates: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.3))
    for i, var in enumerate(VARS):
        for eps in (eps1, eps2):
            ax.vlines(eps.loc[eps["variable"] == var, "start"], i - 0.32, i + 0.32, color=S.VAR_COLOR[var], lw=1.0)
    ax.axvspan(CLEAN_START, CLEAN_END, color=S.GOOD, alpha=0.18, lw=0)
    inv = w2.index[w2[MACHINE] == INVALID]
    if len(inv):
        ax.axvspan(inv.min(), inv.max(), color=S.MUTED, alpha=0.35, lw=0)
    ax.axvline(w1.index.max(), color=S.INK2, lw=0.8)
    ax.set_yticks(range(3), [f"{v}\n{rates[E1][v]:.1f}/day (export 04.11.) | {rates[E2][v]:.1f}/day (export 20.11.)"
                             for v in VARS], fontsize=9)
    ax.set_ylim(-0.6, 2.6)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set_title("Episodes outside the clean-window range (each tick = one episode)")
    ax.set_xlabel("date (2026)")
    fig.legend(handles=[Patch(color=S.GOOD, alpha=0.4, label="04.11. 10:50-15:00 (Andreas: no anomalies)"),
                        Patch(color=S.MUTED, alpha=0.5, label="INVALID rows (04.11. 15:00-15:45)")], loc="lower right", ncol=2,
               bbox_to_anchor=(1.0, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, path)


def fig_schedule(eps1: pd.DataFrame, eps2: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 4.8), sharex=True)
    bins = np.arange(0, CYCLE_H * 60 + 1, 15)
    for i, (eps, lab) in enumerate([(eps1, "export 04.11.2026"), (eps2, "export 20.11.2026")]):
        for j, var in enumerate(VARS):
            ax = axes[i, j]
            ax.hist(D.cycle_phase_min(eps.loc[eps["variable"] == var, "start"], CYCLE_H), bins=bins,
                    color=S.VAR_COLOR[var], rwidth=0.85)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(var, fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f"{lab}\nnumber of episodes", fontsize=9.5)
            ax.set_xticks(range(0, CYCLE_H * 60 + 1, 60), [f"+{h}h" for h in range(CYCLE_H + 1)])
    for ax in axes[1]:
        ax.set_xlabel("hours after 00:00, 04:00, 08:00, 12:00, 16:00 or 20:00\nwhen the episode starts", fontsize=9)
    fig.suptitle("Episode start times cluster at the same minutes of every 4-hour cycle", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    S.save(fig, path)


def overlap_share(events: pd.DataFrame, ref: pd.DataFrame) -> float:
    """Share of alarm events whose [start, end] overlaps at least one reference episode."""
    if len(events) == 0:
        return float("nan")
    order = ref["start"].sort_values().index
    rs, re_ = ref.loc[order, "start"].to_numpy(), ref.loc[order, "end"].to_numpy()
    run_max_end = np.maximum.accumulate(re_)
    idx = np.searchsorted(rs, events["end"].to_numpy(), side="right") - 1
    hit = (idx >= 0) & (run_max_end[np.clip(idx, 0, None)] >= events["start"].to_numpy())
    return float(hit.mean())


# =============================================================================== RQ figures
def fig_rq1(rq1: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    y = np.arange(len(rq1))
    is_ref = (rq1["method"] == METHOD["range"]).to_numpy()
    panels = [("episodes detected %", "episodes detected (%)", "{:.0f} %", True),
              ("alarms that hit an episode %", "alarms that hit an episode (%)", "{:.0f} %", True),
              ("median warning before peak s", f"median warning before the peak (s)\n{M7_CompAir} and {M7_Power}", "{:.0f} s", False)]
    for ax, (col, title, fmt, ref_panel) in zip(axes, panels):
        v = rq1[col].to_numpy(float)
        ax.barh(y, v, color=[S.NORMAL if (r and ref_panel) else S.BLUE for r in is_ref], height=0.55)
        for yi, vi, r in zip(y, v, is_ref):
            ax.text(vi, yi, " " + fmt.format(vi) + ("  (reference)" if (r and ref_panel) else ""), va="center", fontsize=10)
        ax.set_yticks(y, list(rq1["method"]) if ax is axes[0] else [""] * len(y))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("seconds" if "warning" in col else ("% of episodes" if "episodes" in col else "% of alarm events"), fontsize=9.5)
        ax.grid(axis="y", visible=False)
        if ref_panel:
            ax.set_xlim(0, 150)
            ax.set_xticks([0, 25, 50, 75, 100])
        else:
            ax.set_xlim(0, max(np.nanmax(v) * 1.45, 1))
    fig.text(0.01, -0.02, "All three: 0 false alarms in 04.11. 10:50-15:00 (4.2 h; one would equal 5.8 per day). "
             "Reference = episodes outside the clean-window range, so that method is 100 % by construction.",
             fontsize=9.5, color=S.INK2)
    fig.tight_layout()
    S.save(fig, path)


def fig_rq2(rq2: pd.DataFrame, path: Path, size=(7.5, 3.2), fs=10) -> None:
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(rq2))
    v = rq2["median extra delay online s"].to_numpy(float)
    ax.barh(y, v, color=[S.VAR_COLOR[x] for x in rq2["variable"]], height=0.5)
    for yi, vi, p in zip(y, v, rq2["batch episodes also found online %"]):
        ax.text(max(vi, 0), yi, f"  +{vi:.0f} s  |  {p:.0f} % also found online", va="center", fontsize=fs)
    ax.set_yticks(y, list(rq2["variable"]), fontsize=fs)
    ax.invert_yaxis()
    ax.set_xlim(0, max(np.nanmax(v) * 3.2, 10))
    ax.set_xlabel("median extra delay of the online version (seconds)")
    ax.set_ylabel("variable")
    ax.grid(axis="y", visible=False)
    ax.set_title("Replay of export 04.11.2026: 60 s window on past values only (online) vs. centred (batch)", fontsize=fs + 1)
    S.save(fig, path)


def fig_rq3_sampling(rq3s: pd.DataFrame, path: Path, size=(11, 3.7), stacked=False) -> None:
    fig, axes = plt.subplots(2 if stacked else 1, 1 if stacked else 2, figsize=size, sharex=stacked)
    for var in VARS:
        d = rq3s[rq3s["variable"] == var]
        for ax, col in zip(axes, ["detected %", "median delay s"]):
            ax.plot(d["interval s"], d[col], marker="o", ms=5, color=S.VAR_COLOR[var], lw=2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([3, 6, 9, 15, 30, 60], ["3", "6", "9", "15", "30", "60"])
        ax.minorticks_off()
    xl = "sampling interval in seconds\n(every k-th row of the 3 s export)"
    axes[-1].set_xlabel(xl)
    if not stacked:
        axes[0].set_xlabel(xl)
    axes[0].set_ylabel("% of the 3 s episodes\nstill found")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("median delay until\nthe first flag (seconds)")
    axes[0].set_title("Episodes found at 3 s that are still found at coarser sampling", fontsize=12)
    fig.legend(handles=[Line2D([], [], color=S.VAR_COLOR[v], marker="o", label=v) for v in VARS],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    S.save(fig, path)


def fig_rq3_deadband(rq3d: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for var in VARS:
        d = rq3d[rq3d["variable"] == var]
        axes[0].plot(d["hysteresis"], d["rows stored % of cyclic 3 s"], marker="o", ms=5, color=S.VAR_COLOR[var])
        axes[1].plot(d["hysteresis"], d["episodes still found %"], marker="o", ms=5, color=S.VAR_COLOR[var])
    axes[0].set_ylabel("rows stored\n(% of the cyclic 3 s rows)")
    axes[1].set_ylabel("% of the 3 s episodes\nstill found")
    for ax in axes:
        ax.set_xlabel("hysteresis = minimum change before a new row\nis stored (in the variable's own units)")
        ax.set_ylim(0, 105)
    axes[0].set_title("Recording on change: rows stored")
    axes[1].set_title("Episodes still found")
    fig.legend(handles=[Line2D([], [], color=S.VAR_COLOR[v], marker="o", label=v) for v in VARS],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    S.save(fig, path)


def fig_export_timeline(w1, w2, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 1.9))
    for i, (w, col) in enumerate([(w1, S.BLUE), (w2, S.ORANGE)]):
        ax.barh(i, (w.index.max() - w.index.min()).total_seconds() / 86400, left=mdates.date2num(w.index.min()),
                height=0.5, color=col, alpha=0.85)
        ax.text(mdates.date2num(w.index.min()), i - 0.36, CD.EXPORTS[w.attrs["export"]]["label"], fontsize=10, va="bottom")
    ax.barh(0, (CLEAN_END - CLEAN_START).total_seconds() / 86400, left=mdates.date2num(CLEAN_START), height=0.5, color=S.GOOD)
    inv = w2.index[w2[MACHINE] == INVALID]
    ax.barh(1, (inv.max() - inv.min()).total_seconds() / 86400, left=mdates.date2num(inv.min()), height=0.5, color=S.MUTED)
    ax.xaxis_date()
    ax.set_xlabel("date (2026)")
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(1.6, -0.8)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.legend(handles=[Patch(color=S.GOOD, label="04.11. 10:50-15:00 (Andreas: no anomalies)"),
                        Patch(color=S.MUTED, label="INVALID rows (04.11. 15:00-15:45)")], loc="upper right", ncol=2)
    fig.tight_layout()
    S.save(fig, path)


def slide_figures(w1, flags_rare, rq2, rq3s, out: Path) -> None:
    """Figures sized for a 16:9 slide (about 7.4 x 4.7 in)."""
    import matplotlib as mpl
    with mpl.rc_context({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 12,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11}):
        fig = plt.figure(figsize=(7.4, 4.75))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.7)
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        t = D.frequency_table(w1.loc[w1[MACHINE] == RUNNING, M7_FillWeight], TAU)
        rare = t["rare (< 1 in 10,000 rows)"]
        ax1.bar(t["rank"], t["share of rows %"], width=1.0, color=np.where(rare, S.CRITICAL, S.NORMAL), linewidth=0)
        ax1.set_yscale("log")
        ax1.yaxis.set_major_formatter(S.pct_formatter())
        ax1.yaxis.set_minor_formatter(plt.NullFormatter())
        ax1.axhline(100 * TAU, color=S.INK2, lw=0.8)
        ax1.text(t["rank"].max(), 100 * TAU * 1.3, "1 in 10,000 rows", ha="right", va="bottom", fontsize=11, color=S.INK2)
        ax1.set_title(f"1  Rank every value of {M7_FillWeight} by how often it occurs", fontsize=12.5)
        ax1.set_xlabel("rank of the value (1 = occurs most often)")
        ax1.set_ylabel("share of\nrunning rows")
        ax1.grid(axis="x", visible=False)
        run = w1[MACHINE] == RUNNING
        ax2.scatter(w1.index[run.to_numpy(bool)], w1.loc[run, M7_FillWeight], s=0.5, color=S.NORMAL, linewidths=0, rasterized=True)
        f = flags_rare[M7_FillWeight].reindex(w1.index, fill_value=False)
        ax2.scatter(w1.index[f.to_numpy(bool)], w1.loc[f, M7_FillWeight], s=10, color=S.CRITICAL, linewidths=0, zorder=3)
        ax2.axvspan(CLEAN_START, CLEAN_END, color=S.GOOD, alpha=0.18, lw=0)
        n_a, n_c = int((f & (w1[LABEL] == LBL_ANOM)).sum()), int((f & (w1[LABEL] == LBL_CLEAN)).sum())
        ax2.set_title(f"2  Where they lie: {n_a} rare rows until 04.11. 10:50, {n_c} in 10:50-15:00 (green)", fontsize=12.5)
        ax2.set_ylim(478, 522)
        ax2.set_ylabel(f"value of\n{M7_FillWeight}", fontsize=9.5)
        ax2.set_xlabel("date (2026)")
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        S.save(fig, out / "slide_rq1_rare_values.png")
        fig_rq2(rq2, out / "slide_rq2_online_vs_batch.png", size=(7.4, 3.6), fs=11.5)
        fig_rq3_sampling(rq3s, out / "slide_rq3_sampling.png", size=(7.4, 4.75), stacked=True)


# =============================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CD.REPO / "results"))
    ap.add_argument("--processed", default=str(CD.REPO / "data" / "processed"))
    ap.add_argument("--write-processed", action="store_true")
    ap.add_argument("--skip-processed", action="store_true", help=argparse.SUPPRESS)  # old flag, now the default
    args = ap.parse_args()
    OUT, FIG, TAB = Path(args.out), Path(args.out) / "figures", Path(args.out) / "tables"
    S.apply()
    M: dict = {"parameters": {"rare threshold (share of rows)": TAU, "alarm event gap s": EVENT_GAP_S,
                              "episode gap s": EPISODE_GAP_S, "Isolation Forest alarm budget": IF_CONTAMINATION,
                              "clean range widened by SD": ENVELOPE_MARGIN_SD,
                              "clean window": [fmt_ts(CLEAN_START), fmt_ts(CLEAN_END)]}}

    log("loading raw exports")
    w1, raw1 = CD.load_export(E1)
    w2, raw2 = CD.load_export(E2)
    wides, raws = {E1: w1, E2: w2}, {E1: raw1, E2: raw2}
    inv_df, inv = inventory(raws, wides)
    write_csv(inv_df, TAB / "data_inventory.csv")
    M["inventory"], M["inventory_files"] = inv, inv_df.to_dict(orient="records")

    log("rule hits per day + explainer figures")
    M["rule_hits_per_day"] = {}
    for exp, w in wides.items():
        t = rule_hits_per_day(w)
        write_csv(t, TAB / f"rule_hits_per_day_{exp}.csv")
        M["rule_hits_per_day"][exp] = t.to_dict(orient="records")
    fig_normal_vs_anomaly(w1, FIG / "fig00a_same_variable_without_and_with_anomaly.png")
    fig_overview_10min(w1, w2, FIG / "fig00b_highest_value_per_10min.png")

    log("rolling columns")
    r1 = D.add_rolling(w1, 3)                      # past values only (online-capable)
    r1c = D.add_rolling(w1, 3, center=True)        # centred (batch only)
    r2 = D.add_rolling(w2, 10)

    log("characterisation of export 04.11.2026")
    tabs, facts = characterise(w1, raw1)
    for k, v in tabs.items():
        write_csv(v, TAB / f"{k}.csv")
    sr = tabs["ws_cur_state_entries_per_hour"].set_index(WS_Cur_State)
    facts["M07.WS_Cur_State = 1024 (Equipment Failure) per hour"] = {
        "until 10:50": float(sr.at[1024, "per hour until 10:50"]), "10:50-15:00": float(sr.at[1024, "per hour 10:50-15:00"])}
    rng = tabs["ranges_until_vs_after_1050"]
    facts["ranges"] = {f"{r['variable']} | {r[MACHINE]} | {r[LABEL]}": {"min": float(r["min"]), "max": float(r["max"])}
                       for _, r in rng.iterrows()}
    runs = {}
    for exp, w in wides.items():
        runs[exp] = runs_table(w)
        write_csv(runs[exp], TAB / f"runs_{exp}.csv")
    pc = "DefectiveCount / ProcessedCount %"
    r1c_ = runs[E1][runs[E1]["complete run"]]
    facts["DefectiveCount / ProcessedCount % per run"] = {
        "complete runs until 10:50": r1c_.loc[r1c_[LABEL] == LBL_ANOM, pc].describe().round(3).to_dict(),
        "runs in 10:50-15:00": runs[E1].loc[runs[E1][LABEL] == LBL_CLEAN, pc].round(3).tolist(),
        "complete runs export 20.11.2026": runs[E2][runs[E2]["complete run"]][pc].describe().round(3).to_dict()}
    M["characterisation"] = facts

    log("method: rare values")
    rare = {}
    for exp, w in wides.items():
        model = D.fit_frequency(w)
        for (cond, var) in model:
            write_csv(D.frequency_table(w.loc[w[MACHINE] == cond, var], TAU),
                      TAB / "value_frequency" / f"{exp}__{cond_short(cond)}__{CD.safe_name(var)}.csv")
        sh = D.frequency_shares(w, model)
        fl = D.frequency_flags(sh, TAU)
        ev = rare_events(w, sh, fl)
        write_csv(ev, TAB / f"rare_value_events_{exp}.csv")
        rare[exp] = {"shares": sh, "flags": fl, "events": ev, "model": model}
    M["rare values: ranking while running, export 04.11.2026"] = fig_frequency_ranking(w1, FIG / "fig02_value_frequency_ranking.png")
    fig_rare_where(w1, rare[E1]["flags"], FIG / "fig03_rare_values_export_2026-11-04.png",
                   "Rare values over time, export 04.11.2026 (Andreas: anomalies until 04.11. ~10:50)")
    fig_rare_where(w2, rare[E2]["flags"], FIG / "fig03b_rare_values_export_2026-11-20.png",
                   "Rare values over time, export 20.11.2026 (no statement from Andreas)")
    ev1 = rare[E1]["events"]
    M["rare value events"] = {"export 04.11.2026 until 10:50": int((ev1[LABEL] == LBL_ANOM).sum()),
                              "export 04.11.2026 10:50-15:00": int((ev1[LABEL] == LBL_CLEAN).sum()),
                              "export 20.11.2026": int(len(rare[E2]["events"])),
                              "per day export 04.11.2026 until 10:50": int((ev1[LABEL] == LBL_ANOM).sum()) / (facts["anomaly hours"] / 24),
                              "per day export 20.11.2026": len(rare[E2]["events"]) / valid_days(w2)}

    log("method: outside clean-window range -> episodes")
    clean = r1[r1[LABEL] == LBL_CLEAN]
    env1 = D.fit_envelope(clean, margin_sd=ENVELOPE_MARGIN_SD)
    ex1 = D.envelope_excess(r1, env1)
    fl1 = ex1 > 0
    eps1 = D.episodes(fl1, ex1, r1, EPISODE_GAP_S)
    thin = D.add_rolling(w1.iloc[::3], 9)          # rolling limits for the 10 s export from the clean window thinned to 9 s
    env2 = D.fit_envelope(thin[thin[LABEL] == LBL_CLEAN], raw_from=clean, margin_sd=ENVELOPE_MARGIN_SD)
    ex2 = D.envelope_excess(r2, env2)
    fl2 = ex2 > 0
    eps2 = D.episodes(fl2, ex2, r2, EPISODE_GAP_S)
    for exp, eps in [(E1, eps1), (E2, eps2)]:
        write_csv(eps, TAB / f"episodes_outside_clean_range_{exp}.csv")
        lst = eps.sort_values("excess in clean SD", ascending=False).reset_index(drop=True)
        lst.insert(0, "rank", np.arange(1, len(lst) + 1))
        write_csv(lst, TAB / f"episodes_by_size_{exp}.csv")
    days1 = facts["anomaly hours"] / 24
    days2 = valid_days(w2)
    e1a = eps1[eps1[LABEL] == LBL_ANOM]
    rates = {E1: {v: int((e1a["variable"] == v).sum()) / days1 for v in VARS},
             E2: {v: int((eps2["variable"] == v).sum()) / days2 for v in VARS}}
    M["episodes"] = {"export 04.11.2026 until 10:50": {v: int((e1a["variable"] == v).sum()) for v in VARS},
                     "export 04.11.2026 10:50-15:00": int((eps1[LABEL] == LBL_CLEAN).sum()),
                     "export 20.11.2026": {v: int((eps2["variable"] == v).sum()) for v in VARS},
                     "per day": rates, "days until 10:50": days1, "valid days export 20.11.2026": days2,
                     "clean range used": {f"{c} | {k}": [float(x[0]), float(x[1])] for (c, k), x in env1.items()},
                     "median duration s (export 04.11.2026)": e1a.groupby("variable")["duration_s"].median().to_dict()}
    M["example episodes"] = fig_examples(r1, eps1, env1, fl1, FIG / "fig04_examples_outside_clean_range.png")
    fig_calendar(e1a, eps2, w1, w2, FIG / "fig05_episodes_over_time.png", rates)
    sched = []
    for exp, eps in [(E1, e1a), (E2, eps2)]:
        for v in VARS:
            ph = D.cycle_phase_min(eps.loc[eps["variable"] == v, "start"], CYCLE_H)
            share, start = D.best_window_share(ph, 30, CYCLE_H * 60)
            sched.append({"export": CD.EXPORTS[exp]["label"], "variable": v, "episodes": int(len(ph)),
                          "share starting in the busiest 30 min of the 4 h cycle %": 100 * share,
                          "busiest 30 min (h:mm after 00/04/08/12/16/20 h)":
                              f"{start // 60:d}:{start % 60:02d}-{(start + 30) // 60 % CYCLE_H:d}:{(start + 30) % 60:02d}",
                          "if start times were random %": 100 * 30 / (CYCLE_H * 60)})
    sched = pd.DataFrame(sched)
    write_csv(sched, TAB / "episode_start_times_4h_cycle.csv")
    M["episode start times"] = sched.to_dict(orient="records")
    fig_schedule(e1a, eps2, FIG / "fig10_episode_start_times_4h_cycle.png")

    log("RQ1 methods")
    ref = e1a.reset_index(drop=True)
    flags = {METHOD["rare"]: rare[E1]["flags"].any(axis=1), METHOD["range"]: fl1.any(axis=1),
             METHOD["iforest"]: D.isolation_forest_flags(r1, IF_CONTAMINATION)}
    mid = CLEAN_START + (CLEAN_END - CLEAN_START) / 2
    fa_range = 0
    for tr, te in [((CLEAN_START, mid), (mid, CLEAN_END)), ((mid, CLEAN_END), (CLEAN_START, mid))]:
        env_f = D.fit_envelope(r1.loc[tr[0]:tr[1] - pd.Timedelta(seconds=1)], margin_sd=ENVELOPE_MARGIN_SD)
        fa_range += len(D.to_events(D.envelope_flags(r1.loc[te[0]:te[1]], env_f).any(axis=1), EVENT_GAP_S))
    rq1 = []
    for name, f in flags.items():
        pm = D.period_metrics(D.to_events(f, EVENT_GAP_S), r1)
        if name == METHOD["range"]:   # out of sample: limits from one half of the clean window, alarms in the other
            pm["alarm events 04.11. 10:50-15:00"] = fa_range
            pm["false alarms per day"] = fa_range / pm["clean hours"] * 24
        mt = D.match_episodes(ref, f)
        evs = D.to_events(f, EVENT_GAP_S)
        first = ref["start"] + pd.to_timedelta(mt["delay_s"], unit="s")
        lead = (ref["peak time"] - first).dt.total_seconds()
        drift = ref["variable"].isin([M7_CompAir, M7_Power]) & mt["detected"]
        row = {"method": name, "flagged rows %": 100 * float(f.mean()), **pm,
               "episodes detected %": 100 * float(mt["detected"].mean()),
               "median delay s": float(mt["delay_s"].median()),
               "alarms that hit an episode %": 100 * overlap_share(evs[evs["start"] < CLEAN_START], ref),
               "median warning before peak s": float(lead[drift].median())}
        for v in VARS:
            row[f"episodes detected % | {v}"] = 100 * float(mt.loc[ref["variable"] == v, "detected"].mean())
        rq1.append(row)
    rq1 = pd.DataFrame(rq1)
    write_csv(rq1, TAB / "rq1_methods.csv")
    thr = []
    for tau in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        f = D.frequency_flags(rare[E1]["shares"], tau).any(axis=1)
        thr.append({"rare threshold (share of rows)": tau, "flagged rows %": 100 * float(f.mean()),
                    **D.period_metrics(D.to_events(f, EVENT_GAP_S), r1),
                    "episodes detected %": 100 * float(D.match_episodes(ref, f)["detected"].mean())})
    thr = pd.DataFrame(thr)
    write_csv(thr, TAB / "rq1_rare_values_threshold.csv")
    M["rq1"] = {"table": rq1.to_dict(orient="records"), "rare threshold": thr.to_dict(orient="records"),
                "reference episodes": {v: int((ref["variable"] == v).sum()) for v in VARS}}
    fig_rq1(rq1, FIG / "fig06_rq1_methods.png")

    log("RQ2 online vs batch")
    env_c = D.fit_envelope(r1c[r1c[LABEL] == LBL_CLEAN], margin_sd=ENVELOPE_MARGIN_SD)
    fl_b = D.envelope_flags(r1c, env_c)
    eps_b = D.episodes(fl_b, D.envelope_excess(r1c, env_c), r1c, EPISODE_GAP_S)
    ref_b = eps_b[eps_b[LABEL] == LBL_ANOM].reset_index(drop=True)
    rq2 = []
    for v in VARS:
        e = ref_b[ref_b["variable"] == v].reset_index(drop=True)
        cols = [c for c in fl1.columns if D.VARIABLE_OF.get(c) == v]
        mb = D.match_episodes(e, fl_b[cols].any(axis=1), tol_after_s=600)
        mo = D.match_episodes(e, fl1[cols].any(axis=1), tol_after_s=600)
        both = mb["detected"] & mo["detected"]
        diff = mo.loc[both, "delay_s"] - mb.loc[both, "delay_s"]
        rq2.append({"variable": v, "batch episodes": len(e),
                    "batch episodes also found online %": 100 * float(mo["detected"].mean()),
                    "median extra delay online s": float(diff.median()) if len(diff) else np.nan,
                    "p90 extra delay online s": float(diff.quantile(.9)) if len(diff) else np.nan})
    rq2 = pd.DataFrame(rq2)
    write_csv(rq2, TAB / "rq2_online_vs_batch.csv")
    sh_on = D.frequency_shares(w2, rare[E1]["model"])
    f_on = D.frequency_flags(sh_on, TAU).any(axis=1)
    f_ba = rare[E2]["flags"].any(axis=1)
    M["rq2"] = {"table": rq2.to_dict(orient="records"), "window s": D.window_seconds(3),
                "alarm events batch until 10:50": D.period_metrics(D.to_events(fl_b.any(axis=1), EVENT_GAP_S), r1c)["alarm events until 04.11. 10:50"],
                "alarm events online until 10:50": D.period_metrics(D.to_events(fl1.any(axis=1), EVENT_GAP_S), r1)["alarm events until 04.11. 10:50"],
                "rare values learnt on export 04.11. applied to export 20.11.": {
                    "rare rows": int(f_on.sum()), "alarm events": len(D.to_events(f_on, EVENT_GAP_S)),
                    "rare rows when learnt on export 20.11. itself": int(f_ba.sum()),
                    "alarm events when learnt on export 20.11. itself": len(D.to_events(f_ba, EVENT_GAP_S)),
                    "overlap of rare rows (Jaccard)": float((f_on & f_ba).sum() / max((f_on | f_ba).sum(), 1))}}
    fig_rq2(rq2, FIG / "fig07_rq2_online_vs_batch.png")

    log("RQ3 sampling interval")
    rq3s = []
    for k in SAMPLING_FACTORS:
        offs = list(range(min(k, MAX_OFFSETS)))
        acc, dl = {v: [] for v in VARS}, {v: [] for v in VARS}
        for off in offs:
            sub = D.add_rolling(w1.iloc[off::k], 3 * k)
            fk = D.envelope_flags(sub, D.fit_envelope(sub[sub[LABEL] == LBL_CLEAN], raw_from=clean, margin_sd=ENVELOPE_MARGIN_SD))
            for v in VARS:
                e = ref[ref["variable"] == v].reset_index(drop=True)
                cols = [c for c in fk.columns if D.VARIABLE_OF.get(c) == v]
                mt = D.match_episodes(e, fk[cols].any(axis=1), tol_after_s=D.window_seconds(3 * k))
                acc[v].append(mt["detected"].mean())
                dl[v].extend(mt["delay_s"].dropna().tolist())
        for v in VARS:
            rq3s.append({"interval s": 3 * k, "variable": v, "phase offsets averaged": len(offs),
                         "detected %": 100 * float(np.mean(acc[v])),
                         "median delay s": float(np.median(dl[v])) if dl[v] else np.nan,
                         "rows per day per variable": 86400 / (3 * k)})
    rq3s = pd.DataFrame(rq3s)
    write_csv(rq3s, TAB / "rq3_sampling_interval.csv")

    log("RQ3 hysteresis")
    rq3d = []
    for d in DEADBANDS:
        rec = w1.copy()
        stored = {}
        for v in VARS:
            m = D.deadband_mask(w1[v].to_numpy(float), d)
            stored[v] = int(m.sum())
            rec[v] = w1[v].where(m).ffill()
        rr = D.add_rolling(rec, 3)
        fd = D.envelope_flags(rr, D.fit_envelope(rr[rr[LABEL] == LBL_CLEAN], margin_sd=ENVELOPE_MARGIN_SD))
        for v in VARS:
            e = ref[ref["variable"] == v].reset_index(drop=True)
            cols = [c for c in fd.columns if D.VARIABLE_OF.get(c) == v]
            mt = D.match_episodes(e, fd[cols].any(axis=1), tol_after_s=60)
            rq3d.append({"hysteresis": d, "variable": v, "rows stored": stored[v],
                         "rows stored % of cyclic 3 s": 100 * stored[v] / int(w1[v].notna().sum()),
                         "episodes still found %": 100 * float(mt["detected"].mean()),
                         "median delay s": float(mt["delay_s"].median())})
    rq3d = pd.DataFrame(rq3d)
    write_csv(rq3d, TAB / "rq3_hysteresis.csv")
    order_end = {}
    for exp, w in wides.items():
        g = w[MACHINE] == ORDER_END
        n = len(D.to_events(g, 60))
        order_end[exp] = {"samples": int(g.sum()), "occurrences": n, "occurrences per day": n / valid_days(w),
                          "runs per day": int(w[RUN_ID].nunique()) / valid_days(w)}
    M["rq3"] = {"sampling": rq3s.to_dict(orient="records"), "hysteresis": rq3d.to_dict(orient="records"),
                "order-end samples (CurMachSpeed 1100, MachSpeed 550)": order_end,
                "bytes per day": {r["file"]: r["bytes per day"] for r in M["inventory_files"]}}
    fig_rq3_sampling(rq3s, FIG / "fig08_rq3_sampling_interval.png")
    fig_rq3_deadband(rq3d, FIG / "fig09_rq3_hysteresis.png")
    fig_export_timeline(w1, w2, FIG / "fig01_exports_timeline.png")

    log("slide figures")
    slide_figures(w1, rare[E1]["flags"], rq2, rq3s, FIG / "slides")

    if args.write_processed:
        log("writing processed data")
        P = Path(args.processed)
        o1 = r1.copy()
        o1["rare value (any variable)"] = rare[E1]["flags"].any(axis=1)
        o1["outside clean range (any column)"] = fl1.any(axis=1)
        o1["Isolation Forest flag"] = flags[METHOD["iforest"]]
        write_csv(o1, P / f"wide_{E1}.csv", index=True)
        o2 = r2.copy()
        o2["rare value (any variable)"] = rare[E2]["flags"].any(axis=1)
        o2["outside clean range (any column)"] = fl2.any(axis=1)
        write_csv(o2, P / f"wide_{E2}.csv", index=True)

    CD.atomic_write(OUT / "metrics.json", lambda tmp: Path(tmp).write_text(json.dumps(M, indent=2, default=str)))
    write_insights(M, OUT / "INSIGHTS.md")
    log("done")


# =============================================================================== insights
def write_insights(M: dict, path: Path) -> None:
    ch, ep = M["characterisation"], M["episodes"]
    rq1 = pd.DataFrame(M["rq1"]["table"]).set_index("method")
    s = pd.DataFrame(M["rq3"]["sampling"])
    hy = pd.DataFrame(M["rq3"]["hysteresis"])
    oe = M["rq3"]["order-end samples (CurMachSpeed 1100, MachSpeed 550)"]
    L = []
    a = L.append
    a("# Insights - COPA-DATA filler M7 (generated by analysis/run_analysis.py)\n")
    a("All numbers are computed from `data/txt/*.TXT`. Variable names are the real zenon names.\n")
    a("## Data")
    for exp, i in M["inventory"].items():
        a(f"- **{i['label']}**: {i['start']} -> {i['end']} ({i['days']:.2f} days), {i['rows per cyclic variable']:,} rows per "
          f"cyclic variable, {i['timestamps without CurMachSpeed value']} timestamps without a CurMachSpeed value, {i['runs (ProcessedCount resets + 1)']} runs.")
    a("- The exports do not overlap. Andreas' statement (anomalies until 04.11. ~10:50, none 10:50-15:00) covers the export of 04.11.2026 only.")
    a("\n## Simple rules (thresholds = range in 04.11. 10:50-15:00, widened a little)")
    for r in M["rule_hits_per_day"][E1]:
        a(f"- {r['day']}: " + ", ".join(f"{k}: {r[k]}" for k in RULES))
    a("\n## What differs between until 10:50 and 10:50-15:00 (export 04.11.2026)")
    ef = ch["M07.WS_Cur_State = 1024 (Equipment Failure) per hour"]
    a(f"- {WS_Cur_State} = 1024 (Equipment Failure): {ef['until 10:50']:.2f}/h until 10:50, {ef['10:50-15:00']:.2f}/h in 10:50-15:00 (no difference).")
    rg = ch["ranges"]
    for v, cond in [(M7_CompAir, RUNNING), (M7_CompAir, STOPPED), (M7_Power, RUNNING), (M7_FillWeight, RUNNING)]:
        x, y = rg[f"{v} | {cond} | {LBL_ANOM}"], rg[f"{v} | {cond} | {LBL_CLEAN}"]
        a(f"- {v} ({cond}): range {y['min']:.1f}-{y['max']:.1f} in 10:50-15:00, {x['min']:.1f}-{x['max']:.1f} until 10:50.")
    a(f"- Order-end sample ({ORDER_END}): {ch['order-end samples until 10:50']} until 10:50, {ch['order-end samples 10:50-15:00']} in 10:50-15:00.")
    a("\n## Method: rare values")
    for v, f in M["rare values: ranking while running, export 04.11.2026"].items():
        a(f"- {v} (running): {f['distinct values']} distinct values; most frequent {f['most frequent value']:.1f} = {f['its share %']:.2f} % of rows; "
          f"{f['rare values']} rare values = {f['rare rows']} rows.")
    r = M["rare value events"]
    a(f"- Alarm events: {r['export 04.11.2026 until 10:50']} until 10:50, {r['export 04.11.2026 10:50-15:00']} in 10:50-15:00; "
      f"export 20.11.2026: {r['export 20.11.2026']} ({r['per day export 20.11.2026']:.1f}/day).")
    a("\n## Method: outside the clean-window range (episodes, flags of one variable within 5 min)")
    for v in VARS:
        a(f"- {v}: {ep['export 04.11.2026 until 10:50'][v]} episodes until 10:50 ({ep['per day'][E1][v]:.1f}/day), "
          f"{ep['export 20.11.2026'][v]} in export 20.11.2026 ({ep['per day'][E2][v]:.1f}/day). 10:50-15:00: {ep['export 04.11.2026 10:50-15:00']} in total.")
    for r_ in M["episode start times"]:
        a(f"- {r_['export']}, {r_['variable']}: {r_['share starting in the busiest 30 min of the 4 h cycle %']:.0f} % of {r_['episodes']} episodes start in "
          f"{r_['busiest 30 min (h:mm after 00/04/08/12/16/20 h)']} (random would be {r_['if start times were random %']:.1f} %).")
    a("\n## RQ1 - methods (export 04.11.2026)")
    a("| method | flagged rows | false alarms per day (10:50-15:00) | episodes detected | alarms that hit an episode | median warning before peak |")
    a("|---|---|---|---|---|---|")
    for k, row in rq1.iterrows():
        a(f"| {k} | {row['flagged rows %']:.2f} % | {row['false alarms per day']:.1f} | {row['episodes detected %']:.0f} % | "
          f"{row['alarms that hit an episode %']:.0f} % | {row['median warning before peak s']:.0f} s |")
    a("\n## RQ2 - online vs. batch (replay of export 04.11.2026)")
    for r_ in M["rq2"]["table"]:
        a(f"- {r_['variable']}: {r_['batch episodes also found online %']:.0f} % of batch episodes also found online, "
          f"median extra delay {r_['median extra delay online s']:.0f} s.")
    a("\n## RQ3 - archive parameters")
    for v in VARS:
        x = s[s["variable"] == v].set_index("interval s")
        a(f"- {v}: episodes still found {x.at[3, 'detected %']:.0f} % at 3 s, {x.at[9, 'detected %']:.0f} % at 9 s, "
          f"{x.at[30, 'detected %']:.0f} % at 30 s, {x.at[60, 'detected %']:.0f} % at 60 s.")
    a(f"- Order-end samples: {oe[E1]['occurrences per day']:.2f}/day at 3 s vs. {oe[E2]['occurrences per day']:.2f}/day at 10 s "
      f"({oe[E1]['runs per day']:.1f} vs. {oe[E2]['runs per day']:.1f} runs per day).")
    for v in VARS:
        x = hy[hy["variable"] == v].set_index("hysteresis")
        a(f"- {v} recording on change: hysteresis 0 stores {x.at[0.0, 'rows stored % of cyclic 3 s']:.0f} %, 0.5 stores "
          f"{x.at[0.5, 'rows stored % of cyclic 3 s']:.0f} % ({x.at[0.5, 'episodes still found %']:.0f} % found), 2.0 stores "
          f"{x.at[2.0, 'rows stored % of cyclic 3 s']:.0f} % ({x.at[2.0, 'episodes still found %']:.0f} % found).")
    CD.atomic_write(path, lambda tmp: Path(tmp).write_text("\n".join(L) + "\n"))


if __name__ == "__main__":
    main()
