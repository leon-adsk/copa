"""
run_analysis.py - reproducible pipeline: raw zenon exports -> processed data, tables, figures, insights.

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
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TAU = 1e-4               # D1: rare = value occurs in < 1 of 10,000 rows of its operating state
EVENT_GAP_S = 120        # flagged rows closer than 2 min form one alarm event
EPISODE_GAP_S = 300      # envelope flags closer than 5 min form one anomaly episode (per type)
IF_CONTAMINATION = 0.01  # D3 alarm budget: 1 % of the rows of each state
ENVELOPE_MARGIN_SD = 1.0  # D2: clean min-max widened by 1 clean standard deviation (see detectors.py)
CYCLE_H = 4               # anomaly onsets repeat in a 4-hour clock cycle (found in the data, see fig10)
SAMPLING_FACTORS = [1, 2, 3, 5, 10, 20]   # x 3 s -> 3, 6, 9, 15, 30, 60 s
MAX_OFFSETS = 5
DEADBANDS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
SIG_LABEL = {"fill_weight": "Fill weight", "compair": "Compressed air", "power": "Power"}
DET_LABEL = {"D1": "D1 Frequency ranking", "D2": "D2 Clean envelope", "D3": "D3 Isolation Forest"}

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


# =============================================================================== 1 inventory
def inventory(raws: dict, wides: dict) -> tuple[pd.DataFrame, dict]:
    rows, info = [], {}
    for exp, raw in raws.items():
        w = wides[exp]
        dur_d = (w.index.max() - w.index.min()).total_seconds() / 86400
        info[exp] = {"start": fmt_ts(w.index.min()), "end": fmt_ts(w.index.max()), "days": dur_d,
                     "cyclic_rows_per_var": int(len(w)), "invalid_timestamps": int((w["state"] == "invalid").sum()),
                     "orders": int(w["order_id"].nunique()),
                     "state_share_pct": (w["state"].value_counts(normalize=True) * 100).round(3).to_dict()}
        for arch, df in raw.items():
            p = CD.TXT_DIR / CD.EXPORTS[exp]["files"][arch]
            dts = df.sort_values(["var", "ts"]).groupby("var")["ts"].diff().dt.total_seconds()
            rows.append({"export": exp, "archive": arch, "file": p.name, "bytes": p.stat().st_size,
                         "sha256_12": sha256_prefix(p), "rows": len(df), "invalid_rows": int(df["invalid"].sum()),
                         "variables": df["var"].nunique(), "first": fmt_ts(df["ts"].min()), "last": fmt_ts(df["ts"].max()),
                         "median_interval_s": float(dts.median()), "max_interval_s": float(dts.max()),
                         "recording": "on change" if arch == "07" else f"cyclic {CD.EXPORTS[exp]['cyclic_s']} s",
                         "bytes_per_day": p.stat().st_size / dur_d})
    return pd.DataFrame(rows), info


# =============================================================================== 2 characterisation
def characterise(w_old: pd.DataFrame, raw_old: dict) -> tuple[dict, dict]:
    h_anom, h_clean = D.period_hours(w_old)
    ev = raw_old["07"][~raw_old["07"]["invalid"]]
    st = ev[ev["var"] == "M07.WS_Cur_State"]
    rate_rows = []
    for v in sorted(st["value"].unique()):
        t = st.loc[st["value"] == v, "ts"]
        a, c = int((t < CD.CLEAN_START).sum()), int((t >= CD.CLEAN_START).sum())
        rate_rows.append({"ws_state": int(v), "name": CD.WS_STATES.get(int(v), str(v)),
                          "entries_anomaly_period": a, "per_h_anomaly_period": a / h_anom,
                          "entries_clean_window": c, "per_h_clean_window": c / h_clean})
    rng_rows = []
    for state, sigs in D.RAW.items():
        for sig in sigs:
            for per in ["anomaly", "clean"]:
                x = w_old.loc[(w_old["state"] == state) & (w_old["period"] == per), sig].dropna()
                rng_rows.append({"state": state, "signal": sig, "period": per, "rows": len(x), "min": x.min(),
                                 "p0.1": x.quantile(.001), "median": x.median(), "p99.9": x.quantile(.999),
                                 "max": x.max(), "std": x.std()})
    tables = {"state_event_rates": pd.DataFrame(rate_rows), "signal_ranges_by_period": pd.DataFrame(rng_rows)}
    g = w_old[w_old["state"] == "glitch"]
    facts = {"anomaly_hours": h_anom, "clean_hours": h_clean,
             "glitch_samples_anomaly_period": int((g["period"] == "anomaly").sum()),
             "glitch_samples_clean_window": int((g["period"] == "clean").sum())}
    return tables, facts


def orders_table(w: pd.DataFrame) -> pd.DataFrame:
    g = w.dropna(subset=["processed"]).groupby("order_id")
    o = pd.DataFrame({"start": g.apply(lambda d: d.index.min()), "end": g.apply(lambda d: d.index.max()),
                      "processed": g["processed"].max(), "defective": g["defective"].max()})
    o["duration_h"] = (o["end"] - o["start"]).dt.total_seconds() / 3600
    o["defect_pct"] = 100 * o["defective"] / o["processed"]
    o["period"] = w.groupby("order_id")["period"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    o["complete"] = True
    o.iloc[[0, -1], o.columns.get_loc("complete")] = False   # first/last order are cut by the export
    return o.reset_index()


# =============================================================================== 3 D1 frequency ranking
def rare_events(w: pd.DataFrame, shares: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    any_flag = flags.any(axis=1)
    ev = D.to_events(any_flag, EVENT_GAP_S)
    rows = []
    for e in ev.itertuples(index=False):
        f = flags.loc[e.start:e.end]
        sh = shares.loc[e.start:e.end].where(f)
        sigs = [c for c in f.columns if f[c].any()]
        min_share = float(np.nanmin(sh.to_numpy()))
        sig_min = sh.min().idxmin()
        t_min = sh[sig_min].idxmin()
        rows.append({"start": e.start, "end": e.end, "duration_s": (e.end - e.start).total_seconds(),
                     "rare_rows": int(e.n), "signals": "+".join(sigs), "rarest_signal": sig_min,
                     "rarest_value": float(w.at[t_min, sig_min]), "rarest_share_pct": 100 * min_share,
                     "state": w.at[t_min, "state"], "period": w.at[e.start, "period"]})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["rarest_share_pct", "start"]).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def fig_frequency_ranking(w: pd.DataFrame, path: Path) -> dict:
    sigs = ["fill_weight", "compair", "power"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    facts = {}
    for ax, sig in zip(axes, sigs):
        t = D.frequency_table(w.loc[w["state"] == "running", sig], TAU)
        colors = np.where(t["rare"], S.CRITICAL, S.NORMAL)
        ax.bar(t["rank"], t["share_pct"], width=1.0, color=colors, linewidth=0)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(S.pct_formatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.axhline(100 * TAU, color=S.INK2, lw=0.8)
        ax.text(t["rank"].max(), 100 * TAU * 1.25, "1 in 10,000 rows", ha="right", va="bottom", fontsize=9, color=S.INK2)
        top = t.iloc[0]
        n_rare, rows_rare = int(t["rare"].sum()), int(t.loc[t["rare"], "rows"].sum())
        ax.set_title(SIG_LABEL[sig] + " (running)")
        ax.set_xlabel("values ranked by frequency (1 = most frequent)")
        txt = (f"most frequent: {top['value']:.1f} ({top['share_pct']:.1f} % of rows)\n"
               f"{n_rare} rare values = {rows_rare} rows ({100 * rows_rare / t['rows'].sum():.2f} %)")
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=S.INK)
        ax.grid(axis="x", visible=False)
        facts[sig] = {"distinct_values": int(len(t)), "top_value": float(top["value"]), "top_share_pct": float(top["share_pct"]),
                      "rare_values": n_rare, "rare_rows": rows_rare, "rows": int(t["rows"].sum()),
                      "rare_value_range": [float(t.loc[t["rare"], "value"].min()), float(t.loc[t["rare"], "value"].max())]
                      if n_rare else None}
    axes[0].set_ylabel("share of rows (log scale)")
    fig.legend(handles=[Patch(color=S.NORMAL, label="common value"),
                        Patch(color=S.CRITICAL, label="rare value (< 1 in 10,000 rows) = anomaly candidate")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    S.save(fig, path)
    return facts


def fig_where(w: pd.DataFrame, flags: pd.DataFrame, path: Path, title: str, labelled: bool) -> None:
    sigs = ["fill_weight", "compair", "power"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.2), sharex=True)
    for ax, sig in zip(axes, sigs):
        ok = w["state"].isin(["running", "stopped"]) & w[sig].notna()
        if sig == "fill_weight":
            ok &= w["state"] == "running"
        ax.scatter(w.index[ok.to_numpy(bool)], w.loc[ok, sig], s=0.6, color=S.NORMAL, linewidths=0, rasterized=True)
        f = flags[sig].reindex(w.index, fill_value=False)
        ax.scatter(w.index[f.to_numpy(bool)], w.loc[f, sig], s=9, color=S.CRITICAL, linewidths=0, zorder=3)
        ax.set_ylabel(SIG_LABEL[sig])
        if labelled:
            ax.axvspan(CD.CLEAN_START, CD.CLEAN_END, color=S.GOOD, alpha=0.13, lw=0)
            n_a = int((f & (w["period"] == "anomaly")).sum())
            n_c = int((f & (w["period"] == "clean")).sum())
            ax.text(1.005, 0.5, f"rare rows\nanomaly period: {n_a}\nclean window: {n_c}", transform=ax.transAxes,
                    fontsize=9, va="center", color=S.INK)
        else:
            ax.text(1.005, 0.5, f"rare rows: {int(f.sum())}", transform=ax.transAxes, fontsize=9, va="center")
    if labelled:
        axes[0].text(CD.CLEAN_START, axes[0].get_ylim()[1], " clean window\n (COPA-DATA: no anomalies)",
                     fontsize=9, va="top", ha="left", color="#006300")
    if sigs[0] == "fill_weight":
        axes[0].set_ylim(478, 522)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axes[0].set_title(title)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=S.NORMAL, label="all values"),
                        Line2D([], [], marker="o", ls="", color=S.CRITICAL, label="rare value (D1)")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.9, 1.0))
    fig.tight_layout()
    S.save(fig, path)


# =============================================================================== 4 D2 envelope episodes
def fig_anomaly_types(w: pd.DataFrame, eps: pd.DataFrame, env: dict, flags: pd.DataFrame, path: Path) -> list:
    picks = []
    spec = [("A", "compair"), ("B", "power"), ("C", "fill_weight")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (typ, sig) in zip(axes, spec):
        cand = eps[(eps["type"] == typ) & (eps["period"] == "anomaly") & (eps["state_at_peak"] == "running")]
        cand = cand[cand["duration_s"] >= 120] if (cand["duration_s"] >= 120).any() else cand
        e = cand.sort_values("flagged_rows", ascending=False).iloc[0]
        picks.append({"type": typ, "start": fmt_ts(e["start"]), "end": fmt_ts(e["end"])})
        a, b = e["start"] - pd.Timedelta(minutes=12), e["end"] + pd.Timedelta(minutes=8)
        s = w.loc[a:b]
        run = s["state"] == "running"
        lo, hi = env[("running", sig)][:2]
        ax.axhspan(lo, hi, color=S.GOOD, alpha=0.12, lw=0)
        ax.plot(s.index, s[sig].where(run), color=S.TYPE_COLOR[typ], lw=1.0)
        ax.plot(s.index, s[sig].where(~run), color=S.MUTED, lw=1.0)
        cols = [c for c in flags.columns if D.FEATURE_TYPE.get(c) == typ]
        ev = D.to_events(flags.loc[a:b, cols].any(axis=1), EPISODE_GAP_S)
        for x in ev.itertuples(index=False):
            ax.axvspan(x.start, x.end, color=S.CRITICAL, alpha=0.13, lw=0)
        ax.set_title(f"{typ}: {D.TYPE_NAMES[typ]}")
        ax.set_ylabel(SIG_LABEL[sig])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.text(0.01, 0.02, pd.Timestamp(e["start"]).strftime("%d.%m.%Y"), transform=ax.transAxes, fontsize=9, color=S.INK2)
        if sig == "compair":
            ax.set_ylim(0, max(40, s[sig].max() + 2))
        if sig == "fill_weight":
            ax.set_ylim(480, 520)
    fig.legend(handles=[Patch(color=S.GOOD, alpha=0.3, label="normal range (clean window, running, +/- 1 SD)"),
                        Line2D([], [], color=S.MUTED, label="machine stopped"),
                        Patch(color=S.CRITICAL, alpha=0.3, label="flagged episode (D2)")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, path)
    return picks


def fig_calendar(eps_old: pd.DataFrame, eps_new: pd.DataFrame, w_old: pd.DataFrame, w_new: pd.DataFrame,
                 path: Path, rates: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.2))
    types = ["A", "B", "C"]
    for i, typ in enumerate(types):
        for eps in (eps_old, eps_new):
            e = eps[eps["type"] == typ]
            ax.vlines(e["start"], i - 0.32, i + 0.32, color=S.TYPE_COLOR[typ], lw=1.0)
    ax.axvspan(CD.CLEAN_START, CD.CLEAN_END, color=S.GOOD, alpha=0.18, lw=0)
    inv = w_new.index[w_new["state"] == "invalid"]
    if len(inv):
        ax.axvspan(inv.min(), inv.max(), color=S.MUTED, alpha=0.35, lw=0)
    ax.axvline(w_old.index.max(), color=S.INK2, lw=0.8)
    ax.set_yticks(range(3), [f"{t}: {D.TYPE_NAMES[t]}\n{rates['old_3s'][t]:.1f}/day (3 s) | {rates['new_10s'][t]:.1f}/day (10 s)"
                             for t in types])
    ax.set_ylim(-0.6, 2.6)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.text(w_old.index.min(), -0.55, "Export 1 (3 s)", fontsize=10, va="top", color=S.INK2)
    ax.text(w_old.index.max() + pd.Timedelta(hours=6), -0.55, "Export 2 (10 s)", fontsize=10, va="top", color=S.INK2)
    ax.set_title("Where the anomaly episodes lie (each tick = one episode, D2 clean envelope)")
    fig.legend(handles=[Patch(color=S.GOOD, alpha=0.4, label="clean window (COPA-DATA)"),
                        Patch(color=S.MUTED, alpha=0.5, label="INVALID rows (outage)")],
               loc="lower right", ncol=2, bbox_to_anchor=(1.0, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, path)


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
    M: dict = {"parameters": {"tau": TAU, "event_gap_s": EVENT_GAP_S, "episode_gap_s": EPISODE_GAP_S,
                              "if_contamination": IF_CONTAMINATION, "envelope_margin_sd": ENVELOPE_MARGIN_SD, "clean_window": [fmt_ts(CD.CLEAN_START), fmt_ts(CD.CLEAN_END)]}}

    # ---------------------------------------------------------------- load
    log("loading raw exports")
    w_old, raw_old = CD.load_export("old_3s")
    w_new, raw_new = CD.load_export("new_10s")
    wides, raws = {"old_3s": w_old, "new_10s": w_new}, {"old_3s": raw_old, "new_10s": raw_new}
    inv_df, inv = inventory(raws, wides)
    write_csv(inv_df, TAB / "data_inventory.csv")
    M["inventory"] = inv
    M["inventory_files"] = inv_df.to_dict(orient="records")

    log("rolling features")
    r_old = D.add_rolling(w_old, 3)                       # causal (online-capable)
    r_old_c = D.add_rolling(w_old, 3, center=True)        # centred (batch only)
    r_new = D.add_rolling(w_new, 10)

    # ---------------------------------------------------------------- characterisation (labelled export 1)
    log("characterisation")
    tabs, facts = characterise(w_old, raw_old)
    for k, v in tabs.items():
        write_csv(v, TAB / f"{k}.csv")
    sr = tabs["state_event_rates"].set_index("name")
    facts["equipment_failure_per_h"] = {"anomaly_period": float(sr.at["Equipment Failure", "per_h_anomaly_period"]),
                                        "clean_window": float(sr.at["Equipment Failure", "per_h_clean_window"])}
    rng = tabs["signal_ranges_by_period"].set_index(["state", "signal", "period"])
    facts["ranges"] = {f"{st}/{sg}/{pe}": {"min": float(rng.at[(st, sg, pe), "min"]), "max": float(rng.at[(st, sg, pe), "max"])}
                       for st, sg, pe in rng.index}
    orders = {}
    for exp, w in wides.items():
        o = orders_table(w)
        write_csv(o, TAB / f"orders_{exp}.csv")
        orders[exp] = o
    oc = orders["old_3s"][orders["old_3s"]["complete"]]
    on = orders["new_10s"][orders["new_10s"]["complete"]]
    facts["defect_pct"] = {
        "old_anomaly_orders": oc.loc[oc["period"] == "anomaly", "defect_pct"].describe().round(3).to_dict(),
        "old_clean_orders": orders["old_3s"].loc[orders["old_3s"]["period"] == "clean", "defect_pct"].round(3).tolist(),
        "new_orders": on["defect_pct"].describe().round(3).to_dict()}
    M["characterisation"] = facts

    # ---------------------------------------------------------------- D1 frequency ranking (user's idea)
    log("D1 frequency ranking")
    d1 = {}
    for exp, w in wides.items():
        model = D.fit_frequency(w)
        for (state, sig), _ in model.items():
            t = D.frequency_table(w.loc[w["state"] == state, sig], TAU)
            write_csv(t, TAB / "frequency_ranking" / f"{exp}__{state}__{sig}.csv")
        sh = D.frequency_shares(w, model)
        fl = D.frequency_flags(sh, TAU)
        ev = rare_events(w, sh, fl)
        write_csv(ev, TAB / f"rare_value_events_{exp}.csv")
        d1[exp] = {"shares": sh, "flags": fl, "events": ev, "model": model}
    M["frequency_ranking_old_running"] = fig_frequency_ranking(w_old, FIG / "fig02_frequency_ranking.png")
    fig_where(w_old, d1["old_3s"]["flags"], FIG / "fig03_where_rare_values_lie_export1.png",
              "Where the rare values lie - Export 1 (3 s), labelled by COPA-DATA", labelled=True)
    fig_where(w_new, d1["new_10s"]["flags"], FIG / "fig03b_where_rare_values_lie_export2.png",
              "Where the rare values lie - Export 2 (10 s), no label from COPA-DATA", labelled=False)
    ev_old = d1["old_3s"]["events"]
    M["d1_rare_events"] = {
        "old_anomaly_period": int((ev_old["period"] == "anomaly").sum()),
        "old_clean_window": int((ev_old["period"] == "clean").sum()),
        "new_total": int(len(d1["new_10s"]["events"])),
        "new_per_day": len(d1["new_10s"]["events"]) / ((w_new["state"] != "invalid").sum() * 10 / 86400),
        "old_anomaly_per_day": int((ev_old["period"] == "anomaly").sum()) / (facts["anomaly_hours"] / 24)}

    # ---------------------------------------------------------------- D2 envelope + typed episodes
    log("D2 envelope episodes")
    clean = r_old[r_old["period"] == "clean"]
    env_old = D.fit_envelope(clean)
    ex_old = D.envelope_excess(r_old, env_old)
    fl_old = ex_old > 0
    eps_old = D.typed_episodes(fl_old, ex_old, r_old, EPISODE_GAP_S)
    # 10 s export: raw limits from the full clean window, rolling limits from the clean window thinned to 9 s
    thin = D.add_rolling(w_old.iloc[::3], 9)
    env_new = D.fit_envelope(thin[thin["period"] == "clean"], raw_from=clean)
    ex_new = D.envelope_excess(r_new, env_new)
    fl_new = ex_new > 0
    eps_new = D.typed_episodes(fl_new, ex_new, r_new, EPISODE_GAP_S)
    for exp, eps in [("old_3s", eps_old), ("new_10s", eps_new)]:
        write_csv(eps, TAB / f"anomaly_episodes_{exp}.csv")
        lst = eps.sort_values("severity_sigma", ascending=False).reset_index(drop=True)
        lst.insert(0, "rank", np.arange(1, len(lst) + 1))
        write_csv(lst, TAB / f"anomaly_list_by_severity_{exp}.csv")
    days_old_anom = facts["anomaly_hours"] / 24
    days_new_valid = (w_new["state"] != "invalid").sum() * 10 / 86400
    rates = {"old_3s": {t: int(((eps_old["type"] == t) & (eps_old["period"] == "anomaly")).sum()) / days_old_anom for t in "ABC"},
             "new_10s": {t: int((eps_new["type"] == t).sum()) / days_new_valid for t in "ABC"}}
    M["d2_episodes"] = {
        "old_anomaly_period": {t: int(((eps_old["type"] == t) & (eps_old["period"] == "anomaly")).sum()) for t in "ABC"},
        "old_clean_window": int((eps_old["period"] == "clean").sum()),
        "new": {t: int((eps_new["type"] == t).sum()) for t in "ABC"},
        "per_day": rates, "days_old_anomaly": days_old_anom, "days_new_valid": days_new_valid,
        "envelope_old": {f"{k[0]}/{k[1]}": [float(v[0]), float(v[1])] for k, v in env_old.items()},
        "median_duration_s_old": eps_old.groupby("type")["duration_s"].median().to_dict()}
    M["example_episodes"] = fig_anomaly_types(r_old, eps_old, env_old, fl_old, FIG / "fig04_anomaly_types.png")
    fig_calendar(eps_old, eps_new, w_old, w_new, FIG / "fig05_anomaly_calendar.png", rates)
    sched = []
    for exp, eps in [("old_3s", eps_old[eps_old["period"] == "anomaly"]), ("new_10s", eps_new)]:
        for t in "ABC":
            ph = D.cycle_phase_min(eps.loc[eps["type"] == t, "start"], CYCLE_H)
            share, start = D.best_window_share(ph, 30, CYCLE_H * 60)
            sched.append({"export": exp, "type": t, "type_name": D.TYPE_NAMES[t], "episodes": int(len(ph)),
                          "best_30min_share_pct": 100 * share, "best_window_start_min": start,
                          "best_window": f"{start // 60:d}:{start % 60:02d}-{(start + 30) // 60 % CYCLE_H:d}:{(start + 30) % 60:02d}",
                          "uniform_expectation_pct": 100 * 30 / (CYCLE_H * 60)})
    sched = pd.DataFrame(sched)
    write_csv(sched, TAB / "anomaly_schedule_4h_cycle.csv")
    M["schedule"] = sched.to_dict(orient="records")
    fig_schedule(eps_old[eps_old["period"] == "anomaly"], eps_new, FIG / "fig10_anomaly_schedule.png")

    # ---------------------------------------------------------------- RQ1 detector comparison (export 1)
    log("RQ1 detector comparison")
    ref = eps_old[eps_old["period"] == "anomaly"].reset_index(drop=True)
    det_flags = {"D1": d1["old_3s"]["flags"].any(axis=1), "D2": fl_old.any(axis=1),
                 "D3": D.isolation_forest_flags(r_old, IF_CONTAMINATION)}
    # D2 false alarms out of sample: 2-fold split of the clean window by time
    mid = CD.CLEAN_START + (CD.CLEAN_END - CD.CLEAN_START) / 2
    fa_d2 = 0
    for tr, te in [((CD.CLEAN_START, mid), (mid, CD.CLEAN_END)), ((mid, CD.CLEAN_END), (CD.CLEAN_START, mid))]:
        env_f = D.fit_envelope(r_old.loc[tr[0]:tr[1] - pd.Timedelta(seconds=1)])
        test = r_old.loc[te[0]:te[1]]
        fa_d2 += len(D.to_events(D.envelope_flags(test, env_f).any(axis=1), EVENT_GAP_S))
    rq1 = []
    for k, f in det_flags.items():
        pm = D.period_metrics(D.to_events(f, EVENT_GAP_S), r_old)
        if k == "D2":
            pm["false_alarms_per_day"] = fa_d2 / pm["clean_hours"] * 24
            pm["events_clean_window"] = fa_d2
            pm["fa_note"] = "2-fold out-of-sample on the clean window"
        mt = D.match_episodes(ref, f)
        evs = D.to_events(f, EVENT_GAP_S)
        evs_a = evs[evs["start"] < CD.CLEAN_START]
        first = ref["start"] + pd.to_timedelta(mt["delay_s"], unit="s")
        lead = (ref["peak_time"] - first).dt.total_seconds()
        drift = ref["type"].isin(["A", "B"]) & mt["detected"]
        row = {"detector": k, "label": DET_LABEL[k], "flagged_pct": 100 * float(f.mean()), **pm,
               "episodes_detected_pct": 100 * float(mt["detected"].mean()),
               "median_delay_s": float(mt["delay_s"].median()),
               "alarm_events_on_reference_pct": 100 * overlap_share(evs_a, ref),
               "median_warning_before_drift_peak_s": float(lead[drift].median())}
        for t in "ABC":
            m = ref["type"] == t
            row[f"detected_pct_{t}"] = 100 * float(mt.loc[m, "detected"].mean())
            row[f"median_delay_s_{t}"] = float(mt.loc[m, "delay_s"].median())
        rq1.append(row)
    rq1 = pd.DataFrame(rq1)
    write_csv(rq1, TAB / "rq1_detector_comparison.csv")
    sens = []
    for tau in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        f = D.frequency_flags(d1["old_3s"]["shares"], tau).any(axis=1)
        pm = D.period_metrics(D.to_events(f, EVENT_GAP_S), r_old)
        sens.append({"tau": tau, "flagged_pct": 100 * float(f.mean()), **pm,
                     "episodes_detected_pct": 100 * float(D.match_episodes(ref, f)["detected"].mean())})
    sens = pd.DataFrame(sens)
    write_csv(sens, TAB / "rq1_d1_threshold_sensitivity.csv")
    M["rq1"] = {"table": rq1.to_dict(orient="records"), "d1_sensitivity": sens.to_dict(orient="records"),
                "reference_episodes": {t: int((ref["type"] == t).sum()) for t in "ABC"}}
    fig_rq1(rq1, FIG / "fig06_rq1_detectors.png")

    # ---------------------------------------------------------------- RQ2 online vs batch
    log("RQ2 online vs batch")
    env_c = D.fit_envelope(r_old_c[r_old_c["period"] == "clean"])
    fl_batch = D.envelope_flags(r_old_c, env_c)
    ex_batch = D.envelope_excess(r_old_c, env_c)
    eps_batch = D.typed_episodes(fl_batch, ex_batch, r_old_c, EPISODE_GAP_S)
    ref_b = eps_batch[eps_batch["period"] == "anomaly"].reset_index(drop=True)
    rq2_rows = []
    for t in "ABC":
        e = ref_b[ref_b["type"] == t].reset_index(drop=True)
        cols = [c for c in fl_old.columns if D.FEATURE_TYPE.get(c) == t]
        mb = D.match_episodes(e, fl_batch[cols].any(axis=1), tol_after_s=600)
        mo = D.match_episodes(e, fl_old[cols].any(axis=1), tol_after_s=600)
        both = mb["detected"] & mo["detected"]
        diff = (mo.loc[both, "delay_s"] - mb.loc[both, "delay_s"])
        rq2_rows.append({"type": t, "type_name": D.TYPE_NAMES[t], "batch_episodes": len(e),
                         "online_detected_pct": 100 * float(mo["detected"].mean()),
                         "median_extra_delay_online_s": float(diff.median()) if len(diff) else np.nan,
                         "p90_extra_delay_online_s": float(diff.quantile(.9)) if len(diff) else np.nan,
                         "share_detected_later_pct": 100 * float((diff > 0).mean()) if len(diff) else np.nan})
    rq2 = pd.DataFrame(rq2_rows)
    pm_b = D.period_metrics(D.to_events(fl_batch.any(axis=1), EVENT_GAP_S), r_old_c)
    pm_o = D.period_metrics(D.to_events(fl_old.any(axis=1), EVENT_GAP_S), r_old)
    # D1 transfer: model learnt on export 1 (3 s) applied to the export 2 stream vs. learnt on export 2 itself
    sh_on = D.frequency_shares(w_new, d1["old_3s"]["model"])
    f_on = D.frequency_flags(sh_on, TAU).any(axis=1)
    f_ba = d1["new_10s"]["flags"].any(axis=1)
    jacc = float((f_on & f_ba).sum() / max((f_on | f_ba).sum(), 1))
    rq2_extra = {"alarm_events_batch_anomaly_period": pm_b["events_anomaly_period"],
                 "alarm_events_online_anomaly_period": pm_o["events_anomaly_period"],
                 "d1_transfer": {"online_rare_rows": int(f_on.sum()), "batch_rare_rows": int(f_ba.sum()),
                                 "online_events": len(D.to_events(f_on, EVENT_GAP_S)),
                                 "batch_events": len(D.to_events(f_ba, EVENT_GAP_S)),
                                 "jaccard_rows": jacc}}
    write_csv(rq2, TAB / "rq2_online_vs_batch.csv")
    M["rq2"] = {"table": rq2.to_dict(orient="records"), **rq2_extra, "window_s": D.window_seconds(3)}
    fig_rq2(rq2, FIG / "fig07_rq2_online_vs_batch.png")

    # ---------------------------------------------------------------- RQ3 archive parameters
    log("RQ3 sampling interval experiment")
    rq3s = []
    for k in SAMPLING_FACTORS:
        offs = list(range(min(k, MAX_OFFSETS)))
        acc = {t: [] for t in "ABC"}
        dl = {t: [] for t in "ABC"}
        for off in offs:
            sub = D.add_rolling(w_old.iloc[off::k], 3 * k)
            env_k = D.fit_envelope(sub[sub["period"] == "clean"], raw_from=clean)
            fk = D.envelope_flags(sub, env_k)
            for t in "ABC":
                e = ref[ref["type"] == t].reset_index(drop=True)
                cols = [c for c in fk.columns if D.FEATURE_TYPE.get(c) == t]
                mt = D.match_episodes(e, fk[cols].any(axis=1), tol_after_s=D.window_seconds(3 * k))
                acc[t].append(mt["detected"].mean())
                dl[t].extend(mt["delay_s"].dropna().tolist())
        for t in "ABC":
            rq3s.append({"interval_s": 3 * k, "type": t, "type_name": D.TYPE_NAMES[t], "offsets": len(offs),
                         "detected_pct": 100 * float(np.mean(acc[t])), "median_delay_s": float(np.median(dl[t])) if dl[t] else np.nan,
                         "rows_per_day_per_variable": 86400 / (3 * k)})
    rq3s = pd.DataFrame(rq3s)
    write_csv(rq3s, TAB / "rq3_sampling_interval.csv")

    log("RQ3 deadband experiment")
    rq3d = []
    for d in DEADBANDS:
        rec = w_old.copy()
        stored = {}
        for sig in ["fill_weight", "compair", "power"]:
            m = D.deadband_mask(w_old[sig].to_numpy(float), d)
            stored[sig] = int(m.sum())
            rec[sig] = w_old[sig].where(m).ffill()
        rr = D.add_rolling(rec, 3)
        env_d = D.fit_envelope(rr[rr["period"] == "clean"])
        fd = D.envelope_flags(rr, env_d)
        for t, sig in [("A", "compair"), ("B", "power"), ("C", "fill_weight")]:
            e = ref[ref["type"] == t].reset_index(drop=True)
            cols = [c for c in fd.columns if D.FEATURE_TYPE.get(c) == t]
            mt = D.match_episodes(e, fd[cols].any(axis=1), tol_after_s=60)
            ev_clean = D.to_events(fd.loc[fd.index >= CD.CLEAN_START, cols].any(axis=1), EVENT_GAP_S)
            rq3d.append({"deadband": d, "signal": sig, "type": t, "stored_rows": stored[sig],
                         "stored_pct_of_cyclic_3s": 100 * stored[sig] / int(w_old[sig].notna().sum()),
                         "detected_pct": 100 * float(mt["detected"].mean()),
                         "median_delay_s": float(mt["delay_s"].median())})
    rq3d = pd.DataFrame(rq3d)
    write_csv(rq3d, TAB / "rq3_deadband.csv")

    log("RQ3 export comparison")
    glitch = {}
    for exp, w in wides.items():
        g = w["state"] == "glitch"
        days = (w["state"] != "invalid").sum() * CD.EXPORTS[exp]["cyclic_s"] / 86400
        glitch[exp] = {"samples": int(g.sum()), "episodes": len(D.to_events(g, 60)),
                       "episodes_per_day": len(D.to_events(g, 60)) / days,
                       "orders_per_day": int(w["order_id"].nunique()) / days, "valid_days": days}
    M["rq3"] = {"sampling": rq3s.to_dict(orient="records"), "deadband": rq3d.to_dict(orient="records"),
                "glitch": glitch,
                "bytes_per_day": {f"{r['export']}/{r['archive']}": r["bytes_per_day"] for r in M["inventory_files"]}}
    fig_rq3_sampling(rq3s, FIG / "fig08_rq3_sampling_interval.png")
    fig_rq3_deadband(rq3d, FIG / "fig09_rq3_deadband.png")
    fig_export_timeline(w_old, w_new, FIG / "fig01_export_timeline.png", inv)

    log("slide figures")
    slide_figures(w_old, d1["old_3s"]["flags"], rq1, rq2, rq3s, FIG / "slides")

    # ---------------------------------------------------------------- processed data
    if args.write_processed:
        log("writing processed data")
        P = Path(args.processed)
        out_old = r_old.copy()
        out_old["d1_rare"] = d1["old_3s"]["flags"].any(axis=1)
        out_old["d2_flag"] = fl_old.any(axis=1)
        out_old["d3_flag"] = det_flags["D3"]
        write_csv(out_old, P / "wide_export1_3s.csv", index=True)
        out_new = r_new.copy()
        out_new["d1_rare"] = d1["new_10s"]["flags"].any(axis=1)
        out_new["d2_flag"] = fl_new.any(axis=1)
        write_csv(out_new, P / "wide_export2_10s.csv", index=True)

    CD.atomic_write(OUT / "metrics.json", lambda tmp: Path(tmp).write_text(json.dumps(M, indent=2, default=str)))
    write_insights(M, OUT / "INSIGHTS.md")
    log("done")


def overlap_share(events: pd.DataFrame, ref: pd.DataFrame) -> float:
    """Share of alarm events whose [start, end] overlaps at least one reference episode."""
    if len(events) == 0:
        return float("nan")
    rs = ref["start"].sort_values().to_numpy()
    re_ = ref.loc[ref["start"].sort_values().index, "end"].to_numpy()
    run_max_end = np.maximum.accumulate(re_)
    idx = np.searchsorted(rs, events["end"].to_numpy(), side="right") - 1
    hit = (idx >= 0) & (run_max_end[np.clip(idx, 0, None)] >= events["start"].to_numpy())
    return float(hit.mean())


def fig_schedule(eps_old: pd.DataFrame, eps_new: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 4.6), sharex=True)
    bins = np.arange(0, CYCLE_H * 60 + 1, 15)
    for i, (eps, lab) in enumerate([(eps_old, "Export 1 (3 s)"), (eps_new, "Export 2 (10 s)")]):
        for j, t in enumerate("ABC"):
            ax = axes[i, j]
            ph = D.cycle_phase_min(eps.loc[eps["type"] == t, "start"], CYCLE_H)
            ax.hist(ph, bins=bins, color=S.TYPE_COLOR[t], rwidth=0.85)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(f"{t}: {D.TYPE_NAMES[t]}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{lab}\nepisodes")
            ax.set_xticks(range(0, CYCLE_H * 60 + 1, 60), [f"+{h}h" for h in range(CYCLE_H + 1)])
    for ax in axes[1]:
        ax.set_xlabel("onset time within the 4-hour cycle (00:00, 04:00, 08:00 ...)")
    fig.suptitle("Anomaly onsets cluster at fixed times: injected on a 4-hour schedule", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    S.save(fig, path)


# =============================================================================== RQ figures
def fig_rq1(rq1: pd.DataFrame, path: Path, size=(13, 3.4)) -> None:
    fig, axes = plt.subplots(1, 3, figsize=size)
    labels = [DET_LABEL[k] for k in rq1["detector"]]
    y = np.arange(len(rq1))
    is_ref = (rq1["detector"] == "D2").to_numpy()
    panels = [("episodes_detected_pct", "episodes detected (%)", "{:.0f} %", True),
              ("alarm_events_on_reference_pct", "alarms that hit an episode (%)", "{:.0f} %", True),
              ("median_warning_before_drift_peak_s", "median warning before a\ndrift peaks (s, types A+B)", "{:.0f} s", False)]
    for ax, (col, title, fmt, ref_panel) in zip(axes, panels):
        v = rq1[col].to_numpy(float)
        colors = [S.NORMAL if (r and ref_panel) else S.BLUE for r in is_ref]
        ax.barh(y, v, color=colors, height=0.55)
        for yi, vi, r in zip(y, v, is_ref):
            txt = fmt.format(vi) + ("  (reference)" if (r and ref_panel) else "")
            ax.text(vi, yi, " " + txt, va="center", fontsize=10, color=S.INK)
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(y))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
        if ref_panel:
            ax.set_xlim(0, 150)
            ax.set_xticks([0, 25, 50, 75, 100])
        else:
            ax.set_xlim(0, max(np.nanmax(v) * 1.45, 1))
    fig.text(0.01, -0.02, "All three: 0 false alarms in the 4.2 h clean window (one would equal 5.8 per day). "
             "Reference = D2 episodes, so D2 is 100 % by construction.", fontsize=9.5, color=S.INK2)
    fig.tight_layout()
    S.save(fig, path)


def fig_rq2(rq2: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(rq2))
    v = rq2["median_extra_delay_online_s"].to_numpy(float)
    ax.barh(y, v, color=[S.TYPE_COLOR[t] for t in rq2["type"]], height=0.55)
    for yi, vi, p in zip(y, v, rq2["online_detected_pct"]):
        ax.text(max(vi, 0), yi, f"  +{vi:.0f} s  ({p:.0f} % of episodes also found online)", va="center", fontsize=10)
    ax.set_yticks(y, [f"{t}: {n}" for t, n in zip(rq2["type"], rq2["type_name"])])
    ax.invert_yaxis()
    ax.set_xlabel("median extra detection delay online vs. batch (s)")
    ax.set_xlim(0, max(np.nanmax(v) * 2.6, 10))
    ax.grid(axis="y", visible=False)
    ax.set_title("Online (causal 60 s window) vs. batch (centred window), Export 1")
    fig.tight_layout()
    S.save(fig, path)


def fig_rq3_sampling(rq3s: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for t in "ABC":
        d = rq3s[rq3s["type"] == t]
        for ax, col in zip(axes, ["detected_pct", "median_delay_s"]):
            ax.plot(d["interval_s"], d[col], marker="o", ms=5, color=S.TYPE_COLOR[t], lw=2)
            ax.text(d["interval_s"].iloc[-1] * 1.04, d[col].iloc[-1], f"{t}", color=S.INK, va="center", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([3, 6, 9, 15, 30, 60], ["3", "6", "9", "15", "30", "60"])
        ax.minorticks_off()
        ax.set_xlabel("cyclic sampling interval (s), thinned from the 3 s export")
    axes[0].set_ylabel("episodes still detected (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("median detection delay (s)")
    axes[0].set_title("Detection vs. sampling interval")
    axes[1].set_title("Delay vs. sampling interval")
    fig.legend(handles=[Line2D([], [], color=S.TYPE_COLOR[t], marker="o", label=f"{t}: {D.TYPE_NAMES[t]}") for t in "ABC"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    S.save(fig, path)


def fig_rq3_deadband(rq3d: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for t, sig in [("A", "compair"), ("B", "power"), ("C", "fill_weight")]:
        d = rq3d[rq3d["signal"] == sig]
        axes[0].plot(d["deadband"], d["stored_pct_of_cyclic_3s"], marker="o", ms=5, color=S.TYPE_COLOR[t])
        axes[1].plot(d["deadband"], d["detected_pct"], marker="o", ms=5, color=S.TYPE_COLOR[t])
    axes[0].set_ylabel("rows stored (% of cyclic 3 s)")
    axes[1].set_ylabel("episodes still detected (%)")
    for ax in axes:
        ax.set_xlabel("hysteresis / deadband (signal units)")
        ax.set_ylim(0, 105)
    axes[0].set_title("Storage with 'record on change'")
    axes[1].set_title("Detection after reconstruction")
    fig.legend(handles=[Line2D([], [], color=S.TYPE_COLOR[t], marker="o", label=f"{SIG_LABEL[s]} ({t})")
                        for t, s in [("A", "compair"), ("B", "power"), ("C", "fill_weight")]],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    S.save(fig, path)


def fig_export_timeline(w_old, w_new, path: Path, inv: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 1.9))
    for i, (w, lab, col) in enumerate([(w_old, "Export 1: cyclic 3 s", S.BLUE), (w_new, "Export 2: cyclic 10 s", S.ORANGE)]):
        ax.barh(i, (w.index.max() - w.index.min()).total_seconds() / 86400, left=mdates.date2num(w.index.min()),
                height=0.5, color=col, alpha=0.85)
        ax.text(w.index.min(), i - 0.36, lab, fontsize=10, va="bottom", color=S.INK)
    ax.barh(0, (CD.CLEAN_END - CD.CLEAN_START).total_seconds() / 86400, left=mdates.date2num(CD.CLEAN_START),
            height=0.5, color=S.GOOD)
    inv_ts = w_new.index[w_new["state"] == "invalid"]
    ax.barh(1, (inv_ts.max() - inv_ts.min()).total_seconds() / 86400, left=mdates.date2num(inv_ts.min()),
            height=0.5, color=S.MUTED)
    ax.xaxis_date()
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(1.6, -0.8)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.legend(handles=[Patch(color=S.GOOD, label="clean window (COPA-DATA)"), Patch(color=S.MUTED, label="INVALID rows")],
               loc="upper right", ncol=2)
    fig.tight_layout()
    S.save(fig, path)


def slide_figures(w_old, flags_d1, rq1, rq2, rq3s, out: Path) -> None:
    """Figures sized for a 16:9 slide (about 7.4 x 4.6 in), fonts readable without scaling."""
    import matplotlib as mpl
    with mpl.rc_context({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 12.5,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5}):
        # RQ1: the frequency idea -> where the rare values lie
        fig = plt.figure(figsize=(7.4, 4.75))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.65)
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        t = D.frequency_table(w_old.loc[w_old["state"] == "running", "fill_weight"], TAU)
        ax1.bar(t["rank"], t["share_pct"], width=1.0, color=np.where(t["rare"], S.CRITICAL, S.NORMAL), linewidth=0)
        ax1.set_yscale("log")
        ax1.yaxis.set_major_formatter(S.pct_formatter())
        ax1.yaxis.set_minor_formatter(plt.NullFormatter())
        ax1.axhline(100 * TAU, color=S.INK2, lw=0.8)
        ax1.text(t["rank"].max(), 100 * TAU * 1.3, "1 in 10,000 rows", ha="right", va="bottom", fontsize=11, color=S.INK2)
        ax1.set_title("1  Rank every fill-weight value by how often it occurs", fontsize=13.5)
        ax1.set_xlabel("values ranked by frequency (1 = most frequent)")
        ax1.set_ylabel("share of rows")
        ax1.grid(axis="x", visible=False)
        run = w_old["state"] == "running"
        ax2.scatter(w_old.index[run.to_numpy(bool)], w_old.loc[run, "fill_weight"], s=0.5, color=S.NORMAL, linewidths=0, rasterized=True)
        f = flags_d1["fill_weight"].reindex(w_old.index, fill_value=False)
        ax2.scatter(w_old.index[f.to_numpy(bool)], w_old.loc[f, "fill_weight"], s=10, color=S.CRITICAL, linewidths=0, zorder=3)
        ax2.axvspan(CD.CLEAN_START, CD.CLEAN_END, color=S.GOOD, alpha=0.18, lw=0)
        n_a = int((f & (w_old["period"] == "anomaly")).sum())
        n_c = int((f & (w_old["period"] == "clean")).sum())
        ax2.set_title(f"2  See where they lie: {n_a} rare rows in the anomaly period, {n_c} in the clean window (green)", fontsize=13.5)
        ax2.set_ylim(478, 522)
        ax2.set_ylabel("fill weight")
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        S.save(fig, out / "slide_rq1_frequency_idea.png")

        # RQ2: online vs batch
        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        yy = np.arange(len(rq2))
        v = rq2["median_extra_delay_online_s"].to_numpy(float)
        ax.barh(yy, v, color=[S.TYPE_COLOR[x] for x in rq2["type"]], height=0.5)
        for yi, vi, pc in zip(yy, v, rq2["online_detected_pct"]):
            ax.text(max(vi, 0), yi, f"  +{vi:.0f} s  |  {pc:.0f} % also found online", va="center", fontsize=12)
        ax.set_yticks(yy, [f"{x}: {n}" for x, n in zip(rq2["type"], rq2["type_name"])])
        ax.invert_yaxis()
        ax.set_xlim(0, max(np.nanmax(v) * 3.2, 10))
        ax.set_xlabel("median extra delay of the online detector (s)")
        ax.grid(axis="y", visible=False)
        ax.set_title("Replay of Export 1: causal 60 s window (online) vs. centred window (batch)", fontsize=13)
        S.save(fig, out / "slide_rq2_online_vs_batch.png")

        # RQ3: sampling interval
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.75), sharex=True)
        for tt in "ABC":
            d = rq3s[rq3s["type"] == tt]
            axes[0].plot(d["interval_s"], d["detected_pct"], marker="o", ms=6, color=S.TYPE_COLOR[tt], lw=2.2)
            axes[1].plot(d["interval_s"], d["median_delay_s"], marker="o", ms=6, color=S.TYPE_COLOR[tt], lw=2.2)
            axes[0].text(63, d["detected_pct"].iloc[-1], f"{tt}", va="center", fontsize=12, color=S.INK)
            axes[1].text(63, d["median_delay_s"].iloc[-1], f"{tt}", va="center", fontsize=12, color=S.INK)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("detected (%)")
        axes[1].set_ylabel("median delay (s)")
        axes[1].set_xscale("log")
        axes[1].set_xticks([3, 6, 9, 15, 30, 60], ["3", "6", "9", "15", "30", "60"])
        axes[1].minorticks_off()
        axes[1].set_xlabel("cyclic sampling interval (s), thinned from the real 3 s export")
        axes[0].set_title("Episodes found at 3 s that are still found at coarser sampling", fontsize=13)
        fig.legend(handles=[Line2D([], [], color=S.TYPE_COLOR[x], marker="o", label=f"{x}: {D.TYPE_NAMES[x]}") for x in "ABC"],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=11, handlelength=1.5, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        S.save(fig, out / "slide_rq3_sampling.png")


# =============================================================================== insights
def write_insights(M: dict, path: Path) -> None:
    inv, ch, rq1 = M["inventory"], M["characterisation"], pd.DataFrame(M["rq1"]["table"]).set_index("detector")
    d2 = M["d2_episodes"]
    fr = M["frequency_ranking_old_running"]
    s = pd.DataFrame(M["rq3"]["sampling"])
    dbd = pd.DataFrame(M["rq3"]["deadband"])
    g = M["rq3"]["glitch"]
    rq2 = pd.DataFrame(M["rq2"]["table"]).set_index("type")
    L = []
    a = L.append
    a("# Insights - COPA-DATA filler M7 (generated by analysis/run_analysis.py)\n")
    a("All numbers below are computed from `data/txt/*.TXT`; re-run the script to reproduce them.\n")
    a("## Data")
    for exp, i in inv.items():
        a(f"- **{CD.EXPORTS[exp]['label']}**: {i['start']} -> {i['end']} ({i['days']:.2f} days), "
          f"{i['cyclic_rows_per_var']:,} cyclic samples per variable, {i['invalid_timestamps']} INVALID timestamps, {i['orders']} orders (incl. partial first/last).")
    a("- The two exports do not overlap: Export 2 starts where Export 1 ends. Only Export 1 has a label from COPA-DATA "
      f"(no anomalies {M['parameters']['clean_window'][0]} - {M['parameters']['clean_window'][1]}).")
    a("\n## What the anomalies are (Export 1, anomaly period vs. clean window)")
    ef = ch["equipment_failure_per_h"]
    a(f"- Machine stops are normal operation: 'Equipment Failure' occurs {ef['anomaly_period']:.2f}/h in the anomaly period and {ef['clean_window']:.2f}/h in the clean window.")
    r = ch["ranges"]
    a(f"- Compressed air (running): clean max {r['running/compair/clean']['max']:.1f}, anomaly-period max {r['running/compair/anomaly']['max']:.1f}; "
      f"compressed air (stopped): clean max {r['stopped/compair/clean']['max']:.1f}, anomaly-period max {r['stopped/compair/anomaly']['max']:.1f}.")
    a(f"- Power (running): clean max {r['running/power/clean']['max']:.1f}, anomaly-period max {r['running/power/anomaly']['max']:.1f}.")
    a(f"- Fill weight (running): clean range {r['running/fill_weight/clean']['min']:.1f}-{r['running/fill_weight/clean']['max']:.1f}, "
      f"anomaly-period range {r['running/fill_weight/anomaly']['min']:.1f}-{r['running/fill_weight/anomaly']['max']:.1f}.")
    a(f"- Speed glitch (target 550 / speed 1100 at order end): {ch['glitch_samples_anomaly_period']} samples in the anomaly period, "
      f"{ch['glitch_samples_clean_window']} in the clean window -> treated as a transition artefact, not scored.")
    dp = ch["defect_pct"]
    a(f"- Defect rate per order: clean-window orders {dp['old_clean_orders']} %; complete anomaly-period orders median {dp['old_anomaly_orders']['50%']:.2f} % "
      f"(min {dp['old_anomaly_orders']['min']:.2f}, max {dp['old_anomaly_orders']['max']:.2f}).")
    a("\n## Frequency ranking (D1, the 'descending list' idea)")
    for sig, f in fr.items():
        a(f"- {SIG_LABEL[sig]} (running, Export 1): {f['distinct_values']} distinct values; most frequent {f['top_value']:.1f} = {f['top_share_pct']:.2f} % of rows; "
          f"{f['rare_values']} rare values (< 1 in 10,000 rows) = {f['rare_rows']} rows.")
    d1 = M["d1_rare_events"]
    a(f"- Rare-value events (rows merged within {EVENT_GAP_S} s): {d1['old_anomaly_period']} in the anomaly period, {d1['old_clean_window']} in the clean window "
      f"({d1['old_anomaly_per_day']:.1f}/day). Export 2: {d1['new_total']} events ({d1['new_per_day']:.1f}/day).")
    a("\n## Anomaly episodes (D2 clean envelope, merged within 5 min per type)")
    for t in "ABC":
        a(f"- {t} {D.TYPE_NAMES[t]}: {d2['old_anomaly_period'][t]} in Export 1 anomaly period ({d2['per_day']['old_3s'][t]:.1f}/day), "
          f"{d2['new'][t]} in Export 2 ({d2['per_day']['new_10s'][t]:.1f}/day).")
    a(f"- Export 2 contains the same three anomaly types, so it is not anomaly-free; COPA-DATA gave no label for it.")
    a(f"- Envelope = clean min-max widened by {M['parameters']['envelope_margin_sd']} clean standard deviation(s); "
      "the plain min-max of the 4 h window produced false alarms in a 2-fold out-of-sample test.")
    a("\n## Schedule: anomaly onsets repeat in a 4-hour clock cycle")
    for r_ in M["schedule"]:
        a(f"- {CD.EXPORTS[r_['export']]['label']}, {r_['type']} {r_['type_name']}: {r_['best_30min_share_pct']:.0f} % of {r_['episodes']} onsets fall in the "
          f"30-min window {r_['best_window']} of the cycle (uniform would be {r_['uniform_expectation_pct']:.1f} %).")
    a("- Random noise would not cluster like this, so the episodes are injected anomalies. The phase differs by about 2.5 h "
      "between the exports; whether this comes from the restart after the INVALID outage (04.11.2026 15:00-15:45) is an open question for COPA-DATA.")
    a("\n## RQ1 - detectors (Export 1)")
    a("| detector | flagged rows | alarm events/h (anomaly period) | false alarms/day (clean window) | episodes detected | alarm events on an episode | median delay | median warning before drift peak |")
    a("|---|---|---|---|---|---|---|---|")
    for k, row in rq1.iterrows():
        a(f"| {row['label']} | {row['flagged_pct']:.2f} % | {row['events_per_h_anomaly_period']:.2f} | {row['false_alarms_per_day']:.1f} | "
          f"{row['episodes_detected_pct']:.0f} % | {row['alarm_events_on_reference_pct']:.0f} % | {row['median_delay_s']:.0f} s | {row['median_warning_before_drift_peak_s']:.0f} s |")
    a(f"\nClean window = {ch['clean_hours']:.2f} h, so one false alarm there = {24 / ch['clean_hours']:.1f} false alarms/day. "
      "D2 false alarms are out-of-sample (2-fold split of the clean window); D2 episodes are the reference, so its detection rate is 100 % by construction.")
    a("\n## RQ2 - online (causal) vs. batch (centred) on Export 1")
    for t, row in rq2.iterrows():
        a(f"- {t} {row['type_name']}: {row['online_detected_pct']:.0f} % of batch episodes also found online; median extra delay {row['median_extra_delay_online_s']:.0f} s.")
    tr = M["rq2"]["d1_transfer"]
    a(f"- D1 model learnt on Export 1 (3 s) and streamed on Export 2 (10 s): {tr['online_events']} events vs. {tr['batch_events']} when learnt on Export 2 itself "
      f"(row overlap Jaccard {tr['jaccard_rows']:.2f}).")
    a("\n## RQ3 - archive parameters")
    for t in "ABC":
        x = s[s["type"] == t].set_index("interval_s")
        a(f"- {t} {D.TYPE_NAMES[t]}: detected {x.at[3, 'detected_pct']:.0f} % at 3 s, {x.at[9, 'detected_pct']:.0f} % at 9 s, "
          f"{x.at[30, 'detected_pct']:.0f} % at 30 s, {x.at[60, 'detected_pct']:.0f} % at 60 s (median delay {x.at[3, 'median_delay_s']:.0f} s -> {x.at[60, 'median_delay_s']:.0f} s).")
    a(f"- Speed glitch episodes per day: {g['old_3s']['episodes_per_day']:.2f} at 3 s vs. {g['new_10s']['episodes_per_day']:.2f} at 10 s, "
      f"with {g['old_3s']['orders_per_day']:.1f} vs. {g['new_10s']['orders_per_day']:.1f} orders/day: short events are lost with coarser sampling.")
    for sig in ["fill_weight", "compair", "power"]:
        x = dbd[dbd["signal"] == sig].set_index("deadband")
        a(f"- {SIG_LABEL[sig]} record-on-change: deadband 0 stores {x.at[0.0, 'stored_pct_of_cyclic_3s']:.0f} % of the cyclic rows, "
          f"deadband 0.5 stores {x.at[0.5, 'stored_pct_of_cyclic_3s']:.0f} % ({x.at[0.5, 'detected_pct']:.0f} % episodes detected), "
          f"deadband 2.0 stores {x.at[2.0, 'stored_pct_of_cyclic_3s']:.0f} % ({x.at[2.0, 'detected_pct']:.0f} % detected).")
    CD.atomic_write(path, lambda tmp: Path(tmp).write_text("\n".join(L) + "\n"))


if __name__ == "__main__":
    main()
