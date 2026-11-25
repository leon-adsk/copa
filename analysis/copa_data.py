"""
copa_data.py - load the COPA-DATA zenon exports of filler M7 into analysis-ready tables.

Data source (nothing else is ever substituted):
    data/txt/*.TXT   raw zenon archive exports delivered by COPA-DATA (Git LFS).
                     If the files are only LFS pointers, run `git lfs pull` first.

Facts this module relies on, with their source:
    * Columns: Date; Time; Variable name; value; Status            -> Andreas Gruen, mail 13.11.2026
    * Ignore every row whose status contains INVALID               -> Andreas Gruen, mail 13.11.2026
    * Export 1: two cyclic archives (3 s) + one on-change archive  -> Andreas Gruen, mail 13.11.2026
    * Export 2: same archives, cyclic sampling time 10 s           -> Andreas Gruen, mail 20.11.2026
    * "plenty anomalies ... between October 29 and November 4th ~10:50; after that,
       so Nov 4 ~10:50 - 15:00, no anomalies occurred anymore"     -> Andreas Gruen, mail 13.11.2026
      This label covers export 1 only. Export 2 starts 04.11.2026 15:00 and has no label.
    * Sim/Global/M7.Status.StateCurrent uses PackML state numbers  -> Leon, notebooks/master_wide_construction.py
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TXT_DIR = REPO / "data" / "txt"

EXPORTS = {
    "old_3s": {
        "label": "Export 1 (cyclic 3 s)",
        "files": {"FP": "FP261104141238_FP.TXT", "17": "17261104141251_17.TXT", "07": "07261104141229_07.TXT"},
        "cyclic_s": 3,
    },
    "new_10s": {
        "label": "Export 2 (cyclic 10 s)",
        "files": {"FP": "FP261120103526_FP.TXT", "17": "17261120103538_17.TXT", "07": "07261120103434_07.TXT"},
        "cyclic_s": 10,
    },
}

# Andreas Gruen, mail 13.11.2026 (export 1 only): no anomalies from ~10:50 until the export ends at 15:00.
CLEAN_START = pd.Timestamp("2026-11-04 10:50:00")
CLEAN_END = pd.Timestamp("2026-11-04 15:00:00")

# zenon variable name -> short column name (meanings: Andreas Gruen, mail 13.11.2026)
TAGS = {
    "Sim/Global/M7_FillWeight": "fill_weight",                        # fill weight of the current bottle
    "Sim/Global/M7_CompAir": "compair",                               # current compressed-air demand
    "Sim/Global/M7_Power": "power",                                   # current power demand
    "Sim/Global/M7.Status.CurMachSpeed": "speed",                     # current machine speed
    "Sim/Global/M7.Status.MachSpeed": "speed_target",                 # target / nominal speed
    "Sim/Global/M7.Status.StateCurrent": "packml_state",              # machine state (PackML numbers)
    "Sim/Global/M7.Status.UnitModeCurrent": "unit_mode",
    "Sim/Global/M7.Admin.ProductData_ProcessedCount": "processed",    # bottles filled in current order
    "Sim/Global/M7.Admin.ProductData_DefectiveCount": "defective",    # rejected bottles in current order
    "Sim/Global/M7_CompAir_incremental": "compair_cum",
    "Sim/Global/M7_Energy_incremental": "energy_cum",
    "Converter/Global/M07.Power": "ws_power",                         # copy of power (other logic)
    "M07.WS_Cur_Mach_Spd": "ws_speed",                                # copy of speed
    "M07.WS_Good_packages": "ws_good",
    "M07.WS_Tot_Packages": "ws_total",
    "M07.WS_Cur_State": "ws_state",                                   # Weihenstephan state (on change)
    "M07.WS_Cur_Prog": "ws_prog",                                     # Weihenstephan program (on change)
    "M07.WS_Cur_Mode": "ws_mode",                                     # Weihenstephan mode (on change)
}

PACKML_STATES = {1: "Clearing", 2: "Stopped", 3: "Starting", 4: "Idle", 5: "Suspended", 6: "Execute",
                 7: "Stopping", 8: "Aborting", 9: "Aborted", 10: "Holding", 11: "Held", 12: "Unholding",
                 13: "Suspending", 14: "Unsuspending", 15: "Resetting", 16: "Completing", 17: "Complete"}
WS_STATES = {0: "Comm. Error", 1: "Stopped", 2: "Starting", 4: "Prepared", 8: "Lack", 16: "Tailback",
             32: "Lack Branch Line", 64: "Tailback Branch Line", 128: "Operating", 256: "Stopping",
             512: "Aborting", 1024: "Equipment Failure", 2048: "External Failure", 4096: "Emergency Stop",
             8192: "Holding", 16384: "Held", 32768: "Idle", 65536: "Unholding", 131072: "Suspending",
             262144: "Unsuspending", 524288: "Resetting", 1048576: "Clearing"}
WS_PROGS = {1: "Production", 2: "Start Up", 4: "Run Down", 8: "Cleaning", 16: "Changeover",
            32: "Maintenance", 64: "Break"}


class LfsPointerError(RuntimeError):
    pass


def check_real_file(path: Path) -> None:
    """Refuse to work on Git LFS pointer files (the real data would silently be missing)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `git lfs pull` in the repo.")
    if path.stat().st_size < 1024:
        head = path.read_bytes()[:64]
        if head.startswith(b"version https://git-lfs"):
            raise LfsPointerError(f"{path.name} is a Git LFS pointer, not data. Run `git lfs pull`.")


def read_txt(path: Path) -> pd.DataFrame:
    """One raw zenon export -> long table [ts, var, value, status, invalid]."""
    check_real_file(path)
    df = pd.read_csv(path, sep=";", header=None, names=["date", "time", "var", "value", "status"],
                     dtype={"date": str, "time": str, "var": str, "value": float, "status": str})
    df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d.%m.%Y %H:%M:%S.%f")
    df["status"] = df["status"].str.strip()
    df["invalid"] = df["status"].str.contains("INVALID", regex=False)
    return df[["ts", "var", "value", "status", "invalid"]]


def load_raw(export: str) -> dict[str, pd.DataFrame]:
    return {arch: read_txt(TXT_DIR / name) for arch, name in EXPORTS[export]["files"].items()}


def build_wide(raw: dict[str, pd.DataFrame], export: str) -> pd.DataFrame:
    """Cyclic archives FP + 17 pivoted on their common time grid; on-change archive 07 joined as-of.

    INVALID rows are dropped, so their timestamps keep NaN values (state 'invalid')."""
    cyc = pd.concat([raw["FP"], raw["17"]], ignore_index=True)
    grid = pd.DatetimeIndex(np.sort(cyc["ts"].unique()), name="ts")
    valid = cyc[~cyc["invalid"]]
    wide = (valid.pivot_table(index="ts", columns="var", values="value", aggfunc="last")
            .reindex(grid).rename(columns=TAGS))
    wide.columns.name = None

    ev = raw["07"][~raw["07"]["invalid"]]
    ev = (ev.pivot_table(index="ts", columns="var", values="value", aggfunc="last")
          .rename(columns=TAGS).sort_index().ffill())
    ev.columns.name = None
    wide = pd.merge_asof(wide.reset_index(), ev.reset_index(), on="ts", direction="backward").set_index("ts")

    # Operating state used for all detectors (derived from speed; see analysis/README.md):
    #   running = speed 990 (Execute, target 1000) or 1089 (Completing, target 1100)
    #   stopped = speed 0;  glitch = speed 1100 with target 550 (one-sample transition at order end)
    spd, tgt = wide["speed"], wide["speed_target"]
    wide["state"] = np.select(
        [spd.isna(), (spd == 1100) & (tgt == 550), spd == 0, spd > 0],
        ["invalid", "glitch", "stopped", "running"], default="other")
    wide["order_id"] = (wide["processed"].ffill().diff() < 0).cumsum()
    if export == "old_3s":
        wide["period"] = np.where(wide.index >= CLEAN_START, "clean", "anomaly")
    else:
        wide["period"] = "unlabelled"
    wide.attrs["export"] = export
    wide.attrs["step_s"] = EXPORTS[export]["cyclic_s"]
    return wide


def load_export(export: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw = load_raw(export)
    return build_wide(raw, export), raw


def atomic_write(path: Path, write_fn) -> None:
    """Write to a local temp file first, then move into place (OneDrive folders can time out on large writes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=path.suffix)
    os.close(fd)
    try:
        write_fn(tmp)
        shutil.move(tmp, path)
        os.chmod(path, 0o644)           # mkstemp creates 0600 files
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
