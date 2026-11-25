"""
copa_data.py - load the COPA-DATA zenon exports of filler M7. All columns keep their real zenon names.

Data source (nothing else is ever substituted):
    data/txt/*.TXT   raw zenon archive exports delivered by COPA-DATA (Git LFS).
                     If the files are only LFS pointers, run `git lfs pull` first.

Facts this module relies on, with their source:
    * Columns: Date; Time; Variable name; value; Status            -> Andreas Gruen, mail 13.11.2026
    * Ignore every row whose status contains INVALID               -> Andreas Gruen, mail 13.11.2026
    * Export of 04.11.2026: two cyclic archives (3 s) + one archive recording on change -> mail 13.11.2026
    * Export of 20.11.2026: same archives, cyclic sampling time 10 s                    -> mail 20.11.2026
    * "plenty anomalies ... between October 29 and November 4th ~10:50; after that, so Nov 4 ~10:50 - 15:00,
       no anomalies occurred anymore"                              -> Andreas Gruen, mail 13.11.2026
      This statement covers the export of 04.11.2026 (it ends 04.11.2026 15:00). The export of 20.11.2026 starts
      04.11.2026 15:00; Andreas made no statement about it.
    * Variable meanings: README.md "Variable Descriptions" (copied from Andreas' mail)
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

# ----------------------------------------------------------------------------- real variable names
M7_FillWeight = "Sim/Global/M7_FillWeight"
M7_CompAir = "Sim/Global/M7_CompAir"
M7_Power = "Sim/Global/M7_Power"
CurMachSpeed = "Sim/Global/M7.Status.CurMachSpeed"
MachSpeed = "Sim/Global/M7.Status.MachSpeed"
StateCurrent = "Sim/Global/M7.Status.StateCurrent"
ProcessedCount = "Sim/Global/M7.Admin.ProductData_ProcessedCount"
DefectiveCount = "Sim/Global/M7.Admin.ProductData_DefectiveCount"
WS_Cur_State = "M07.WS_Cur_State"
WS_Cur_Prog = "M07.WS_Cur_Prog"
WS_Cur_Mode = "M07.WS_Cur_Mode"

# ----------------------------------------------------------------------------- exports (file names from COPA-DATA)
EXPORTS = {
    "export_2026-11-04": {
        "label": "Export 04.11.2026 (cyclic 3 s)",
        "files": {"FP": "FP261104141238_FP.TXT", "17": "17261104141251_17.TXT", "07": "07261104141229_07.TXT"},
        "cyclic_s": 3,
    },
    "export_2026-11-20": {
        "label": "Export 20.11.2026 (cyclic 10 s)",
        "files": {"FP": "FP261120103526_FP.TXT", "17": "17261120103538_17.TXT", "07": "07261120103434_07.TXT"},
        "cyclic_s": 10,
    },
}
LABELLED = "export_2026-11-04"   # the only export Andreas' anomaly statement covers

CLEAN_START = pd.Timestamp("2026-11-04 10:50:00")   # Andreas: no anomalies from ~10:50 ...
CLEAN_END = pd.Timestamp("2026-11-04 15:00:00")     # ... until the end of the export (15:00)

# ----------------------------------------------------------------------------- derived columns (defined from real values)
MACHINE = "machine (from CurMachSpeed)"
RUNNING = "running: CurMachSpeed 990 or 1089"
STOPPED = "stopped: CurMachSpeed 0"
ORDER_END = "order end: CurMachSpeed 1100 with MachSpeed 550"
INVALID = "no CurMachSpeed value (INVALID row or no sample in archive FP)"
RUN_ID = "run_id"                  # as in Leon's master.csv: +1 at every reset of ProcessedCount
LABEL = "Andreas label"
LBL_ANOM = "anomalies (until 04.11. ~10:50)"
LBL_CLEAN = "no anomalies (04.11. 10:50-15:00)"
LBL_NONE = "no statement (export 20.11.2026)"

PACKML_STATES = {1: "Clearing", 2: "Stopped", 3: "Starting", 4: "Idle", 5: "Suspended", 6: "Execute",
                 7: "Stopping", 8: "Aborting", 9: "Aborted", 10: "Holding", 11: "Held", 12: "Unholding",
                 13: "Suspending", 14: "Unsuspending", 15: "Resetting", 16: "Completing", 17: "Complete"}
WS_STATES = {0: "Comm. Error", 1: "Stopped", 2: "Starting", 4: "Prepared", 8: "Lack", 16: "Tailback",
             32: "Lack Branch Line", 64: "Tailback Branch Line", 128: "Operating", 256: "Stopping",
             512: "Aborting", 1024: "Equipment Failure", 2048: "External Failure", 4096: "Emergency Stop",
             8192: "Holding", 16384: "Held", 32768: "Idle", 65536: "Unholding", 131072: "Suspending",
             262144: "Unsuspending", 524288: "Resetting", 1048576: "Clearing"}


class LfsPointerError(RuntimeError):
    pass


def check_real_file(path: Path) -> None:
    """Refuse to work on Git LFS pointer files (the real data would silently be missing)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `git lfs pull` in the repo.")
    if path.stat().st_size < 1024 and path.read_bytes()[:64].startswith(b"version https://git-lfs"):
        raise LfsPointerError(f"{path.name} is a Git LFS pointer, not data. Run `git lfs pull`.")


def read_txt(path: Path) -> pd.DataFrame:
    """One raw zenon export -> rows with Andreas' column names plus a timestamp."""
    check_real_file(path)
    df = pd.read_csv(path, sep=";", header=None, names=["Date", "Time", "Variable name", "value", "Status"],
                     dtype={"Date": str, "Time": str, "Variable name": str, "value": float, "Status": str})
    df["Status"] = df["Status"].str.strip()
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d.%m.%Y %H:%M:%S.%f")
    df["INVALID"] = df["Status"].str.contains("INVALID", regex=False)
    df["file"] = path.name
    return df


def load_raw(export: str) -> dict[str, pd.DataFrame]:
    return {arch: read_txt(TXT_DIR / name) for arch, name in EXPORTS[export]["files"].items()}


def build_wide(raw: dict[str, pd.DataFrame], export: str) -> pd.DataFrame:
    """One row per timestamp of the cyclic archives (FP + 17); the on-change archive (07) carried forward.
    Columns are the real variable names. INVALID rows are dropped (their timestamps stay, values NaN)."""
    cyc = pd.concat([raw["FP"], raw["17"]], ignore_index=True)
    grid = pd.DatetimeIndex(np.sort(cyc["timestamp"].unique()), name="timestamp")
    valid = cyc[~cyc["INVALID"]]
    wide = valid.pivot_table(index="timestamp", columns="Variable name", values="value", aggfunc="last").reindex(grid)
    wide.columns.name = None
    ev = raw["07"][~raw["07"]["INVALID"]]
    ev = ev.pivot_table(index="timestamp", columns="Variable name", values="value", aggfunc="last").sort_index().ffill()
    ev.columns.name = None
    wide = pd.merge_asof(wide.reset_index(), ev.reset_index(), on="timestamp", direction="backward").set_index("timestamp")

    spd, tgt = wide[CurMachSpeed], wide[MachSpeed]
    wide[MACHINE] = np.select([spd.isna(), (spd == 1100) & (tgt == 550), spd == 0, spd.isin([990, 1089])],
                              [INVALID, ORDER_END, STOPPED, RUNNING], default="other CurMachSpeed value")
    wide[RUN_ID] = (wide[ProcessedCount].ffill().diff() < 0).cumsum()
    if export == LABELLED:
        wide[LABEL] = np.where(wide.index >= CLEAN_START, LBL_CLEAN, LBL_ANOM)
    else:
        wide[LABEL] = LBL_NONE
    wide.attrs["export"] = export
    wide.attrs["step_s"] = EXPORTS[export]["cyclic_s"]
    return wide


def load_export(export: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw = load_raw(export)
    return build_wide(raw, export), raw


def safe_name(variable: str) -> str:
    """Variable name usable in a file name: 'Sim/Global/M7_CompAir' -> 'Sim_Global_M7_CompAir'."""
    return variable.replace("/", "_").replace(" ", "_")


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
