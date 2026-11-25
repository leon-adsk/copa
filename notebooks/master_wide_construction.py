import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import re

    from plotly import express as px
    from pathlib import Path
    from matplotlib import pyplot as plt
    import plotly.graph_objects as go

    return Path, pd, px


@app.cell
def _(Path, pd):
    master_df = pd.read_csv(Path("./data/master.csv"), parse_dates=["datetime"])
    return (master_df,)


@app.cell
def _(master_df):
    l = master_df[master_df["var"] == "Sim/Global/M7.Status.StateCurrent"]["value"].unique()
    l.sort()
    l # statuscodes in email are wrong, this looks like packml standard
    return


@app.cell
def _(master_df, pd):
    PACKML_STATES = {
        1: "Clearing",
        2: "Stopped",
        3: "Starting",
        4: "Idle",
        5: "Suspended",
        6: "Execute",
        7: "Stopping",
        8: "Aborting",
        9: "Aborted",
        10: "Holding",
        11: "Held",
        12: "Unholding",
        13: "Suspending",
        14: "Unsuspending",
        15: "Resetting",
        16: "Completing",
        17: "Complete",
    }

    WS_STATES = {
        0: "Comm. Error",
        1: "Stopped",
        2: "Starting",
        4: "Prepared",
        8: "Lack",
        16: "Tailback",
        32: "Lack Branch Line",
        64: "Tailback Branch Line",
        128: "Operating",
        256: "Stopping",
        512: "Aborting",
        1024: "Equipment Failure",
        2048: "External Failure",
        4096: "Emergency Stop",
        8192: "Holding",
        16384: "Held",
        32768: "Idle",
        65536: "Unholding",
        131072: "Suspending",
        262144: "Unsuspending",
        524288: "Resetting",
        1048576: "Clearing",
    }

    def _decode_ws(val):
        num = pd.to_numeric(val, errors="coerce")
        if pd.isna(num):
            return str(val)
        code = int(num)
        if code in WS_STATES:
            return WS_STATES[code]
        active = [name for bit, name in WS_STATES.items() if bit > 0 and (code & bit)]
        return " | ".join(active) if active else str(code)

    df = master_df.copy()
    df["state_name"] = pd.NA

    packml_mask = df["var"] == "Sim/Global/M7.Status.StateCurrent"
    packml_codes = pd.to_numeric(df.loc[packml_mask, "value"], errors="coerce")
    df.loc[packml_mask, "state_name"] = packml_codes.map(PACKML_STATES).fillna(
        df.loc[packml_mask, "value"].astype(str)
    )

    ws_mask = df["var"] == "M07.WS_Cur_State"
    df.loc[ws_mask, "state_name"] = df.loc[ws_mask, "value"].apply(_decode_ws)

    df["status"] = df["status"].astype(str)
    df
    return (df,)


@app.cell
def _(df, pd):
    cyclic = df[df["archive"] != "07"].copy()
    cyclic["datetime"] = cyclic["datetime"].dt.floor("2s")

    wide = (
        cyclic.pivot_table(
            index=["run_id", "datetime"],
            columns="var",
            values="value",
            aggfunc="last",
        )
        .groupby(level="run_id")
        .ffill(limit=1)
        .reset_index()
        .sort_values("datetime")
    )

    events = (
        df[df["archive"] == "07"]
        .pivot_table(index="datetime", columns="var", values="value", aggfunc="last")
        .sort_index()
        .ffill()
        .reset_index()
    )

    master_wide = pd.merge_asof(wide, events, on="datetime", direction="backward")
    master_wide.columns.name = None
    master_wide
    return (master_wide,)


@app.cell
def _(master_wide):
    master_wide.to_csv("./data/master_wide.csv", index=False)
    return


@app.cell
def _(master_wide):
    print(master_wide.columns.tolist())
    print(master_wide.index.names, type(master_wide.columns).__name__)
    print([repr(c) for c in master_wide.columns if 'datetime' in str(c)])

    return


@app.cell
def _():
    return


@app.cell
def _(master_df, px):
    _runs = (
        master_df.groupby(["machine", "run_id"], as_index=False)
        .agg(start=("datetime", "min"), end=("datetime", "max"))
    )
    _runs["duration_min"] = ((_runs["end"] - _runs["start"]).dt.total_seconds() / 60).round(1)

    _pkgs = (
        master_df[master_df["var"] == "Sim/Global/M7.Admin.ProductData_ProcessedCount"]
        .groupby("run_id")["value"]
        .max()
        .rename("processed_pkgs")
    )
    _runs = _runs.merge(_pkgs, on="run_id", how="left")
    _runs["run_label"] = "Run " + _runs["run_id"].astype(str)

    fig = px.timeline(
        _runs,
        x_start="start",
        x_end="end",
        y="machine",
        color="processed_pkgs",
        hover_name="run_label",
        hover_data={
            "machine": False,
            "start": "|%Y-%m-%d %H:%M:%S",
            "end": "|%Y-%m-%d %H:%M:%S",
            "duration_min": True,
            "processed_pkgs": ":,.0f",
        },
        color_continuous_scale="Viridis",
        title="Production Runs Timeline (M07)",
        labels={
            "processed_pkgs": "Processed Packages",
            "duration_min": "Duration (min)",
            "machine": "Machine",
        },
    )
    fig.update_traces(marker_line_color="white", marker_line_width=0.6)
    fig.update_layout(height=300, xaxis_title="Time", yaxis_title="")
    fig
    return


if __name__ == "__main__":
    app.run()
