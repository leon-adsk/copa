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

    csv_path = Path("./data/csv")
    return Path, csv_path, mo, np, pd, plt, px, re, sns


@app.cell
def _(Path, csv_path, pd, re):
    def convert_to_csv():
        for file in Path("./data/txt/").iterdir():
            df = pd.read_csv(f"./data/txt/{file.name}", delimiter=";", header=None)
            df = df.rename(columns={0: "date", 1: "time", 2: "var", 3: "value", 4: "status"})
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%d.%m.%Y %H:%M:%S.%f")
            df["machine"] = df["var"].apply(lambda x: re.findall("M[0-9]*", x)[0])
            df.to_csv(csv_path / f"{file.name[:len(file.name)-4]}.csv", index=False)

    return


@app.cell
def _(Path, pd):
    def load_dataset(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    return (load_dataset,)


@app.cell
def _(Path, csv_path, load_dataset, sns):
    def visualize_vars(path: Path):
        df = load_dataset(path)
        g = sns.FacetGrid(df, col="var", col_wrap=4, sharey=False, aspect=1.6)
        g.map_dataframe(sns.lineplot, x="datetime", y="value")
        g.tick_params(axis="x", labelrotation=45)
        g.set_titles("{col_name}")
        return g

    visualize_vars(csv_path / "FP261104141238_FP.csv")
    return


@app.cell
def _(Path, load_dataset):
    def summarize_datasets():
        csv_path = Path("./data/csv/")
        for file in csv_path.iterdir():
            if file.name[0] == '.':
                continue
    
            print("FILE: " + file.name)

            df = load_dataset(file)
            time_between_records = df["datetime"].diff().mean()

            print(f"Entries: {len(df)}")
            print(f"Machines: {", ".join(df["machine"].unique())}")
            print(f"Mean TbR: {time_between_records}")
            print(f"Vars: {", ".join(df["var"].unique())}")
            print("-----------------------")

    summarize_datasets()
    return


@app.cell
def _(csv_path, load_dataset, mo, pd):
    def inspect_runs(gap_threshold_sec: int = 60):
        rows = []
        run_segments = []

        for file in sorted(csv_path.glob("*.csv")):
            df = load_dataset(file).sort_values("datetime")
            archive = file.stem.split("_")[-1]
            diffs = df["datetime"].diff().dt.total_seconds()

            gap_mask = diffs > gap_threshold_sec
            run_ids = gap_mask.cumsum()

            rows.append({
                "file": file.name,
                "archive": archive,
                "start": df["datetime"].min(),
                "end": df["datetime"].max(),
                "max_gap_s": round(diffs.max(), 2),
                "detected_runs": int(run_ids.nunique()),
            })

            for run_id, grp in df.groupby(run_ids):
                run_segments.append({
                    "file": file.name,
                    "archive": archive,
                    "run_id": int(run_id),
                    "start": grp["datetime"].min(),
                    "end": grp["datetime"].max(),
                    "rows": len(grp),
                })

        return pd.DataFrame(rows), pd.DataFrame(run_segments)

    summary_df, runs_df = inspect_runs(gap_threshold_sec=60)
    mo.vstack([
        mo.md("### File Coverage Summary"),
        summary_df,
        mo.md("### Detected Run Segments"),
        runs_df,
    ])
    return (runs_df,)


@app.cell
def _(px, runs_df):
    fig_runs = px.timeline(
        runs_df,
        x_start="start",
        x_end="end",
        y="file",
        color="archive",
        hover_data=["run_id", "rows"],
        title="Detected Runs Across Archive Files",
    )
    fig_runs.update_yaxes(autorange="reversed")
    fig_runs
    return


@app.cell
def _(csv_path, load_dataset, mo):
    _df = load_dataset(csv_path / "FP261104141238_FP.csv").sort_values("datetime")

    # Check counter resets
    _cnt = _df[_df["var"] == "Sim/Global/M7.Admin.ProductData_ProcessedCount"].copy()
    _resets = _cnt[_cnt["value"].diff() < 0][["datetime", "value"]]

    # Check unique states and how often they change
    _st = _df[_df["var"] == "Sim/Global/M7.Status.StateCurrent"].copy()
    _state_changes = _st[_st["value"] != _st["value"].shift(1)][["datetime", "value"]]

    mo.vstack([
        mo.md(f"**Counter resets found:** {len(_resets)}"),
        _resets.head(20),
        mo.md(f"**Unique states:** {sorted(_st['value'].unique())} | **Transitions:** {len(_state_changes)}"),
        _state_changes.head(20),
    ])
    return


@app.cell
def _(csv_path, load_dataset, mo, np, pd, re):
    # Create master csv file containing archive and run information for all runs
    _csv_files = sorted(f for f in csv_path.glob("*.csv") if f.name != "master.csv")

    _fp_dfs = [load_dataset(f) for f in _csv_files if f.stem.endswith("_FP")]
    _fp_all = pd.concat(_fp_dfs, ignore_index=True).sort_values("datetime")
    _cnt = _fp_all[_fp_all["var"] == "Sim/Global/M7.Admin.ProductData_ProcessedCount"]
    _resets = _cnt[_cnt["value"] < _cnt["value"].shift(1)]["datetime"].to_numpy(dtype="datetime64[us]")

    _parts = []
    for _file in _csv_files:
        _df = load_dataset(_file)
        _df["archive"] = _file.stem.split("_")[-1]
        _df["machine"] = _df["machine"].apply(
            lambda m: f"M{int(re.search(r'\d+', str(m)).group()):02d}"
        )
        _parts.append(_df)

    master_df = pd.concat(_parts, ignore_index=True)

    _dt_vals = master_df["datetime"].to_numpy(dtype="datetime64[us]")
    _min_dt = np.array([_dt_vals.min()], dtype="datetime64[us]")
    _run_starts = np.unique(np.concatenate([_min_dt, _resets]))
    master_df["run_id"] = np.searchsorted(_run_starts, _dt_vals, side="right") - 1

    _cols = ["datetime", "machine", "archive", "run_id", "var", "value", "status"]
    master_df = master_df[_cols].sort_values(["datetime", "archive", "var"], ignore_index=True)

    master_path = csv_path.parent / "master.csv"
    master_df.to_csv(master_path, index=False)

    mo.md(
        f"Saved `{master_path}` — **{len(master_df):,}** rows across "
        f"**{master_df['run_id'].nunique()}** runs and archives "
        f"`{sorted(master_df['archive'].unique())}`."
    )
    return (master_df,)


@app.cell
def _(pd):
    pd.read_csv("./data/master.csv").head()
    return


@app.cell
def _(master_df, mo, px):
    _run_bounds = master_df.groupby("run_id").agg(
        start=("datetime", "min"),
        end=("datetime", "max"),
        archives=("archive", lambda s: ", ".join(sorted(s.unique()))),
        total_rows=("value", "size"),
    )
    _run_bounds["duration_min"] = (
        (_run_bounds["end"] - _run_bounds["start"]).dt.total_seconds() / 60
    ).round(1)

    _fp = master_df[master_df["archive"] == "FP"]
    _cnt_max = (
        _fp[_fp["var"] == "Sim/Global/M7.Admin.ProductData_ProcessedCount"]
        .groupby("run_id")["value"]
        .max()
        .rename("processed_pkgs")
    )
    _def_max = (
        _fp[_fp["var"] == "Sim/Global/M7.Admin.ProductData_DefectiveCount"]
        .groupby("run_id")["value"]
        .max()
        .rename("defective_pkgs")
    )
    _spd_mean = (
        _fp[(_fp["var"] == "Sim/Global/M7.Status.CurMachSpeed") & (_fp["value"] > 0)]
        .groupby("run_id")["value"]
        .mean()
        .round(1)
        .rename("avg_active_speed")
    )
    _pwr_mean = (
        _fp[_fp["var"] == "Sim/Global/M7_Power"]
        .groupby("run_id")["value"]
        .mean()
        .round(2)
        .rename("avg_power")
    )

    runs_summary = _run_bounds.join([_cnt_max, _def_max, _spd_mean, _pwr_mean]).reset_index()

    _fig = px.bar(
        runs_summary,
        x="run_id",
        y="processed_pkgs",
        color="duration_min",
        hover_data=["start", "end", "duration_min", "defective_pkgs", "avg_active_speed", "archives"],
        title="Processed Packages & Duration per Run",
        labels={
            "run_id": "Run ID",
            "processed_pkgs": "Processed Packages",
            "duration_min": "Duration (min)",
        },
    )

    mo.vstack([
        _fig,
        runs_summary,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Charts and Such
    """)
    return


@app.cell
def _(pd, plt):
    import matplotlib.dates as mdates

    def create_gantt(df, target_machine, view_start=None, view_end=None, max_blocks=5000):
        # 1. Filter to just the target machine first to save memory and time
        df_mach = df[df['machine'] == target_machine].copy()

        if df_mach.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f"No data found for machine: {target_machine}", ha='center')
            return fig

        if view_start is None:
            view_start = df_mach['datetime'].min()
        if view_end is None:
            view_end = df_mach['datetime'].max()

        view_start = pd.to_datetime(view_start)
        view_end = pd.to_datetime(view_end)

        df_mach = df_mach.sort_values(by='datetime')

        # 2. Vectorized state changes (only need to check 'var' now)
        changed = df_mach['var'] != df_mach['var'].shift(1)
        blocks = df_mach[changed].copy().rename(columns={'datetime': 'start'})

        blocks['end'] = blocks['start'].shift(-1)
        # The final state block ends at the very last recorded timestamp
        blocks['end'] = blocks['end'].fillna(df_mach['datetime'].max())

        # 3. View bounds and culling
        in_view = (blocks['end'] >= view_start) & (blocks['start'] <= view_end)
        blocks = blocks[in_view].copy()

        if blocks.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No data in this window", ha='center')
            return fig

        blocks['start'] = blocks['start'].clip(lower=view_start)
        blocks['end'] = blocks['end'].clip(upper=view_end)
        blocks['duration_td'] = blocks['end'] - blocks['start']

        if len(blocks) > max_blocks:
            blocks = blocks.nlargest(max_blocks, 'duration_td').sort_values(by='start')

        # 4. Rendering
        fig, ax = plt.subplots(figsize=(12, 6))

        states = sorted(blocks['var'].unique())
        colors = plt.cm.tab10.colors
        color_map = {state: colors[i % len(colors)] for i, state in enumerate(states)}

        blocks['start_num'] = mdates.date2num(blocks['start'])
        blocks['end_num'] = mdates.date2num(blocks['end'])
        blocks['duration'] = blocks['end_num'] - blocks['start_num']

        y_ticks, y_labels = [], []

        # Plot one row per state
        for i, state in enumerate(states):
            state_data = blocks[blocks['var'] == state]
            if not state_data.empty:
                xranges = list(zip(state_data['start_num'], state_data['duration']))
                # Draw all blocks for this state on the same Y-level
                ax.broken_barh(xranges, (i*10 + 2, 6), facecolors=color_map[state])

            y_ticks.append(i*10 + 5)
            y_labels.append(state)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        ax.set_xlim(mdates.date2num(view_start), mdates.date2num(view_end))
        ax.xaxis_date() 
        ax.set_title(f"State Timeline: {target_machine}")

        plt.tight_layout()
        return fig

    return (create_gantt,)


@app.cell
def _(pd, px):
    def create_plotly_gantt(df, target_machine, view_start=None, view_end=None, max_blocks=3000):
        df_mach = df[df['machine'] == target_machine].copy()

        if df_mach.empty:
            return px.timeline(title=f"No data found for machine: {target_machine}")

        if view_start is None:
            view_start = df_mach['datetime'].min()
        if view_end is None:
            view_end = df_mach['datetime'].max()

        view_start = pd.to_datetime(view_start)
        view_end = pd.to_datetime(view_end)

        df_mach = df_mach.sort_values(by='datetime')

        changed = df_mach['var'] != df_mach['var'].shift(1)
        blocks = df_mach[changed].copy().rename(columns={'datetime': 'start'})

        blocks['end'] = blocks['start'].shift(-1)
        blocks['end'] = blocks['end'].fillna(df_mach['datetime'].max())

        in_view = (blocks['end'] >= view_start) & (blocks['start'] <= view_end)
        blocks = blocks[in_view].copy()

        if blocks.empty:
            return px.timeline(title="No data in this window")

        blocks['start'] = blocks['start'].clip(lower=view_start)
        blocks['end'] = blocks['end'].clip(upper=view_end)
        blocks['duration_td'] = blocks['end'] - blocks['start']

        if len(blocks) > max_blocks:
            blocks = blocks.nlargest(max_blocks, 'duration_td').sort_values(by='start')

        fig = px.timeline(
            blocks,
            x_start="start",
            x_end="end",
            y="var",
            color="var",
            title=f"Interactive State Timeline: {target_machine}"
        )

        fig.update_yaxes(categoryorder="category descending")
        fig.update_layout(showlegend=False)

        return fig

    return


@app.cell
def _(csv_path, load_dataset, mo):
    import math
    df = load_dataset(csv_path / "FP261104141238_FP.csv")
    earliest = df["datetime"].min()
    latest = df["datetime"].max()
    start_date = mo.ui.datetime(label="start date", start=earliest, stop=latest, value=earliest)
    end_date = mo.ui.datetime(label="end date", start=earliest, stop=latest, value=latest)

    mo.vstack([start_date, end_date])
    return df, end_date, math, start_date


@app.cell
def _(create_gantt, df, end_date, math, plt, start_date):
    _fig = create_gantt(df, target_machine="M7", max_blocks=math.inf, view_end=end_date.value, view_start=start_date.value)
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
