
import pandas as pd
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

# Automatically locate the repository root
repo_path = Path(__file__).resolve().parent.parent

# Original zenon archive directory
base_path = repo_path / "data" / "txt"

# Input files from the original simulation run
large_path = base_path / "FP261104141238_FP.TXT"
medium_path = base_path / "17261104141251_17.TXT"
small_path = base_path / "07261104141229_07.TXT"

# Create output directory for processed data
output_dir = repo_path / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "M7_full_source_times_corrected.csv"

# Verify that all source files exist
for path in [large_path, medium_path, small_path]:

    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    print(f"Source file found: {path.name}")


# ============================================================
# 2. LOAD AND CLEAN THE ORIGINAL ARCHIVES
# ============================================================

def load_archive(path):

    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["date", "time", "tag", "value", "status"],
        dtype=str,
        keep_default_na=False
    )

    # Remove INVALID observations
    df = df[
        ~df["status"].str.contains(
            r"\bINVALID\b",
            case=False,
            regex=True
        )
    ].copy()

    # Combine date and time while preserving milliseconds
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d.%m.%Y %H:%M:%S.%f"
    )

    # Convert measurement values to numeric format
    df["value"] = pd.to_numeric(df["value"])

    # Create whole-second timestamps for joining
    df["second"] = df["datetime"].dt.floor("s")

    # Sort observations chronologically
    return df.sort_values("datetime", kind="stable")


large = load_archive(large_path)
medium = load_archive(medium_path)
small = load_archive(small_path)

print("Large:", large.shape)
print("Medium:", medium.shape)
print("Small:", small.shape)


# ============================================================
# 3. VARIABLE NAMES
# ============================================================

large_names = {
    "Sim/Global/M7_Energy_incremental":
        "accumulated_power_consumption",

    "Sim/Global/M7.Admin.ProductData_DefectiveCount":
        "defective_bottles_current_order",

    "Sim/Global/M7_CompAir_incremental":
        "accumulated_compressed_air_consumption",

    "Sim/Global/M7_CompAir":
        "current_compressed_air_input",

    "Sim/Global/M7_FillWeight":
        "current_bottle_fill_weight",

    "Sim/Global/M7_Power":
        "current_filler_power_input_large",

    "Sim/Global/M7.Status.CurMachSpeed":
        "current_machine_speed_large",

    "Sim/Global/M7.Status.MachSpeed":
        "target_machine_speed",

    "Sim/Global/M7.Status.StateCurrent":
        "current_machine_state_code_large",

    "Sim/Global/M7.Status.UnitModeCurrent":
        "current_unit_mode_code_large",

    "Sim/Global/M7.Admin.ProductData_ProcessedCount":
        "total_filled_bottles_current_order_large"
}


medium_names = {
    "M07.WS_Good_packages":
        "good_filled_bottles_current_order",

    "M07.WS_Tot_Packages":
        "total_filled_bottles_current_order_medium",

    "M07.WS_Cur_Mach_Spd":
        "current_machine_speed_medium",

    "Converter/Global/M07.Power":
        "current_filler_power_input_medium"
}


small_names = {
    "M07.WS_Cur_State":
        "current_machine_state_code_small",

    "M07.WS_Cur_Prog":
        "current_program_code",

    "M07.WS_Cur_Mode":
        "current_unit_mode_code_small"
}


# ============================================================
# 4. CONVERT CYCLIC ARCHIVES TO WIDE FORMAT
# ============================================================

def convert_cyclic_to_wide(df, names, source):

    # Verify that cyclic observations occur on whole seconds
    assert (df["datetime"] == df["second"]).all()

    # Verify expected variable names
    assert set(df["tag"].unique()) == set(names.keys())

    # Check for duplicate timestamp-variable combinations
    assert not df.duplicated(["second", "tag"]).any()

    # Convert variables from rows into individual columns
    wide = df.pivot(
        index="second",
        columns="tag",
        values="value"
    )

    # Rename columns
    wide = wide.rename(columns=names)

    # Preserve the specified column order
    wide = wide.reindex(columns=list(names.values()))

    # Preserve original source timestamps
    wide[f"timestamp_{source}"] = wide.index.strftime(
        "%Y-%m-%d %H:%M:%S.000"
    )

    return wide


large_wide = convert_cyclic_to_wide(
    large, large_names, "large"
)

medium_wide = convert_cyclic_to_wide(
    medium, medium_names, "medium"
)

print("\nLarge archive preview:")
print(large_wide.head())

print("\nMedium archive preview:")
print(medium_wide.head())


# ============================================================
# 5. MACHINE STATE, PROGRAM AND MODE DEFINITIONS
# ============================================================

state_codes = {
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
    1048576: "Clearing"
}


program_codes = {
    1: "Production",
    2: "Start Up",
    4: "Run Down",
    8: "Cleaning",
    16: "Changeover",
    32: "Maintenance",
    64: "Break"
}


mode_codes = {
    1: "OFF",
    2: "Manual",
    4: "Semi-Automatic",
    8: "Automatic"
}


event_mapping = {
    "M07.WS_Cur_State": ("machine_state", state_codes),
    "M07.WS_Cur_Prog": ("program", program_codes),
    "M07.WS_Cur_Mode": ("control_mode", mode_codes)
}


# ============================================================
# 6. PROCESS SMALL ARCHIVE EVENTS
# ============================================================

event_rows = {}
previous_values = {}

for row in small.itertuples(index=False):

    second = row.second
    tag = row.tag
    value = row.value

    label_column, code_mapping = event_mapping[tag]

    # Translate numerical codes into descriptive labels
    label = code_mapping.get(
        value,
        f"Unknown code {value:g}"
    )

    # Preserve original millisecond timestamps
    exact_time = row.datetime.strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]

    if second not in event_rows:

        event_rows[second] = {
            "timestamps": [],
            "details": []
        }

    event = event_rows[second]

    # Preserve distinct original event timestamps
    if exact_time not in event["timestamps"]:

        event["timestamps"].append(exact_time)

    # Identify first recorded observations
    if tag not in previous_values:

        description = (
            f"first recorded as {label} ({value:g})"
        )

    # Identify repeated observations
    elif previous_values[tag][0] == value:

        description = (
            f"recorded again as {label} ({value:g})"
        )

    # Identify changes in state, program or control mode
    else:

        old_value, old_label = previous_values[tag]

        description = (
            f"changed from {old_label} ({old_value:g}) "
            f"to {label} ({value:g})"
        )

    previous_values[tag] = (value, label)

    # Preserve every event, including multiple events
    # occurring within the same second
    event["details"].append(
        f"{exact_time} - {tag}: {description}"
    )

    # Store numerical codes and descriptive labels
    # For same-second repetitions, retain the latest value
    event[small_names[tag]] = value
    event[label_column] = label


# Convert event records into a DataFrame
small_wide = pd.DataFrame.from_dict(
    event_rows,
    orient="index"
)

# Preserve all original event timestamps
small_wide["timestamp_small"] = (
    small_wide["timestamps"].apply(" | ".join)
)

# Preserve all event descriptions
small_wide["event_details"] = (
    small_wide["details"].apply(" | ".join)
)

# Remove temporary processing columns
small_wide = small_wide.drop(
    columns=["timestamps", "details"]
)

small_wide.index.name = "second"

print("\nSmall archive preview:")
print(small_wide.head())


# ============================================================
# 7. CREATE CONTINUOUS ONE-SECOND TIMELINE
# ============================================================

# Identify the earliest observation across all archives
start_time = min(
    large["datetime"].min(),
    medium["datetime"].min(),
    small["datetime"].min()
).floor("s")

# Identify the latest observation across all archives
end_time = max(
    large["datetime"].max(),
    medium["datetime"].max(),
    small["datetime"].max()
).floor("s")

# Generate a continuous one-second timeline
timeline = pd.date_range(
    start=start_time,
    end=end_time,
    freq="s"
)

combined = pd.DataFrame(index=timeline)

combined.index.name = "timestamp"

print("\nStart:", start_time)
print("End:", end_time)
print("Total seconds:", len(combined))


# ============================================================
# 8. JOIN ALL THREE ARCHIVES
# ============================================================

# Left joins preserve the complete timeline.
# Missing measurements remain empty at this stage.

combined = combined.join(
    large_wide,
    how="left"
)

combined = combined.join(
    medium_wide,
    how="left"
)

combined = combined.join(
    small_wide,
    how="left"
)


# ============================================================
# 9. ARRANGE COLUMNS
# ============================================================

column_order = [
    "timestamp_large",
    "timestamp_medium",
    "timestamp_small",

    "machine_state",
    "program",
    "control_mode",

    *large_names.values(),
    *medium_names.values(),
    *small_names.values(),

    "event_details"
]

combined = combined[column_order]

combined = combined.reset_index()

print("\nFinal dataset shape:", combined.shape)


# ============================================================
# 10. VALIDATE ORIGINAL MEASUREMENT COUNTS
# ============================================================

for source, original, names in [
    ("Large", large, large_names),
    ("Medium", medium, medium_names),
    ("Small", small, small_names)
]:

    original_count = len(original)

    joined_count = combined[
        list(names.values())
    ].notna().sum().sum()

    collisions = original.duplicated(
        ["second", "tag"]
    ).sum()

    print(f"\n{source} archive")
    print("Original readings:", original_count)
    print("Populated measurement cells:", joined_count)
    print("Same-second extra records:", collisions)

    # Account for observations sharing a second and tag
    assert joined_count + collisions == original_count


# Verify that every valid small-archive event is retained
event_count = sum(
    len(row["details"])
    for row in event_rows.values()
)

assert event_count == len(small)

print("\nAll small-archive events preserved:", event_count)


# Verify continuous one-second intervals
assert (
    combined["timestamp"].diff().dropna()
    == pd.Timedelta(seconds=1)
).all()

print("One-second timeline verification passed!")


# ============================================================
# 11. PRESERVE FULL TIMESTAMP FORMATTING
# ============================================================

combined["timestamp"] = (
    combined["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S.000"
    )
)

timestamp_columns = [
    "timestamp",
    "timestamp_large",
    "timestamp_medium",
    "timestamp_small"
]

# Convert timestamp columns into explicit strings
for col in timestamp_columns:

    combined[col] = combined[col].astype("string")


# ============================================================
# 12. EXPORT FINAL CSV
# ============================================================

combined.to_csv(
    output_path,
    index=False,
    na_rep="",
    encoding="utf-8-sig"
)

print("\nCSV successfully saved to:")
print(output_path)


# ============================================================
# 13. VERIFY TIMESTAMPS IN THE SAVED CSV
# ============================================================

check = pd.read_csv(
    output_path,
    dtype=str,
    usecols=timestamp_columns
)

print("\nVerifying saved timestamps:")

print(
    check.loc[
        check["timestamp_small"].notna(),
        timestamp_columns
    ].head(10).to_string(index=False)
)

# Confirm timestamps survived the CSV export unchanged
pd.testing.assert_frame_equal(
    combined[timestamp_columns].fillna("").astype(str),
    check[timestamp_columns].fillna("").astype(str)
)

print("\nTimestamp verification passed!")

print("\nTotal rows:", len(combined))
print("Total columns:", len(combined.columns))

print("\nDATA PREPARATION COMPLETED SUCCESSFULLY!")
