import argparse
import json
import os
import re
import sys

import pandas as pd


def get_chrome_trace_df(input_file_path):
    """
    Parses a Chrome trace file and returns a pandas DataFrame.

    Args:
        input_file_path (str): The path to the input Chrome trace JSON file.

    Returns:
        pandas.DataFrame: A DataFrame containing the parsed event data.
    """
    try:
        with open(input_file_path, "r") as f:
            trace_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}")
        return None

    trace_events = trace_data.get("traceEvents", [])
    if not trace_events:
        print("No trace events found in the file.")
        return None

    pid_pattern = re.compile(r"Core(\d+)\s+CTA(\d+)")
    tid_pattern = re.compile(r"warp (\d+)")

    parsed_data = []
    for event in trace_events:
        pid = event.get("pid", "")
        tid = event.get("tid", "")

        pid_match = pid_pattern.search(pid)
        tid_match = tid_pattern.search(tid)

        if pid_match:
            core_id = int(pid_match.group(1))
            cta_id = int(pid_match.group(2))
            # This is the local warp ID within a CTA
            local_warp_id = int(tid_match.group(1)) if tid_match else 0

            dur = event.get("dur")
            parsed_data.append(
                {
                    "name": event.get("name"),
                    "category": event.get("cat"),
                    "cycles": dur * 1000 if dur is not None else None,
                    "timestamp_ns": (
                        event.get("ts") * 1000 if event.get("ts") is not None else None
                    ),
                    "core": core_id,
                    "cta": cta_id,
                    "local_warp_id": local_warp_id,
                }
            )

    return pd.DataFrame(parsed_data)


def get_cutracer_hist_df(input_file_path):
    """
    Parses a CUTRICER trace histogram file and returns an aggregated DataFrame.

    Args:
        input_file_path (str): The path to the input CUTRICER trace CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with aggregated instruction counts per warp and region.
    """
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None

    required_columns = {"warp_id", "region_id", "count"}
    if not required_columns.issubset(df.columns):
        print(f"Error: Input CSV must contain the columns: {list(required_columns)}")
        return None

    summary = df.groupby(["warp_id", "region_id"])["count"].sum().reset_index()
    summary = summary.rename(
        columns={
            "count": "total_instruction_count",
            "warp_id": "global_warp_id",  # Clarify that this is the global warp ID
        }
    )
    return summary


def parse_cutracer_log(log_file_path, kernel_hash_hex):
    """
    Parses a CUTRICER log file to find the launch parameters for a *specific* kernel hash.
    If multiple launches with the same hash are found, it uses the first one and prints a warning.

    Args:
        log_file_path (str): Path to the CUTRICER log file.
        kernel_hash_hex (str): The mandatory target kernel hash (e.g., "0x7fa21c3").

    Returns:
        dict: { 'grid_size': (gx, gy, gz), 'block_size': (bx, by, bz) } or None if not found.
    """
    grid_pattern = re.compile(r"grid size\s+([0-9]+),([0-9]+),([0-9]+)")
    block_pattern = re.compile(r"block size\s+([0-9]+),([0-9]+),([0-9]+)")
    hash_pattern = re.compile(r"kernel hash\s+0x([0-9a-fA-F]+)")

    expected_hash = kernel_hash_hex.lower().lstrip("0x")
    matching_launches = []

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                if "LAUNCH" not in line:
                    continue

                hash_match = hash_pattern.search(line)
                if not hash_match or hash_match.group(1).lower() != expected_hash:
                    continue

                # Found a matching launch line
                grid_match = grid_pattern.search(line)
                block_match = block_pattern.search(line)

                if grid_match and block_match:
                    launch_info = {
                        "grid_size": tuple(map(int, grid_match.groups())),
                        "block_size": tuple(map(int, block_match.groups())),
                    }
                    matching_launches.append(launch_info)

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None

    if not matching_launches:
        return None

    if len(matching_launches) > 1:
        print(
            f"Warning: Found {len(matching_launches)} launches for kernel hash '{kernel_hash_hex}'. "
            "Defaulting to the first one.",
            file=sys.stderr,
        )

    return matching_launches[0]


def calculate_ipc(cycles, instruction_count):
    """
    Calculates Instructions Per Cycle (IPC).

    Args:
        cycles (float): Number of cycles.
        instruction_count (float): Number of instructions.

    Returns:
        float: IPC value, or None if calculation not possible.
    """
    if (
        pd.isna(cycles)
        or pd.isna(instruction_count)
        or cycles <= 0
        or instruction_count <= 0
    ):
        return None

    # Calculate IPC
    ipc = instruction_count / cycles
    return ipc


def validate_warp_coverage(chrome_df, hist_df):
    """
    Validates that Chrome trace and CUTRICER histogram have matching warp IDs.

    Args:
        chrome_df (pandas.DataFrame): Chrome trace data with global_warp_id
        hist_df (pandas.DataFrame): CUTRICER histogram data with global_warp_id

    Returns:
        bool: True if all warp IDs match, False otherwise
    """
    chrome_warp_ids = set(chrome_df["global_warp_id"].unique())
    hist_warp_ids = set(hist_df["global_warp_id"].unique())

    chrome_only = chrome_warp_ids - hist_warp_ids
    hist_only = hist_warp_ids - chrome_warp_ids

    if chrome_only:
        print(
            f"ERROR: Chrome trace has {len(chrome_only)} warps not found in histogram:"
        )
        sorted_chrome_only = sorted(list(chrome_only))
        if len(sorted_chrome_only) <= 10:
            print(f"  Missing warp IDs: {sorted_chrome_only}")
        else:
            print(f"  First 10 missing warp IDs: {sorted_chrome_only[:10]}")
            print(f"  ... and {len(sorted_chrome_only) - 10} more")

    if hist_only:
        print(f"ERROR: Histogram has {len(hist_only)} warps not found in Chrome trace:")
        sorted_hist_only = sorted(list(hist_only))
        if len(sorted_hist_only) <= 10:
            print(f"  Extra warp IDs: {sorted_hist_only}")
        else:
            print(f"  First 10 extra warp IDs: {sorted_hist_only[:10]}")
            print(f"  ... and {len(sorted_hist_only) - 10} more")

    if len(chrome_only) == 0 and len(hist_only) == 0:
        print("✓ Warp ID validation passed: All warps match between data sources")
        return True
    else:
        print(
            f"✗ Warp ID validation failed: {len(chrome_only)} Chrome-only + {len(hist_only)} histogram-only warps"
        )
        return False


def merge_traces(
    chrome_trace_path,
    cutracer_hist_path,
    cutracer_log_path,
    output_path,
    kernel_hash_hex=None,
):
    """
    Merges data from Chrome trace, CUTRICER histogram, and CUTRICER log.
    """
    # If kernel_hash_hex is not provided, try to extract it from the histogram file name
    if not kernel_hash_hex:
        hist_filename = os.path.basename(cutracer_hist_path)
        hash_match = re.search(r"kernel_([0-9a-fA-F]+)_", hist_filename)
        if hash_match:
            kernel_hash_hex = hash_match.group(1)
            print(f"Successfully extracted kernel hash: {kernel_hash_hex}")
        else:
            print(
                f"ERROR: Could not extract kernel hash from filename: {hist_filename}"
            )
            print(
                "       Please provide a kernel hash using the --kernel-hash argument or"
            )
            print(
                "       ensure the histogram file name is in the format 'kernel_<hash>_...'."
            )
            return

    print("Parsing log file to get metadata...")
    launch_info = parse_cutracer_log(cutracer_log_path, kernel_hash_hex)
    if not launch_info:
        print(
            f"Could not find launch info for kernel hash {kernel_hash_hex} in log file. Aborting merge."
        )
        return

    grid_size = launch_info["grid_size"]  # (x, y, z)
    block_size = launch_info["block_size"]  # (x, y, z)

    # Calculate total threads per block and warps per block
    threads_per_block = block_size[0] * block_size[1] * block_size[2]
    warp_size = 32  # Standard for NVIDIA GPUs
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size

    if kernel_hash_hex:
        print(
            f"Launch info parsed for kernel hash {kernel_hash_hex}: Grid size = {grid_size}, Block size = {block_size}, Warps per block = {warps_per_block}"
        )
    else:
        print(
            f"Launch info parsed: Grid size = {grid_size}, Block size = {block_size}, Warps per block = {warps_per_block}"
        )

    print("Parsing Chrome trace file...")
    chrome_df = get_chrome_trace_df(chrome_trace_path)
    if chrome_df is None:
        print("Failed to parse Chrome trace. Aborting merge.")
        return

    print("Parsing CUTRICER histogram file...")
    hist_df = get_cutracer_hist_df(cutracer_hist_path)
    if hist_df is None:
        print("Failed to parse CUTRICER histogram. Aborting merge.")
        return

    # Calculate global_warp_id in the chrome trace data
    # For 3D grids, we need to check if cta is already linearized or if we need to linearize it
    # Based on your grid (1, 528, 1), it seems cta is already a linear block ID
    # global_warp_id = block_linear_id * warps_per_block + local_warp_id

    print("Analyzing CTA ID distribution...")
    print(f"CTA ID range: {chrome_df['cta'].min()} to {chrome_df['cta'].max()}")
    total_blocks = grid_size[0] * grid_size[1] * grid_size[2]
    print(f"Expected total blocks from grid size: {total_blocks}")

    # If cta IDs are 1-based and linearized, we need to convert to 0-based
    # If cta IDs start from a different base, we need to adjust accordingly
    chrome_df["global_warp_id"] = (
        chrome_df["cta"] * warps_per_block + chrome_df["local_warp_id"]
    )

    print(
        f"Global warp ID range: {chrome_df['global_warp_id'].min()} to {chrome_df['global_warp_id'].max()}"
    )

    print("Validating warp coverage...")
    is_valid = validate_warp_coverage(chrome_df, hist_df)
    if not is_valid:
        print("WARNING: Proceeding with merge despite warp ID mismatches.")
        print("         This may result in missing data in the output.")

    print("Merging the dataframes...")
    # Merge all regions - create cross join between chrome trace events and histogram regions
    merged_df = pd.merge(chrome_df, hist_df, on="global_warp_id", how="left")

    # Reorder and select columns for clarity
    output_columns = [
        "core",
        "cta",
        "local_warp_id",
        "global_warp_id",
        "region_id",
        "name",
        "category",
        "cycles",
        "timestamp_ns",
        "total_instruction_count",
    ]

    # Add IPC calculation
    print("Calculating IPC (Instructions Per Cycle)...")
    merged_df["ipc"] = merged_df.apply(
        lambda row: calculate_ipc(row["cycles"], row["total_instruction_count"]), axis=1
    )
    output_columns.append("ipc")

    # Filter for columns that actually exist after the merge
    final_columns = [col for col in output_columns if col in merged_df.columns]
    final_df = merged_df[final_columns]

    # Sort by global_warp_id first, then by region_id for better organization
    print("Sorting output by global_warp_id, then region_id...")
    final_df = final_df.sort_values(
        ["global_warp_id", "region_id"], ascending=True
    ).reset_index(drop=True)

    final_df.to_csv(output_path, index=False)
    print(f"Successfully merged data and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse and merge trace files from Chrome's tracer and CUTRICER.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--chrome-trace",
        dest="chrome_trace_input",
        help="Path to the Chrome trace JSON file.",
    )
    parser.add_argument(
        "--cutracer-trace",
        dest="cutracer_trace_input",
        help="Path to the CUTRICER histogram CSV file.",
    )
    parser.add_argument(
        "--cutracer-log",
        dest="cutracer_log_input",
        help="Path to the CUTRICER log file to enable merge mode.",
    )
    parser.add_argument(
        "--kernel-hash",
        dest="kernel_hash_hex",
        help="Optional kernel hash (e.g., 0x7fa21c3) to select a specific launch from the log.",
    )
    parser.add_argument("--output", required=True, help="Path for the output CSV file.")

    args = parser.parse_args()

    # --- Main Logic ---
    if args.cutracer_log_input:
        # Merge mode
        if not all([args.chrome_trace_input, args.cutracer_trace_input]):
            parser.error("--cutracer-log requires --chrome-trace and --cutracer-trace.")

        merge_traces(
            args.chrome_trace_input,
            args.cutracer_trace_input,
            args.cutracer_log_input,
            args.output,
            args.kernel_hash_hex,
        )
    elif args.chrome_trace_input:
        # Standalone Chrome trace parsing
        df = get_chrome_trace_df(args.chrome_trace_input)
        if df is not None:
            df.to_csv(args.output, index=False)
            print(f"Successfully parsed Chrome trace and saved to {args.output}")
    elif args.cutracer_trace_input:
        # Standalone CUTRICER hist parsing
        df = get_cutracer_hist_df(args.cutracer_trace_input)
        if df is not None:
            df.to_csv(args.output, index=False)
            print(f"Successfully parsed CUTRICER histogram and saved to {args.output}")
    else:
        parser.print_help()
