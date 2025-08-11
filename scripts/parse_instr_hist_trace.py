import json
import re
import argparse
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

            parsed_data.append(
                {
                    "name": event.get("name"),
                    "category": event.get("cat"),
                    "duration_ns": event.get("dur"),
                    "timestamp_ns": event.get("ts"),
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


def parse_cutracer_log(log_file_path):
    """
    Parses a CUTRICER log file to find kernel launch parameters.

    Args:
        log_file_path (str): Path to the CUTRICER log file.

    Returns:
        dict: A dictionary with 'grid_size' and 'block_size' tuples.
    """
    grid_pattern = re.compile(r"grid size\s+([0-9]+),([0-9]+),([0-9]+)")
    block_pattern = re.compile(r"block size\s+([0-9]+),([0-9]+),([0-9]+)")

    last_launch_info = {}
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                if "LAUNCH" in line:
                    grid_match = grid_pattern.search(line)
                    block_match = block_pattern.search(line)
                    if grid_match and block_match:
                        last_launch_info = {
                            "grid_size": tuple(map(int, grid_match.groups())),
                            "block_size": tuple(map(int, block_match.groups())),
                        }
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None

    return last_launch_info if last_launch_info else None


def merge_traces(chrome_trace_path, cutracer_hist_path, cutracer_log_path, output_path):
    """
    Merges data from Chrome trace, CUTRICER histogram, and CUTRICER log.
    """
    print("Parsing log file to get metadata...")
    launch_info = parse_cutracer_log(cutracer_log_path)
    if not launch_info:
        print("Could not parse launch info from log file. Aborting merge.")
        return

    block_x = launch_info["block_size"][0]
    # Assuming warp size is 32, which is standard for NVIDIA GPUs.
    warp_size = 32
    warps_per_block = (block_x + warp_size - 1) // warp_size
    print(
        f"Launch info parsed: Block size = {launch_info['block_size']}, Warps per block = {warps_per_block}"
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
    # global_warp_id = cta_id * warps_per_block + local_warp_id
    chrome_df["global_warp_id"] = (
        chrome_df["cta"] * warps_per_block + chrome_df["local_warp_id"]
    )

    print("Merging the dataframes...")
    # Merge on the calculated global_warp_id and the region_id (assuming region 0 for simplicity for now)
    # The histogram from cutracer_trace may have multiple regions, we'll merge with the primary one (0).
    merged_df = pd.merge(
        chrome_df, hist_df[hist_df["region_id"] == 0], on="global_warp_id", how="left"
    )

    # Reorder and select columns for clarity
    output_columns = [
        "core",
        "cta",
        "local_warp_id",
        "global_warp_id",
        "region_id",
        "name",
        "category",
        "duration_ns",
        "timestamp_ns",
        "total_instruction_count",
    ]
    # Filter for columns that actually exist after the merge
    final_columns = [col for col in output_columns if col in merged_df.columns]
    final_df = merged_df[final_columns]

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
