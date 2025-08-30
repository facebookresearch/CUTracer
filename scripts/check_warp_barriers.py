#!/usr/bin/env python3
"""
Script to check if the last two instructions for each warp are BAR.SYNC.DEFER_BLOCKING
"""

import subprocess
import sys
import os

def run_grep_for_warp(warp_id, log_file):
    """Run grep command for a specific warp and return the output lines"""
    try:
        cmd = ['grep', '-rn', f'warp {warp_id} ', log_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        return [line for line in lines if line.strip()]  # Filter out empty lines
    except subprocess.CalledProcessError:
        return []

def check_last_two_instructions(lines):
    """Check if the last two lines contain BAR.SYNC.DEFER_BLOCKING"""
    if len(lines) < 2:
        return False, f"Only {len(lines)} instruction(s) found"
    
    last_two = lines[-2:]
    barrier_count = 0
    
    for line in last_two:
        if 'BAR.SYNC.DEFER_BLOCKING' in line:
            barrier_count += 1
    
    return barrier_count == 2, last_two

def main():
    # Configuration
    warp_file = '/tmp/a.txt'
    log_file = '/home/yhao/tlx_env/tritonbench/kernel_2abe4d1da99b63a0_iter0_gdpa_kernel_tma_ws_blackwell.log'
    
    # Check if files exist
    if not os.path.exists(warp_file):
        print(f"Error: Warp file {warp_file} not found!")
        sys.exit(1)
    
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found!")
        sys.exit(1)
    
    # Read warp IDs from file
    try:
        with open(warp_file, 'r') as f:
            warp_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading warp file: {e}")
        sys.exit(1)
    
    # Skip the first warp (8) as already checked
    if warp_ids and warp_ids[0] == '8':
        warp_ids = warp_ids[1:]
        print("Skipping warp 8 (already checked)")
    
    print(f"Checking {len(warp_ids)} warps...")
    print("=" * 50)
    
    passed_count = 0
    failed_count = 0
    
    for i, warp_id in enumerate(warp_ids, 1):
        print(f"\n[{i}/{len(warp_ids)}] Checking warp {warp_id}...")
        
        # Get all instructions for this warp
        lines = run_grep_for_warp(warp_id, log_file)
        
        if not lines:
            print(f"  ✗ No instructions found for warp {warp_id}")
            failed_count += 1
            continue
        
        # Check last two instructions
        is_valid, details = check_last_two_instructions(lines)
        
        if is_valid:
            print(f"  ✓ Last two instructions are BAR.SYNC.DEFER_BLOCKING")
            passed_count += 1
        else:
            print(f"  ✗ Failed - {details if isinstance(details, str) else 'Last two instructions:'}")
            if isinstance(details, list):
                for j, line in enumerate(details, 1):
                    # Extract just the instruction part (after the last " - ")
                    instruction = line.split(' - ')[-1] if ' - ' in line else line
                    print(f"    {j}. {instruction}")
            failed_count += 1
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Total warps checked: {len(warp_ids)}")
    print(f"Passed: {passed_count} ✓")
    print(f"Failed: {failed_count} ✗")
    
    if failed_count == 0:
        print("\n🎉 All warps have BAR.SYNC.DEFER_BLOCKING as their last two instructions!")
    else:
        print(f"\n⚠️  {failed_count} warp(s) do not match the expected pattern.")

if __name__ == '__main__':
    main()
