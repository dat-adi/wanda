#!/usr/bin/env python3
import torch
import os
import argparse
from pathlib import Path

def check_eight_consecutive_zeros(tensor):
    """
    Check for exactly 8 consecutive zeros in a tensor.
    Returns (has_eight_consecutive_zeros: bool, count: int)
    """
    # Flatten the tensor to 1D for easier processing
    flat_tensor = tensor.flatten()

    count = 0
    current_zeros = 0

    for value in flat_tensor:
        if value == 0:
            current_zeros += 1
        else:
            # When we hit a non-zero, check if we had exactly 8 consecutive zeros
            if current_zeros >= 8:
                # Count how many groups of 8+ consecutive zeros we found
                count += (current_zeros // 8)
            current_zeros = 0

    # Check the end of the tensor in case it ends with zeros
    if current_zeros >= 8:
        count += (current_zeros // 8)

    return count > 0, count

def process_pt_file(file_path):
    """Process a single .pt file and check all tensors within it."""
    try:
        data = torch.load(file_path, map_location='cpu')

        total_count = 0
        has_any_eight_consecutive = False

        # Handle different data structures
        if isinstance(data, torch.Tensor):
            # Single tensor
            has_eight, count = check_eight_consecutive_zeros(data)
            total_count += count
            has_any_eight_consecutive = has_any_eight_consecutive or has_eight
        elif isinstance(data, dict):
            # Dictionary of tensors (like state_dict)
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    has_eight, count = check_eight_consecutive_zeros(value)
                    total_count += count
                    has_any_eight_consecutive = has_any_eight_consecutive or has_eight
        elif isinstance(data, (list, tuple)):
            # List/tuple of tensors
            for item in data:
                if isinstance(item, torch.Tensor):
                    has_eight, count = check_eight_consecutive_zeros(item)
                    total_count += count
                    has_any_eight_consecutive = has_any_eight_consecutive or has_eight

        return has_any_eight_consecutive, total_count

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0

def main():
    parser = argparse.ArgumentParser(description='Check for eight consecutive zeros in .pt files')
    parser.add_argument('folder', type=str, help='Folder containing .pt files')

    args = parser.parse_args()

    folder_path = Path(args.folder)

    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return

    # Find all .pt files
    pt_files = list(folder_path.glob('*.pt'))

    if not pt_files:
        print(f"No .pt files found in {folder_path}")
        return

    print(f"Found {len(pt_files)} .pt files")
    print("Checking for EIGHT consecutive zeros...")
    print("-" * 50)

    overall_has_eight_consecutive = False
    overall_count = 0

    for pt_file in pt_files:
        has_eight_consecutive, count = process_pt_file(pt_file)
        overall_has_eight_consecutive = overall_has_eight_consecutive or has_eight_consecutive
        overall_count += count

        print(f"{pt_file.name}: {has_eight_consecutive} (count: {count})")

    print("-" * 50)
    print(f"Overall result:")
    print(f"Has 8 consecutive zeros: {overall_has_eight_consecutive}")
    print(f"Total count of 8-consecutive-zero groups: {overall_count}")

if __name__ == "__main__":
    main()