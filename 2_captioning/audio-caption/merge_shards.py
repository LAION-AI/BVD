#!/usr/bin/env python
"""
Merge shard tar files back into single tar files.
E.g., 0000_shard00.tar + 0000_shard01.tar -> 0000.tar
"""
import os
import json
import tarfile
import argparse
from pathlib import Path
from collections import defaultdict
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge shard tar files into single tar files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing shard tar files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for merged tar files (default: same as input_dir)",
    )
    parser.add_argument(
        "--delete_shards",
        action="store_true",
        help="Delete shard files after successful merge",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually merging",
    )
    return parser.parse_args()


def find_shard_files(input_dir: Path) -> dict:
    """Find all shard files and group by base tar name.
    
    Returns:
        Dict mapping base tar name (e.g., "0000.tar") to list of shard paths
    """
    shard_pattern = re.compile(r'^(\d{4})_shard(\d{2})\.tar$')
    shards = defaultdict(list)
    
    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        match = shard_pattern.match(f.name)
        if match:
            base_name = f"{match.group(1)}.tar"
            shard_index = int(match.group(2))
            shards[base_name].append((shard_index, f))
    
    # Sort shards by index
    for base_name in shards:
        shards[base_name].sort(key=lambda x: x[0])
    
    return dict(shards)


def merge_tar_files(shard_paths: list, output_path: Path) -> int:
    """Merge multiple shard tar files into a single tar file.
    
    Args:
        shard_paths: List of (shard_index, path) tuples, sorted by index
        output_path: Path for the merged output tar
        
    Returns:
        Number of files added to the merged tar
    """
    total_files = 0
    
    with tarfile.open(output_path, 'w') as out_tar:
        for shard_index, shard_path in shard_paths:
            with tarfile.open(shard_path, 'r') as in_tar:
                for member in in_tar.getmembers():
                    if member.isfile():
                        f = in_tar.extractfile(member)
                        if f is not None:
                            out_tar.addfile(member, f)
                            total_files += 1
    
    return total_files


def update_sizes_json(output_dir: Path, merged_tars: dict):
    """Update sizes.json with merged tar sample counts."""
    sizes_path = output_dir / "sizes.json"
    
    if sizes_path.exists():
        with open(sizes_path, 'r') as f:
            sizes_data = json.load(f)
    else:
        sizes_data = {}
    
    # Remove shard entries and add merged entries
    keys_to_remove = [k for k in sizes_data if '_shard' in k]
    for k in keys_to_remove:
        del sizes_data[k]
    
    # Add merged tar counts (divide by 2 since each sample has audio + json)
    for tar_name, file_count in merged_tars.items():
        sizes_data[tar_name] = file_count // 2
    
    with open(sizes_path, 'w') as f:
        json.dump(sizes_data, f, indent=4, sort_keys=True)
    
    print(f"Updated sizes.json with {len(merged_tars)} merged tars")


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find shard files
    print(f"Scanning for shard files in: {input_dir}")
    shards = find_shard_files(input_dir)
    
    if not shards:
        print("No shard files found.")
        return
    
    print(f"Found {len(shards)} tar files to merge:")
    for base_name, shard_list in sorted(shards.items()):
        shard_indices = [idx for idx, _ in shard_list]
        print(f"  {base_name}: {len(shard_list)} shards (indices: {shard_indices})")
    
    if args.dry_run:
        print("\nDry run - no files will be modified.")
        return
    
    # Merge each set of shards
    print("\nMerging shards...")
    merged_tars = {}
    
    for base_name, shard_list in sorted(shards.items()):
        output_path = output_dir / base_name
        print(f"  Merging {len(shard_list)} shards -> {output_path}")
        
        file_count = merge_tar_files(shard_list, output_path)
        merged_tars[base_name] = file_count
        
        print(f"    Added {file_count} files ({file_count // 2} samples)")
    
    # Update sizes.json
    update_sizes_json(output_dir, merged_tars)
    
    # Delete shard files if requested
    if args.delete_shards:
        print("\nDeleting shard files...")
        for base_name, shard_list in shards.items():
            for shard_index, shard_path in shard_list:
                print(f"  Deleting: {shard_path.name}")
                shard_path.unlink()
        print("Shard files deleted.")
    
    print(f"\nMerge complete! {len(merged_tars)} tar files created in {output_dir}")


if __name__ == "__main__":
    main()

