#!/usr/bin/env python3
"""
pack_colab_data.py
==================
Utility to package EEG dataset files (*.mat) into a compressed tar.gz archive
ready for upload to Google Drive 5TB and processing in Google Colab.

Usage examples:
---------------
1. Package all pending subjects (those not yet in output_data):
   python scripts/pack_colab_data.py --pending-only

2. Package a test batch of the first 5 pending subjects:
   python scripts/pack_colab_data.py --pending-only --limit 5

3. Package all subjects from storage/input_data:
   python scripts/pack_colab_data.py --all

4. Package specific subjects:
   python scripts/pack_colab_data.py --subjects 2,3,4,5
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description="Package EEG subjects for Google Drive / Colab.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Path to input directory containing *.mat files. (Defaults to searching storage/input_data, storage/input_pending_rqa, and storage/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="storage/output_data",
        help="Path to output directory to check for already processed subjects (for --pending-only).",
    )
    parser.add_argument(
        "--dest-archive",
        type=str,
        default="storage/eeg_data_colab.tar.gz",
        help="Output tar.gz file path.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Only package subjects that do NOT have a corresponding *_features.json in output-dir.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Package all discovered subjects.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of subjects to package (e.g. --limit 5). 0 means unlimited.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Comma-separated list of subject IDs or names to package (e.g. '2,3,4' or 'Data_Creativity_Sub_2').",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dest_path = repo_root / args.dest_archive
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Discover all unique .mat subject files
    candidate_dirs = []
    if args.input_dir:
        candidate_dirs.append(Path(args.input_dir))
    else:
        # Default search paths
        candidate_dirs = [
            repo_root / "storage" / "input_data",
            repo_root / "storage" / "input_pending_rqa",
            repo_root / "storage",
        ]

    subject_map: dict[str, Path] = {}
    for cdir in candidate_dirs:
        if cdir.exists():
            for mat_file in sorted(cdir.glob("*.mat")):
                stem = mat_file.stem
                # Exclude aggregated results
                if "NL_Results" in stem or "Creativity_NL_Data" in stem:
                    continue
                if stem not in subject_map:
                    subject_map[stem] = mat_file

    print(f"[Discovery] Found {len(subject_map)} distinct subject .mat files in repository.")

    # 2. Filter subjects
    out_dir = repo_root / args.output_dir
    selected: list[tuple[str, Path]] = []

    # Parse --subjects filter if provided
    filter_keys = set()
    if args.subjects:
        for s in args.subjects.split(","):
            s = s.strip()
            if s:
                filter_keys.add(s)

    for stem, p in sorted(subject_map.items()):
        # Check specific filter
        if filter_keys:
            matches = False
            for fk in filter_keys:
                if stem == fk or stem == f"S{fk}" or stem.endswith(f"_{fk}") or stem.endswith(f"_Sub_{fk}"):
                    matches = True
                    break
                # Check for Data_Creativity_Sub_{fk} pattern
                import re
                m = re.search(r'(?:Sub_|S)(\d+)$', stem, re.IGNORECASE)
                if m and m.group(1) == fk:
                    matches = True
                    break
            if not matches:
                continue

        # Check pending filter
        if args.pending_only:
            feat_json = out_dir / f"{stem}_features.json"
            if feat_json.exists():
                continue

        selected.append((stem, p))

    if args.limit > 0:
        selected = selected[: args.limit]

    if not selected:
        print("[Warning] No subjects matched the criteria to package.")
        return

    print(f"\n[Packaging] Preparing to package {len(selected)} subjects into: {dest_path.name}")
    total_raw_bytes = sum(p.stat().st_size for _, p in selected)
    print(f"[Packaging] Total uncompressed size: {total_raw_bytes / (1024 * 1024):.2f} MB")

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_subjects": len(selected),
        "total_uncompressed_bytes": total_raw_bytes,
        "subjects": [],
    }

    # 3. Create tar.gz archive
    start_time = time.time()
    with tarfile.open(dest_path, "w:gz") as tar:
        for idx, (stem, p) in enumerate(selected, 1):
            print(f"  [{idx}/{len(selected)}] Adding {p.name} ({p.stat().st_size / (1024 * 1024):.1f} MB)...")
            tar.add(p, arcname=p.name)
            manifest["subjects"].append({
                "subject": stem,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
            })

    compressed_size = dest_path.stat().st_size
    elapsed = time.time() - start_time

    # Write manifest alongside
    manifest_path = dest_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[Success] Packaging completed in {elapsed:.1f}s!")
    print(f"Archive file: {dest_path.resolve()}")
    print(f"Compressed archive size: {compressed_size / (1024 * 1024):.2f} MB")
    print(f"Manifest written to: {manifest_path.resolve()}")
    print(f"Subjects packaged: {[s for s, _ in selected]}")


if __name__ == "__main__":
    main()
