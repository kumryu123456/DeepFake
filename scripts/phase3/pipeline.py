"""
Integrated Audio Segmentation Pipeline
Combines TEN VAD segmentation with pyannote speaker verification
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(command, description):
    """Run a subprocess command with error handling"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}\n")

    result = subprocess.run(command, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False

    print(f"\n[OK] {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Integrated segmentation pipeline")
    parser.add_argument("--input", required=True, help="Input directory with WAV files")
    parser.add_argument("--token", required=True, help="HuggingFace authentication token")
    parser.add_argument("--min-duration", type=float, default=4.0, help="Minimum segment duration")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Maximum segment duration")
    parser.add_argument("--min-confidence", type=float, default=0.95, help="Minimum speaker confidence")

    args = parser.parse_args()

    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    segments_dir = f"data/phase3_segments_{timestamp}"
    verified_dir = f"data/phase3_verified_{timestamp}"
    metadata_segments = f"data/tenvad_metadata_{timestamp}.jsonl"
    metadata_verified = f"data/verified_metadata_{timestamp}.jsonl"

    # Get python executable
    python_exe = sys.executable

    # Step 1: TEN VAD Segmentation
    print(f"\n{'#'*60}")
    print(f"# PHASE 3: TEN VAD + pyannote Pipeline")
    print(f"{'#'*60}")
    print(f"Input: {args.input}")
    print(f"Segments output: {segments_dir}")
    print(f"Verified output: {verified_dir}")
    print(f"Duration filter: {args.min_duration}s - {args.max_duration}s")
    print(f"Speaker confidence: {args.min_confidence}")
    print(f"{'#'*60}\n")

    segment_command = [
        python_exe,
        "scripts/phase3/segment_with_tenvad.py",
        "--input", args.input,
        "--output", segments_dir,
        "--min-duration", str(args.min_duration),
        "--max-duration", str(args.max_duration),
        "--metadata", metadata_segments
    ]

    if not run_command(segment_command, "Step 1: TEN VAD Segmentation"):
        sys.exit(1)

    # Step 2: Speaker Verification
    verify_command = [
        python_exe,
        "scripts/phase3/verify_speakers.py",
        "--input", segments_dir,
        "--output", verified_dir,
        "--token", args.token,
        "--metadata-in", metadata_segments,
        "--metadata-out", metadata_verified,
        "--min-confidence", str(args.min_confidence)
    ]

    if not run_command(verify_command, "Step 2: Speaker Verification"):
        sys.exit(1)

    # Final summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Verified segments: {verified_dir}")
    print(f"Verified metadata: {metadata_verified}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
