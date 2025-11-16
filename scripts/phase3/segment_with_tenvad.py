"""
TEN VAD Audio Segmentation Script
Segments audio files using TEN VAD (Voice Activity Detection)
Filters segments to 4-10 seconds duration
"""

import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime

# Import TEN VAD
try:
    from ten_vad import TenVad
except ImportError:
    print("ERROR: ten_vad not installed. Please install: pip install git+https://github.com/TEN-framework/ten-vad.git")
    sys.exit(1)


class TenVADSegmenter:
    """TEN VAD-based audio segmentation with 4-10s filtering"""

    def __init__(self, min_duration=4.0, max_duration=10.0, sample_rate=16000):
        """
        Initialize TEN VAD segmenter

        Args:
            min_duration: Minimum segment duration in seconds (default: 4.0)
            max_duration: Maximum segment duration in seconds (default: 10.0)
            sample_rate: Audio sample rate (default: 16000)
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate

        # Initialize TEN VAD model
        print("Loading TEN VAD model...")
        self.vad = TenVad()
        print("TEN VAD model loaded successfully")

    def segment_audio(self, audio_path, output_dir):
        """
        Segment audio file using TEN VAD

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save segments

        Returns:
            List of segment metadata dictionaries
        """
        print(f"\nProcessing: {audio_path}")

        # Load audio
        audio, sr = sf.read(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            print(f"Warning: Audio sample rate {sr} != {self.sample_rate}, using as-is")

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Convert to int16 format (required by TEN VAD)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Run TEN VAD - process in chunks
        print("Running TEN VAD detection...")
        timestamps = self._process_with_vad(audio_int16, sr)

        print(f"Found {len(timestamps)} speech segments")

        # Filter and save segments
        segments_metadata = []
        base_name = Path(audio_path).stem

        for idx, (start_sec, end_sec) in enumerate(timestamps):
            duration = end_sec - start_sec

            # Filter by duration
            if duration < self.min_duration:
                print(f"  Segment {idx}: {duration:.2f}s - TOO SHORT, skipping")
                continue

            if duration > self.max_duration:
                print(f"  Segment {idx}: {duration:.2f}s - TOO LONG, skipping")
                continue

            # Extract segment (use original float audio for saving)
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            segment_audio = audio[start_sample:end_sample]

            # Save segment
            output_filename = f"{base_name}_seg_{idx:03d}.wav"
            output_path = os.path.join(output_dir, output_filename)
            sf.write(output_path, segment_audio, sr)

            # Store metadata
            metadata = {
                "filename": output_filename,
                "source_file": os.path.basename(audio_path),
                "start_time": float(start_sec),
                "end_time": float(end_sec),
                "duration": float(duration),
                "sample_rate": int(sr),
                "vad_engine": "TEN_VAD"
            }
            segments_metadata.append(metadata)

            print(f"  [OK] Segment {idx}: {duration:.2f}s -> {output_filename}")

        return segments_metadata

    def _process_with_vad(self, audio_int16, sample_rate):
        """
        Process audio with TEN VAD to detect speech segments

        Args:
            audio_int16: Audio as int16 numpy array
            sample_rate: Audio sample rate

        Returns:
            List of (start_sec, end_sec) tuples
        """
        hop_size = self.vad.hop_size
        total_samples = len(audio_int16)
        num_chunks = total_samples // hop_size

        # Track voice activity
        voice_active = False
        current_start = None
        raw_segments = []

        for i in range(num_chunks):
            # Extract chunk
            start_idx = i * hop_size
            end_idx = start_idx + hop_size
            chunk = audio_int16[start_idx:end_idx]

            # Process chunk
            probability, flags = self.vad.process(chunk)

            # Check if voice is detected (flags == 1 means voice detected)
            is_voice = flags == 1

            # Track state changes
            if is_voice and not voice_active:
                # Voice started
                voice_active = True
                current_start = i * hop_size
            elif not is_voice and voice_active:
                # Voice stopped
                voice_active = False
                current_end = i * hop_size

                # Convert to seconds and save segment
                start_sec = current_start / sample_rate
                end_sec = current_end / sample_rate
                raw_segments.append((start_sec, end_sec))

        # Handle case where voice is still active at end
        if voice_active and current_start is not None:
            end_sec = total_samples / sample_rate
            start_sec = current_start / sample_rate
            raw_segments.append((start_sec, end_sec))

        # Merge short segments into longer ones
        merged_segments = self._merge_short_segments(raw_segments)

        return merged_segments

    def _merge_short_segments(self, segments, max_gap=0.5):
        """
        Merge consecutive short segments into longer segments

        Args:
            segments: List of (start_sec, end_sec) tuples
            max_gap: Maximum gap between segments to merge (seconds)

        Returns:
            List of merged (start_sec, end_sec) tuples
        """
        if not segments:
            return []

        merged = []
        current_start, current_end = segments[0]

        for i in range(1, len(segments)):
            next_start, next_end = segments[i]
            gap = next_start - current_end

            # Merge if gap is small
            if gap <= max_gap:
                # Extend current segment
                current_end = next_end
            else:
                # Save current segment and start new one
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        # Add final segment
        merged.append((current_start, current_end))

        return merged


def main():
    parser = argparse.ArgumentParser(description="Segment audio using TEN VAD")
    parser.add_argument("--input", required=True, help="Input directory with WAV files")
    parser.add_argument("--output", required=True, help="Output directory for segments")
    parser.add_argument("--min-duration", type=float, default=4.0, help="Minimum segment duration (seconds)")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Maximum segment duration (seconds)")
    parser.add_argument("--metadata", default="data/tenvad_metadata.jsonl", help="Output metadata file")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize segmenter
    segmenter = TenVADSegmenter(
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )

    # Process all WAV files
    input_dir = Path(args.input)
    wav_files = sorted(input_dir.glob("*.wav"))

    if not wav_files:
        print(f"ERROR: No WAV files found in {input_dir}")
        return

    print(f"\n{'='*60}")
    print(f"TEN VAD Segmentation")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {args.output}")
    print(f"Duration filter: {args.min_duration}s - {args.max_duration}s")
    print(f"Total files: {len(wav_files)}")
    print(f"{'='*60}\n")

    # Process each file
    all_metadata = []
    total_segments = 0

    for audio_path in wav_files:
        try:
            segments = segmenter.segment_audio(str(audio_path), args.output)
            all_metadata.extend(segments)
            total_segments += len(segments)
            print(f"  → Generated {len(segments)} valid segments")
        except Exception as e:
            print(f"  ERROR processing {audio_path}: {e}")
            continue

    # Save metadata
    print(f"\n{'='*60}")
    print(f"Segmentation Complete")
    print(f"{'='*60}")
    print(f"Total segments generated: {total_segments}")
    print(f"Average segments per video: {total_segments/len(wav_files):.2f}")

    # Write metadata to JSONL
    os.makedirs(os.path.dirname(args.metadata), exist_ok=True)
    with open(args.metadata, 'w', encoding='utf-8') as f:
        for metadata in all_metadata:
            f.write(json.dumps(metadata) + '\n')

    print(f"Metadata saved to: {args.metadata}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
