"""
Speaker Verification Script using pyannote.audio
Verifies single speaker in audio segments
"""

import os
import sys
import argparse
import json
import shutil
import torch
import soundfile as sf
from pathlib import Path
from pyannote.audio import Pipeline


class SpeakerVerifier:
    """Single speaker verification using pyannote diarization"""

    def __init__(self, auth_token):
        """
        Initialize pyannote diarization pipeline

        Args:
            auth_token: HuggingFace authentication token
        """
        print("Loading pyannote speaker diarization model...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=auth_token
        )
        print("Speaker diarization model loaded successfully")

    def verify_single_speaker(self, audio_path):
        """
        Verify if audio contains only one speaker

        Args:
            audio_path: Path to audio file

        Returns:
            tuple: (is_single_speaker: bool, num_speakers: int, confidence: float)
        """
        # Pre-load audio with soundfile to avoid torchcodec issue
        waveform, sample_rate = sf.read(audio_path)

        # Convert to torch tensor (pyannote expects (channel, time) format)
        waveform_torch = torch.from_numpy(waveform).float()

        # Add channel dimension if mono
        if len(waveform_torch.shape) == 1:
            waveform_torch = waveform_torch.unsqueeze(0)
        elif len(waveform_torch.shape) == 2:
            # If stereo (time, channel), transpose to (channel, time)
            waveform_torch = waveform_torch.transpose(0, 1)

        # Create audio dictionary for pyannote
        audio_dict = {
            "waveform": waveform_torch,
            "sample_rate": sample_rate
        }

        # Run diarization
        diarization_output = self.pipeline(audio_dict)

        # Extract the Annotation object from DiarizeOutput
        diarization = diarization_output.speaker_diarization

        # Get unique speakers
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)

        num_speakers = len(speakers)
        is_single = num_speakers == 1

        # Calculate confidence based on speech duration per speaker
        if is_single:
            confidence = 1.0
        else:
            # Calculate how dominant the main speaker is
            speaker_durations = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

            # Confidence is the ratio of dominant speaker to total
            if speaker_durations:
                max_duration = max(speaker_durations.values())
                total_duration = sum(speaker_durations.values())
                confidence = max_duration / total_duration if total_duration > 0 else 0.0
            else:
                confidence = 0.0

        return is_single, num_speakers, confidence


def main():
    parser = argparse.ArgumentParser(description="Verify single speaker in audio segments")
    parser.add_argument("--input", required=True, help="Input directory with segments")
    parser.add_argument("--output", required=True, help="Output directory for verified segments")
    parser.add_argument("--token", required=True, help="HuggingFace authentication token")
    parser.add_argument("--metadata-in", required=True, help="Input metadata JSONL file")
    parser.add_argument("--metadata-out", default="data/verified_metadata.jsonl", help="Output metadata file")
    parser.add_argument("--min-confidence", type=float, default=0.95, help="Minimum speaker confidence threshold")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize verifier
    verifier = SpeakerVerifier(auth_token=args.token)

    # Load input metadata
    input_metadata = []
    with open(args.metadata_in, 'r', encoding='utf-8') as f:
        for line in f:
            input_metadata.append(json.loads(line.strip()))

    print(f"\n{'='*60}")
    print(f"Speaker Verification")
    print(f"{'='*60}")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Total segments: {len(input_metadata)}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"{'='*60}\n")

    # Verify each segment
    verified_metadata = []
    stats = {
        "total": len(input_metadata),
        "single_speaker": 0,
        "multiple_speakers": 0,
        "low_confidence": 0
    }

    for idx, metadata in enumerate(input_metadata, 1):
        filename = metadata["filename"]
        audio_path = os.path.join(args.input, filename)

        if not os.path.exists(audio_path):
            print(f"[{idx}/{len(input_metadata)}] WARNING: File not found: {filename}")
            continue

        print(f"[{idx}/{len(input_metadata)}] Verifying: {filename}")

        try:
            is_single, num_speakers, confidence = verifier.verify_single_speaker(audio_path)

            # Update metadata
            metadata["num_speakers"] = num_speakers
            metadata["speaker_confidence"] = float(confidence)
            metadata["single_speaker"] = is_single

            # Decision logic
            if is_single and confidence >= args.min_confidence:
                # Copy to verified directory
                output_path = os.path.join(args.output, filename)
                shutil.copy2(audio_path, output_path)
                verified_metadata.append(metadata)
                stats["single_speaker"] += 1
                print(f"  [PASS] Single speaker (confidence: {confidence:.2%})")
            elif not is_single:
                stats["multiple_speakers"] += 1
                print(f"  [FAIL] {num_speakers} speakers detected")
            else:
                stats["low_confidence"] += 1
                print(f"  [FAIL] Low confidence ({confidence:.2%})")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save verified metadata
    os.makedirs(os.path.dirname(args.metadata_out), exist_ok=True)
    with open(args.metadata_out, 'w', encoding='utf-8') as f:
        for metadata in verified_metadata:
            f.write(json.dumps(metadata) + '\n')

    # Print summary
    print(f"\n{'='*60}")
    print(f"Verification Complete")
    print(f"{'='*60}")
    print(f"Total segments: {stats['total']}")
    print(f"[OK] Single speaker: {stats['single_speaker']} ({stats['single_speaker']/stats['total']*100:.1f}%)")
    print(f"[FAIL] Multiple speakers: {stats['multiple_speakers']} ({stats['multiple_speakers']/stats['total']*100:.1f}%)")
    print(f"[FAIL] Low confidence: {stats['low_confidence']} ({stats['low_confidence']/stats['total']*100:.1f}%)")
    print(f"\nVerified segments saved to: {args.output}")
    print(f"Metadata saved to: {args.metadata_out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
