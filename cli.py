"""
Command-line interface for Hathor.

This module contains the CLI argument parsing and main entry point
for the Hathor audio analysis application.
"""

import argparse
import sys
from pathlib import Path
from features import extract_features
from ml import use_model
from gui import launch_gui

def main():  # noqa: C901 – simple script
    parser = argparse.ArgumentParser(description="Hathor - Analyse & organise audio samples using ML classification")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with .mp3/.wav files")
    parser.add_argument("--output-dir", type=Path, help="Destination root for organised samples")
    parser.add_argument("--move", action="store_true", help="Actually move files instead of reporting only")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (requires customtkinter)")
    parser.add_argument("--model", type=Path, default="audio_classifier.joblib", help="ML model path (default: audio_classifier.joblib)")
    args = parser.parse_args()

    # Check for GUI availability
    try:
        import customtkinter as ctk
        GUI_AVAILABLE = True
    except ImportError:
        GUI_AVAILABLE = False

    if args.gui and not GUI_AVAILABLE:
        sys.exit("customtkinter not installed. Install via: pip install customtkinter")

    audio_files = [*args.input_dir.rglob("*.mp3"), *args.input_dir.rglob("*.wav")]
    if not audio_files:
        sys.exit("No .mp3/.wav files found under input directory.")

    records = []  # (path, category, feats)
    for path in audio_files:
        try:
            feats = extract_features(path)
            records.append((path, feats))
        except Exception as exc:  # broad – quick script
            print(f"⚠️ {path.name}: {exc}", file=sys.stderr)

    # CLI summary
    for p, feats in records:
        print(
            f"{p.name:<40} | {feats['duration']:.2f}s | bass:{feats['low_energy_ratio']*100:.1f}% | trans:{feats['transient_strength']:.2f}"
        )

    # Use ML model for classification
    ml_records = use_model(records, args.model)

    # Launch GUI if requested
    if args.gui:
        launch_gui(ml_records)

if __name__ == "__main__":
    main() 