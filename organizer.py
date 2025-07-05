"""
Audio Sample Organizer

A utility to analyse .mp3/.wav samples, classify them with simple audio
heuristics (kick/bass oneshot, drum loop, other) and optionally move them
into category folders or present the results in a lightweight GUI.

Dependencies
------------
- librosa               # analysis
- soundfile             # audio I/O (librosa backend)
- numpy                 # numeric helpers
- PySimpleGUI (optional) # GUI mode

Install with:
    pip install librosa soundfile numpy PySimpleGUI

Usage examples
--------------
CLI report only:
    python organizer.py --input-dir ./samples

Move files into ./organized/<category>:
    python organizer.py --input-dir ./samples --output-dir ./organized --move

Launch a GUI inspector (no moves):
    python organizer.py --input-dir ./samples --gui
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf  # noqa: F401

try:
    import PySimpleGUI as sg

    GUI_AVAILABLE = True
except ImportError:  # pragma: no cover – GUI optional
    GUI_AVAILABLE = False

# -----------------------------
# Tunable categorisation rules
# -----------------------------
MAX_ONESHOT_DURATION = 2.0  # seconds
BASS_FREQUENCY_CUTOFF = 200  # Hz (energy below this counts as "bass")
BASS_ENERGY_RATIO_THRESHOLD = 0.30  # ≥ 30 % energy in bass region → bass‑heavy
TRANSIENT_STRENGTH_THRESHOLD = 0.30  # empirically chosen – tweak for your lib

CATEGORY_FOLDERS = {
    "kick_bass_oneshot": "kick_bass",
    "drum_loop": "drum_loops",
    "other": "other",
}

# -----------------------------
# Feature extraction helpers
# -----------------------------

def extract_features(file_path: Path) -> dict[str, float]:
    """Load *file_path* and compute a handful of lightweight descriptors."""
    y, sr = librosa.load(file_path, sr=None, mono=True)  # keep native SR

    # Duration (s)
    duration = librosa.get_duration(y=y, sr=sr)

    # Spectral centroid – proxy for perceived brightness
    centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Low‑end energy ratio (≤ *BASS_FREQUENCY_CUTOFF*)
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = stft[freqs <= BASS_FREQUENCY_CUTOFF, :]
    low_energy_ratio = float(np.sum(low_band) / (np.sum(stft) + 1e-9))

    # Average onset strength – rough proxy for „transient‑iness“
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    transient_strength = float(np.mean(onset_env))

    return {
        "duration": duration,
        "centroid_mean": centroid_mean,
        "low_energy_ratio": low_energy_ratio,
        "transient_strength": transient_strength,
    }


# -----------------------------
# Rule‑based classifier
# -----------------------------

def categorise(feats: dict[str, float]) -> str:
    """Return a label based on simple, interpretable heuristics."""
    # Kick / Bass one‑shot
    if (
        feats["duration"] <= MAX_ONESHOT_DURATION
        and feats["low_energy_ratio"] >= BASS_ENERGY_RATIO_THRESHOLD
    ):
        return "kick_bass_oneshot"

    # Drum loop (long & transient‑heavy)
    if (
        feats["duration"] > MAX_ONESHOT_DURATION
        and feats["transient_strength"] >= TRANSIENT_STRENGTH_THRESHOLD
    ):
        return "drum_loop"

    # Fallback
    return "other"


# -----------------------------
# I/O helpers
# -----------------------------

def maybe_move(src: Path, dest_root: Path, category: str) -> None:
    """Move *src* into *dest_root/category/*. Keeps directory flat."""
    dest = dest_root / CATEGORY_FOLDERS[category]
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dest / src.name)


# -----------------------------
# (Optional) Tiny GUI
# -----------------------------

def launch_gui(records):
    headings = [
        "File",
        "Category",
        "Duration (s)",
        "Bass %",
        "Transient",
    ]
    data = [
        [
            str(p),
            c,
            f"{f['duration']:.2f}",
            f"{f['low_energy_ratio']*100:.1f}%",
            f"{f['transient_strength']:.2f}",
        ]
        for p, c, f in records
    ]
    layout = [[sg.Table(values=data, headings=headings, auto_size_columns=True)], [sg.Button("Close")]]
    window = sg.Window("Sample Organiser", layout, resizable=True)
    while True:
        event, _ = window.read()
        if event in (sg.WIN_CLOSED, "Close"):
            break
    window.close()


# -----------------------------
# CLI entry‑point
# -----------------------------

def main():  # noqa: C901 – simple script
    parser = argparse.ArgumentParser(description="Analyse & organise audio samples")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with .mp3/.wav files")
    parser.add_argument("--output-dir", type=Path, help="Destination root for organised samples")
    parser.add_argument("--move", action="store_true", help="Actually move files instead of reporting only")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (requires PySimpleGUI)")
    args = parser.parse_args()

    if args.gui and not GUI_AVAILABLE:
        sys.exit("PySimpleGUI not installed. Install via: pip install PySimpleGUI")

    audio_files = [*args.input_dir.rglob("*.mp3"), *args.input_dir.rglob("*.wav")]
    if not audio_files:
        sys.exit("No .mp3/.wav files found under input directory.")

    records = []  # (path, category, feats)
    for path in audio_files:
        try:
            feats = extract_features(path)
            label = categorise(feats)
            records.append((path, label, feats))
            if args.move and args.output_dir:
                maybe_move(path, args.output_dir, label)
        except Exception as exc:  # broad – quick script
            print(f"⚠️ {path.name}: {exc}", file=sys.stderr)

    # CLI summary
    for p, c, feats in records:
        print(
            f"{p.name:<40} → {c:<18} | {feats['duration']:.2f}s | bass:{feats['low_energy_ratio']*100:.1f}% | trans:{feats['transient_strength']:.2f}"
        )

    if args.gui:
        launch_gui(records)


if __name__ == "__main__":
    main()
