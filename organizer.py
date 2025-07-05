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
    "hi_hat": "hi_hats",
    "snare": "snares",
    "clap": "claps",
    "vocal_loop": "vocal_loops",
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

    # Average onset strength – rough proxy for "transient-iness"
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    transient_strength = float(np.mean(onset_env))

    # Spectral rolloff
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # Harmonic-to-noise ratio (helps detect vocals)
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(harmonic**2))
    total_energy = float(np.sum(y**2))
    harmonic_ratio = harmonic_energy / (total_energy + 1e-9)

    # Spectral contrast (vocals have more variation)
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))

    # Pitch tracking (vocals have clear pitch)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_strength = float(np.mean(magnitudes))

    return {
        "duration": duration,
        "centroid_mean": centroid_mean,
        "low_energy_ratio": low_energy_ratio,
        "transient_strength": transient_strength,
        "spectral_rolloff": spectral_rolloff,
        "harmonic_ratio": harmonic_ratio,
        "spectral_contrast": spectral_contrast,
        "pitch_strength": pitch_strength,
    }


# -----------------------------
# Rule‑based classifier
# -----------------------------

def categorise(feats: dict[str, float]) -> str:
    """Return a label based on simple, interpretable heuristics."""
    # Kick / Bass one-shot
    if (
        feats["duration"] <= MAX_ONESHOT_DURATION
        and feats["low_energy_ratio"] >= BASS_ENERGY_RATIO_THRESHOLD
    ):
        return "kick_bass_oneshot"

    # Drum loop (long & transient-heavy)
    if (
        feats["duration"] > MAX_ONESHOT_DURATION
        and feats["transient_strength"] >= TRANSIENT_STRENGTH_THRESHOLD
    ):
        return "drum_loop"

    # Hi-hats: high centroid, high rolloff, short duration
    if (feats["centroid_mean"] > 3000 and 
        feats["spectral_rolloff"] > 8000 and 
        feats["duration"] < 1.0):
        return "hi_hat"
    
    # Snares: medium-high centroid, high transient, medium duration
    elif (feats["centroid_mean"] > 1500 and 
          feats["transient_strength"] > 0.4 and 
          0.5 < feats["duration"] < 2.0):
        return "snare"
    
    # Claps: very high transient, short duration
    elif (feats["transient_strength"] > 0.6 and 
          feats["duration"] < 0.5):
        return "clap"
    
    # Vocal loops: high harmonic content, clear pitch, longer duration
    elif (feats["harmonic_ratio"] > 0.6 and 
          feats["pitch_strength"] > 0.1 and 
          feats["duration"] > 1.0 and
          feats["spectral_contrast"] > 0.3):
        return "vocal_loop"

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
    import threading
    import wave
    import pyaudio
    import time

    # def play_audio(file_path, stop_event):
    #     print(f"[DEBUG] Starting playback for: {file_path}")
    #     try:
    #         wf = wave.open(str(file_path), 'rb')
    #         print(f"[DEBUG] Opened as WAV: {file_path}")
    #         stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
    #                          channels=wf.getnchannels(),
    #                          rate=wf.getframerate(),
    #                          output=True)
    #         data = wf.readframes(1024)
    #         while data and not stop_event.is_set():
    #             stream.write(data)
    #             data = wf.readframes(1024)
    #         stream.stop_stream()
    #         stream.close()
    #         wf.close()
    #     except Exception as e:
    #         print(f"[DEBUG] WAV playback failed: {e}. Trying librosa...")
    #         try:
    #             import librosa
    #             import numpy as np
    #             y, sr = librosa.load(str(file_path), sr=None, mono=False)
    #             # Normalize audio
    #             y = y / (np.max(np.abs(y)) + 1e-9)
    #             y = (y * 32767).astype(np.int16)
    #             # Ensure stereo
    #             if y.ndim == 1:
    #                 y = np.stack([y, y], axis=0)
    #             y = y.T  # shape (n_samples, 2)
    #             stream = pa.open(format=pyaudio.paInt16, channels=2, rate=sr, output=True)
    #             chunk_size = 4096
    #             idx = 0
    #             while idx < len(y) and not stop_event.is_set():
    #                 chunk = y[idx:idx+chunk_size]
    #                 stream.write(chunk.tobytes())
    #                 idx += chunk_size
    #             stream.stop_stream()
    #             stream.close()
    #         except Exception as e2:
    #             print(f"[ERROR] Could not play {file_path}: {e2}")

    headings = [
        "File",
        "Category",
        "Duration (s)",
        "Bass %",
        "Transient",
        "Harmonic %",
        "Pitch",
    ]
    data = [
        [
            str(p),
            c,
            f"{f['duration']:.2f}",
            f"{f['low_energy_ratio']*100:.1f}%",
            f"{f['transient_strength']:.2f}",
            f"{f['harmonic_ratio']*100:.1f}%",
            f"{f['pitch_strength']:.2f}",
        ]
        for p, c, f in records
    ]
    # pa = pyaudio.PyAudio()
    # current_play_thread = None
    # stop_event = threading.Event()
    table_elem = sg.Table(
        values=data,
        headings=headings,
        auto_size_columns=True,
        key='-TABLE-',
        enable_events=True,
        select_mode=sg.TABLE_SELECT_MODE_BROWSE,
        justification='left',
        row_height=20,
        enable_click_events=True,
        right_click_selects=True,
        expand_x=True,
        expand_y=True,
        num_rows=min(20, len(data)),
        background_color='white',
        text_color='black',
        alternating_row_color='#f0f0f0',
        selected_row_colors=('white', '#007acc'),
        tooltip='Double-click to play sample (DISABLED)',
    )
    layout = [
        [table_elem],
        [sg.Button("Close")]
    ]
    window = sg.Window("Sample Organiser", layout, resizable=True, use_default_focus=False, finalize=True)
    selected_index = None
    # playing_index = None
    last_click_time = 0
    last_click_row = None
    while True:
        event, values = window.read(timeout=100)
        print(f"[DEBUG] Event: {event}")
        if event in (sg.WIN_CLOSED, "Close"):
            # stop_event.set()
            # if current_play_thread and current_play_thread.is_alive():
            #     current_play_thread.join()
            break
        if event == '-TABLE-':
            if values['-TABLE-']:
                selected_index = values['-TABLE-'][0]
        if isinstance(event, tuple) and event[0] == '-TABLE-' and event[1] == '+CLICKED+':
            current_time = time.time()
            clicked_row = event[2][0]
            # Check if this is a double-click (same row within 500ms)
            if (last_click_row == clicked_row and 
                current_time - last_click_time < 0.5):
                # Double-click detected
                file_path = records[clicked_row][0]
                print(f"[DEBUG] Double-clicked row {clicked_row}: {file_path}")
                print(f"[INFO] Audio playback is currently disabled for safety")
                # Reset click tracking
                last_click_time = 0
                last_click_row = None
            else:
                # Single click - update tracking
                last_click_time = current_time
                last_click_row = clicked_row
        # if event == 'Stop':
        #     stop_event.set()
        #     if current_play_thread and current_play_thread.is_alive():
        #         current_play_thread.join()
        #     playing_index = None
        
        # Update table selection after processing all events
        # if playing_index is not None:
        #     try:
        #         window['-TABLE-'].update(select_rows=[playing_index])
        #     except Exception as e:
        #         print(f"[DEBUG] Table update failed: {e}")
    window.close()
    # pa.terminate()


# -----------------------------
# CLI entry-point
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
            print(f"⚠️ {path.name}: {exc}", file=sys.stderr)

    # CLI summary
    for p, c, feats in records:
        print(
            f"{p.name:<40} → {c:<18} | {feats['duration']:.2f}s | bass:{feats['low_energy_ratio']*100:.1f}% | trans:{feats['transient_strength']:.2f}"
        )

    if args.gui:
        launch_gui(records)


if __name__ == "__main__":
    main()
