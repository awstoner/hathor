#!/usr/bin/env python3
"""
Extracts audio features from all .mp3/.wav files in a directory and writes them to a CSV for manual labeling.
Automatically labels files based on keywords in their filenames.

Usage:
    python extract_features_to_csv.py --input-dir ./samples --output features_for_labeling.csv

Requirements:
    pip install librosa soundfile numpy pandas
"""
import argparse
import re
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import soundfile as sf  # noqa: F401

def get_label_from_filename(filename: str) -> str:
    """Extract label from filename based on keywords."""
    filename_lower = filename.lower()
    
    # Define keyword mappings with separate kick and 808 categories
    keyword_mappings = {
        'kick': 'kick',
        'bass': 'bass',  # Generic bass category
        '808': '808',
        'clap': 'clap',
        'snare': 'snare',
        'hihat': 'hi_hat',
        'hi-hat': 'hi_hat',
        'hi_hat': 'hi_hat',
        'hat': 'hi_hat',
        'vocal': 'vocal_loop',
        'voice': 'vocal_loop',
        'sing': 'vocal_loop',
        'loop': 'drum_loop',
        'drum': 'drum_loop',
        'percussion': 'drum_loop',
        'cymbal': 'hi_hat',
        'tom': 'snare',
        'rim': 'snare',
        'crash': 'hi_hat',
        'ride': 'hi_hat',
        'open': 'hi_hat',
        'closed': 'hi_hat',
    }
    
    # Check for keywords in filename
    for keyword, label in keyword_mappings.items():
        if keyword in filename_lower:
            return label
    
    # If no keywords found, return empty string for manual labeling
    return ''

def extract_features(file_path: Path) -> dict:
    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = stft[freqs <= 200, :]
    low_energy_ratio = float(np.sum(low_band) / (np.sum(stft) + 1e-9))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    transient_strength = float(np.mean(onset_env))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(harmonic**2))
    total_energy = float(np.sum(y**2))
    harmonic_ratio = harmonic_energy / (total_energy + 1e-9)
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_strength = float(np.mean(magnitudes))
    
    # Additional features for ML
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfccs))
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate = float(np.mean(zcr))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = float(np.mean(chroma_stft))
    
    return {
        'duration': duration,
        'centroid_mean': centroid_mean,
        'low_energy_ratio': low_energy_ratio,
        'transient_strength': transient_strength,
        'spectral_rolloff': spectral_rolloff,
        'harmonic_ratio': harmonic_ratio,
        'spectral_contrast': spectral_contrast,
        'pitch_strength': pitch_strength,
        'mfcc_mean': mfcc_mean,
        'rms_mean': rms_mean,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_flatness': spectral_flatness,
        'chroma_stft_mean': chroma_stft_mean,
    }

def main():
    parser = argparse.ArgumentParser(description="Extract audio features for ML labeling.")
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory with .mp3/.wav files')
    parser.add_argument('--output', type=Path, required=True, help='Output CSV file')
    args = parser.parse_args()

    audio_files = [*args.input_dir.rglob('*.mp3'), *args.input_dir.rglob('*.wav')]
    if not audio_files:
        print("No .mp3/.wav files found under input directory.")
        return

    # Sort files by name for easier labeling
    audio_files.sort(key=lambda x: x.name.lower())
    
    rows = []
    auto_labeled = 0
    manual_labeled = 0
    
    for path in audio_files:
        try:
            feats = extract_features(path)
            auto_label = get_label_from_filename(path.name)
            
            row = {'file_path': str(path)}
            row.update(feats)
            row['label'] = auto_label
            rows.append(row)
            
            if auto_label:
                print(f"Extracted: {path.name} → {auto_label} (auto-labeled)")
                auto_labeled += 1
            else:
                print(f"Extracted: {path.name} → (needs manual labeling)")
                manual_labeled += 1
                
        except Exception as exc:
            print(f"⚠️ {path.name}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    
    print(f"\nExported {len(rows)} samples to {args.output}")
    print(f"Auto-labeled: {auto_labeled}")
    print(f"Need manual labeling: {manual_labeled}")
    print(f"\nReview the CSV and update any incorrect auto-labels!")

if __name__ == "__main__":
    main() 