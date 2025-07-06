"""
Audio feature extraction for Hathor.

This module contains functions for extracting audio features from audio files
for machine learning classification.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# -----------------------------
# Tunable categorisation rules
# -----------------------------
MAX_ONESHOT_DURATION = 2.0  # seconds
BASS_FREQUENCY_CUTOFF = 200  # Hz (energy below this counts as "bass")
BASS_ENERGY_RATIO_THRESHOLD = 0.30  # ≥ 30 % energy in bass region → bass‑heavy
TRANSIENT_STRENGTH_THRESHOLD = 0.30  # empirically chosen – tweak for your lib

CATEGORY_FOLDERS = {
    "kick": "kicks",
    "808": "808s",
    "bass": "bass",
    "drum_loop": "drum_loops",
    "hi_hat": "hi_hats",
    "snare": "snares",
    "clap": "claps",
    "vocal_loop": "vocal_loops",
    "other": "other",
}

def extract_features(file_path: Path) -> dict[str, float]:
    """Load *file_path* and compute a set of audio descriptors for ML."""
    y, sr = librosa.load(file_path, sr=None, mono=True)  # keep native SR

    # Duration (s)
    duration = librosa.get_duration(y=y, sr=sr)

    # Spectral centroid – proxy for perceived brightness
    centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Low‑end energy ratio (≤ *BASS_FREQUENCY_CUTOFF*)
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = stft[freqs <= BASS_FREQUENCY_CUTOFF, :]
    low_energy_ratio = float(np.sum(low_band) / (np.sum(stft) + 1e-9))

    # Average onset strength – rough proxy for "transient-iness"
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

    # --- Additional features for ML ---
    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfccs))

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate = float(np.mean(zcr))

    # Spectral bandwidth
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # Spectral flatness
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = float(np.mean(chroma_stft))

    return {
        "duration": duration,
        "centroid_mean": centroid_mean,
        "low_energy_ratio": low_energy_ratio,
        "transient_strength": transient_strength,
        "spectral_rolloff": spectral_rolloff,
        "harmonic_ratio": harmonic_ratio,
        "spectral_contrast": spectral_contrast,
        "pitch_strength": pitch_strength,
        "mfcc_mean": mfcc_mean,
        "rms_mean": rms_mean,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_flatness": spectral_flatness,
        "chroma_stft_mean": chroma_stft_mean,
    }

# Add any other helper functions here 