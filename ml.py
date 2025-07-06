"""
Machine learning functionality for Hathor.

This module contains functions for loading and using trained ML models
for audio classification.
"""

import joblib
import numpy as np
import sys
from pathlib import Path

def use_model(records, model_path):
    """Use trained ML model for classification."""
    try:
        import joblib
        import pandas as pd
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False

    if not ML_AVAILABLE:
        sys.exit("scikit-learn not installed. Install via: pip install scikit-learn")

    try:
        # Load both model and metadata
        model = joblib.load(model_path)
        metadata = joblib.load(str(model_path).replace('.joblib', '_metadata.joblib'))
        label_encoder = metadata['label_encoder']
        print(f"Loaded model from {model_path}")
    except Exception as e:
        sys.exit(f"Failed to load model: {e}")

    # Extract features for prediction
    features_list = []
    for _, features in records:
        feature_vector = [
            features['duration'],
            features['centroid_mean'],
            features['low_energy_ratio'],
            features['transient_strength'],
            features['spectral_rolloff'],
            features['harmonic_ratio'],
            features['spectral_contrast'],
            features['pitch_strength'],
            features['mfcc_mean'],
            features['rms_mean'],
            features['zero_crossing_rate'],
            features['spectral_bandwidth'],
            features['spectral_flatness'],
            features['chroma_stft_mean'],
        ]
        features_list.append(feature_vector)

    # Predict
    predictions_encoded = model.predict(features_list)
    proba = model.predict_proba(features_list)

    # Decode predictions back to category names
    predictions = label_encoder.inverse_transform(predictions_encoded)

    # Get confidence for each prediction
    confidences = [float(np.max(p)) for p in proba]

    # Update records with ML predictions and confidence
    ml_records = []
    for i, (file_path, features) in enumerate(records):
        ml_records.append((file_path, predictions[i], features, confidences[i]))

    # Print results
    print("\nML Model Predictions:")
    for p, c, feats, conf in ml_records:
        print(
            f"{p.name:<40} → {c:<18} | {feats['duration']:.2f}s | bass:{feats['low_energy_ratio']*100:.1f}% | trans:{feats['transient_strength']:.2f} | conf:{conf*100:.1f}%"
        )

    return ml_records

# Add any other ML helpers here 