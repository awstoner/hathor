#!/usr/bin/env python3
"""
Audio Classifier Training Script

Trains a machine learning model to classify audio samples based on extracted features.
Uses Random Forest classifier with cross-validation and provides detailed performance metrics.

Usage:
    python train_audio_classifier.py features.csv

Requirements:
    pip install scikit-learn pandas numpy matplotlib seaborn
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(csv_path):
    """Load and prepare the dataset for training."""
    print(f"Loading data from {csv_path}...")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Warning: Found missing values:")
        print(missing[missing > 0])
        df = df.dropna()
        print(f"After dropping missing values: {len(df)} samples")
    
    # Separate features and labels
    feature_columns = [
        'duration', 'centroid_mean', 'low_energy_ratio', 'transient_strength',
        'spectral_rolloff', 'harmonic_ratio', 'spectral_contrast', 'pitch_strength',
        'mfcc_mean', 'rms_mean', 'zero_crossing_rate', 'spectral_bandwidth',
        'spectral_flatness', 'chroma_stft_mean'
    ]
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    X = df[feature_columns]
    y = df['label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels: {label_encoder.classes_}")
    print(f"Label distribution:")
    for label, count in zip(label_encoder.classes_, np.bincount(y_encoded)):
        print(f"  {label}: {count}")
    
    return X, y_encoded, label_encoder, feature_columns

def train_model(X, y, feature_names):
    """Train the Random Forest model with hyperparameter tuning."""
    print("\nTraining Random Forest model...")
    
    # Check dataset size
    if len(X) < 10:
        print(f"Warning: Small dataset ({len(X)} samples). Results may not be reliable.")
        print("Consider collecting more samples for better model performance.")
    
    # Check if we can use stratified splitting
    min_samples_per_class = min(np.bincount(y))
    if min_samples_per_class < 2:
        print(f"Warning: Some classes have fewer than 2 samples. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Adjust cross-validation based on dataset size
    if len(X_train) < 5:
        cv_folds = 2
        print(f"Using {cv_folds}-fold cross-validation due to small dataset")
    else:
        cv_folds = min(5, len(X_train))
        print(f"Using {cv_folds}-fold cross-validation")
    
    # Simplified parameter grid for small datasets
    if len(X_train) < 10:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    else:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Initialize the model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform grid search
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return best_model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model and print detailed metrics."""
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return accuracy, y_pred, cm

def plot_results(model, X_test, y_test, y_pred, cm, label_encoder, feature_names):
    """Create visualization plots."""
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Audio Classifier Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix Heatmap
    ax1 = axes[0, 0]
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax1
    )
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. Feature Importance
    ax2 = axes[0, 1]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax2.bar(range(len(importances)), importances[indices])
    ax2.set_title('Feature Importance')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Importance')
    ax2.set_xticks(range(len(importances)))
    ax2.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    # 3. Prediction Distribution
    ax3 = axes[1, 0]
    unique, counts = np.unique(y_pred, return_counts=True)
    ax3.bar(unique, counts)
    ax3.set_title('Prediction Distribution')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Count')
    ax3.set_xticks(unique)
    ax3.set_xticklabels([label_encoder.classes_[i] for i in unique], rotation=45)
    
    # 4. Actual vs Predicted
    ax4 = axes[1, 1]
    ax4.scatter(y_test, y_pred, alpha=0.6)
    ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax4.set_xlabel('Actual')
    ax4.set_ylabel('Predicted')
    ax4.set_title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig('audio_classifier_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to 'audio_classifier_results.png'")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass

def save_model(model, label_encoder, feature_names, accuracy):
    """Save the trained model and metadata."""
    print("\nSaving model...")
    
    # Save the model
    model_filename = 'audio_classifier.joblib'
    joblib.dump(model, model_filename)
    
    # Save metadata
    metadata = {
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'accuracy': accuracy,
        'model_type': 'RandomForestClassifier',
        'parameters': model.get_params()
    }
    joblib.dump(metadata, 'audio_classifier_metadata.joblib')
    
    print(f"Model saved as '{model_filename}'")
    print(f"Metadata saved as 'audio_classifier_metadata.joblib'")
    print(f"Model accuracy: {accuracy:.3f}")

def main():
    """Main training pipeline."""
    if len(sys.argv) != 2:
        print("Usage: python train_audio_classifier.py <features.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        # Load and prepare data
        X, y, label_encoder, feature_names = load_and_prepare_data(csv_path)
        
        # Train model
        model, X_train, X_test, y_train, y_test = train_model(X, y, feature_names)
        
        # Evaluate model
        accuracy, y_pred, cm = evaluate_model(model, X_test, y_test, label_encoder)
        
        # Create visualizations
        plot_results(model, X_test, y_test, y_pred, cm, label_encoder, feature_names)
        
        # Save model
        save_model(model, label_encoder, feature_names, accuracy)
        
        print("\nTraining completed successfully!")
        print("You can now use the model with:")
        print("python organizer.py --input-dir ./samples --model audio_classifier.joblib")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 