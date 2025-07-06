#!/usr/bin/env python3
"""
Test script for the pre-trained ML model.

This script helps users verify that the ML model is working correctly
with their audio samples.
"""

import sys
from pathlib import Path
from organizer import extract_features, use_model

def test_ml_model():
    """Test the pre-trained ML model with example samples."""
    
    # Check if model files exist
    model_path = Path("audio_classifier.joblib")
    metadata_path = Path("audio_classifier_metadata.joblib")
    
    if not model_path.exists():
        print("❌ Error: audio_classifier.joblib not found!")
        print("Make sure you're in the project root directory.")
        return False
    
    if not metadata_path.exists():
        print("❌ Error: audio_classifier_metadata.joblib not found!")
        print("Make sure you're in the project root directory.")
        return False
    
    print("✅ Found pre-trained model files")
    
    # Check for example samples
    example_dir = Path("example_samples")
    if not example_dir.exists():
        print("❌ Error: example_samples directory not found!")
        print("Please add some .wav or .mp3 files to the example_samples directory.")
        return False
    
    # Find audio files
    audio_files = list(example_dir.glob("*.wav")) + list(example_dir.glob("*.mp3"))
    
    if not audio_files:
        print("❌ Error: No audio files found in example_samples directory!")
        print("Please add some .wav or .mp3 files for testing.")
        return False
    
    print(f"✅ Found {len(audio_files)} audio files for testing")
    
    # Extract features from all files
    print("\n📊 Extracting features...")
    records = []
    for file_path in audio_files:
        try:
            features = extract_features(file_path)
            records.append((file_path, "unknown", features))
            print(f"  ✓ {file_path.name}")
        except Exception as e:
            print(f"  ❌ {file_path.name}: {e}")
    
    if not records:
        print("❌ Error: Could not extract features from any files!")
        return False
    
    # Use ML model for classification
    print("\n🤖 Running ML classification...")
    try:
        ml_records = use_model(records, str(model_path))
        
        print("\n📋 Classification Results:")
        print("-" * 80)
        print(f"{'File':<30} {'Category':<15} {'Confidence':<10}")
        print("-" * 80)
        
        for file_path, category, features, confidence in ml_records:
            file_name = file_path.name[:29] + "..." if len(file_path.name) > 30 else file_path.name
            conf_str = f"{confidence*100:.1f}%" if confidence else "N/A"
            print(f"{file_name:<30} {category:<15} {conf_str:<10}")
        
        print("-" * 80)
        print("✅ ML model test completed successfully!")
        print("\n💡 Tip: Run 'python organizer.py --input-dir ./example_samples --model audio_classifier.joblib --gui'")
        print("      to see results in the interactive GUI.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during ML classification: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Pre-trained ML Model")
    print("=" * 50)
    
    success = test_ml_model()
    
    if success:
        print("\n🎉 All tests passed! Your ML model is ready to use.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1) 