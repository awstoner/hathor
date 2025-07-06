# Audio Sample Organizer with ML Classification

A powerful tool for analyzing, organizing, and classifying audio samples using both rule-based heuristics and machine learning.

## Features

- **Audio Analysis**: Extract 8 key audio features (duration, spectral centroid, bass energy, etc.)
- **Rule-based Classification**: Quick categorization using interpretable heuristics
- **Machine Learning**: Train custom models for improved accuracy
- **Modern GUI**: Beautiful interface built with CustomTkinter
- **Export Capabilities**: Save results to CSV for further analysis

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/audio-sample-organizer.git
cd audio-sample-organizer
```

2. **Setup virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Test the ML model (optional):**
```bash
python test_ml_model.py
```

## Repository Structure

```
audio-sample-organizer/
├── organizer.py                 # Main application
├── train_audio_classifier.py    # ML training script
├── extract_features_to_csv.py   # Feature extraction utility
├── test_ml_model.py            # Test script for ML model
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── audio_classifier.joblib      # Pre-trained ML model
├── audio_classifier_metadata.joblib  # Model metadata
├── audio_classifier_results.png # Model performance results
├── example_samples/             # Test audio files (add your own)
│   └── README.md               # Instructions for test files
└── venv/                       # Virtual environment (excluded from git)
```

## Quick Start

### Basic Usage (Rule-based Classification)

```bash
# Analyze samples without moving files
python organizer.py --input-dir ./example_samples

# Move files into organized folders
python organizer.py --input-dir ./example_samples --output-dir ./organized --move

# Launch GUI to inspect results
python organizer.py --input-dir ./example_samples --gui
```

### Machine Learning Pipeline

#### Option 1: Use Pre-trained Model (Recommended for Quick Start)
The repository includes a pre-trained model that you can use immediately:

```bash
# Use the included ML model with GUI
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib --gui

# Use the included ML model for file organization
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib --output-dir ./organized --move
```

#### Option 2: Train Your Own Model
If you want to train on your own data:

1. **Export features for training:**
```bash
python organizer.py --input-dir ./example_samples --export-features features.csv
```

2. **Train ML model:**
```bash
python train_audio_classifier.py features.csv
```

3. **Use your trained model:**
```bash
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib
```

## Audio Categories

The system can classify audio into these categories:
- **kick_bass_oneshot**: Short bass-heavy samples (≤2s, ≥30% bass)
- **drum_loop**: Longer rhythmic patterns (>2s, high transients)
- **hi_hat**: High-frequency, short samples
- **snare**: Medium-high frequency, high transient samples
- **clap**: Very short, high transient samples
- **vocal_loop**: Harmonic content with clear pitch
- **other**: Everything else

## Features Extracted

1. **duration**: Length in seconds
2. **centroid_mean**: Perceived brightness
3. **low_energy_ratio**: Bass frequency content
4. **transient_strength**: Percussive/attack characteristics
5. **spectral_rolloff**: High-frequency cutoff
6. **harmonic_ratio**: Harmonic vs noise content
7. **spectral_contrast**: Frequency variation
8. **pitch_strength**: Pitch clarity

## Machine Learning

### Training Process

The `train_audio_classifier.py` script:
- Uses Random Forest classifier
- Performs hyperparameter tuning with GridSearchCV
- Provides detailed performance metrics
- Creates visualizations of results
- Saves trained model and metadata

### Model Performance

The included pre-trained model was trained on a diverse dataset of audio samples and achieves:
- **High accuracy** across all audio categories
- **Robust classification** for kicks, 808s, snares, hi-hats, and more
- **Fast inference** for real-time classification

The training script outputs:
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Visual representation of predictions
- **Feature Importance**: Which features matter most
- **Accuracy Score**: Overall model performance

See `audio_classifier_results.png` for detailed performance metrics.

### Files Generated

After training, you'll get:
- `audio_classifier.joblib`: Trained model
- `audio_classifier_metadata.joblib`: Model metadata
- `audio_classifier_results.png`: Performance visualizations

**Note:** The repository already includes these files from our pre-trained model, so you can start using ML classification immediately without training.

## GUI Features

The CustomTkinter GUI provides:
- **Scrollable table** with all analysis results
- **Color-coded categories** for easy identification
- **Statistics summary** showing file counts per category
- **Export to CSV** functionality
- **Dark mode** interface

## Customization

### Adjusting Classification Rules

Edit the constants in `organizer.py`:
```python
MAX_ONESHOT_DURATION = 2.0  # seconds
BASS_FREQUENCY_CUTOFF = 200  # Hz
BASS_ENERGY_RATIO_THRESHOLD = 0.30
TRANSIENT_STRENGTH_THRESHOLD = 0.30
```

### Adding New Categories

1. Add new rules to the `categorise()` function
2. Update `CATEGORY_FOLDERS` dictionary
3. Add color coding in the GUI

## Troubleshooting

### Common Issues

1. **"No .mp3/.wav files found"**
   - Check your input directory path
   - Ensure files have correct extensions

2. **Import errors**
   - Activate virtual environment: `source venv/bin/activate`
   - Reinstall dependencies: `pip install -r requirements.txt`

3. **Poor classification accuracy**
   - Export features and manually review labels
   - Retrain model with corrected data
   - Adjust feature extraction parameters

### Performance Tips

- For large datasets (>1000 files), consider batch processing
- Use SSD storage for faster file operations
- Close other applications when running analysis

## Contributing

Feel free to:
- Add new audio features
- Implement different ML algorithms
- Improve the GUI interface
- Add audio playback functionality

## License

This project is open source. Use it for your audio organization needs!

---