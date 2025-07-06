# Audio Sample Organizer

A powerful tool for analyzing, organizing, and classifying audio samples using machine learning.

## Features

- **Audio Analysis**: Extract 14 key audio features for comprehensive analysis
- **Machine Learning**: Pre-trained model for accurate classification
- **Modern GUI**: Beautiful interface built with CustomTkinter
- **Audio Playback**: Listen to samples directly in the GUI with play/stop controls
- **Smart Sorting**: Click any column header to sort by that feature
- **Search & Filter**: Find samples by filename or filter by category
- **Enhanced Features**: View brightness, energy, noise levels, and more
- **Confidence Scores**: See ML model confidence with visual progress bars
- **Export Capabilities**: Save results to CSV for further analysis
- **Colorblind-Friendly**: High-contrast colors for easy category identification

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

## Repository Structure

```
audio-sample-organizer/
├── organizer.py                 # Main application
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

### Using the Pre-trained Model

The repository includes a pre-trained model that you can use immediately:

```bash
# Use the included ML model with GUI
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib --gui

# Use the included ML model for file organization
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib --output-dir ./organized --move

# Use the included ML model for analysis only
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib
```

## Audio Categories

The system can classify audio into these categories:
- **kick**: Short, punchy kick drum samples
- **808**: Bass-heavy 808 samples with longer tails
- **bass**: Other bass samples and sub-bass
- **snare**: Snare drum samples with high transients
- **hi_hat**: High-frequency hi-hat and cymbal samples
- **clap**: Short, sharp clap and percussion samples
- **drum_loop**: Full drum loops and rhythmic patterns
- **vocal_loop**: Vocal samples with clear pitch and harmonic content
- **other**: Everything else that doesn't fit the above categories

## Features Extracted

The system extracts 14 comprehensive audio features:

### **Basic Features**
1. **duration**: Length in seconds
2. **centroid_mean**: Perceived brightness
3. **low_energy_ratio**: Bass frequency content
4. **transient_strength**: Percussive/attack characteristics
5. **spectral_rolloff**: High-frequency cutoff
6. **harmonic_ratio**: Harmonic vs noise content
7. **spectral_contrast**: Frequency variation
8. **pitch_strength**: Pitch clarity

### **Enhanced Features**
9. **mfcc_mean**: Mel-frequency cepstral coefficients
10. **rms_mean**: Root mean square energy
11. **zero_crossing_rate**: Waveform complexity
12. **spectral_bandwidth**: Frequency spread
13. **spectral_flatness**: Noise vs tonal content
14. **chroma_stft_mean**: Musical note characteristics

## Machine Learning

### Training Process

The `train_audio_classifier.py` script:
- Uses Random Forest classifier
- Performs hyperparameter tuning with GridSearchCV
- Provides detailed performance metrics
- Creates visualizations of results
- Saves trained model and metadata

### Model Performance

The included pre-trained model was trained on a diverse dataset of audio samples and provides:
- **Accurate classification** across all audio categories
- **Confidence scores** for each prediction
- **Fast processing** for real-time classification

See `audio_classifier_results.png` for detailed performance metrics.



## GUI Features

The CustomTkinter GUI provides a comprehensive interface for audio sample analysis:

### **Interactive Table**
- **Scrollable table** with all analysis results
- **Clickable column headers** for sorting by any feature
- **Row highlighting** during audio playback
- **Color-coded categories** for easy identification

### **Audio Analysis Features**
- **Duration**: Length of each sample
- **Bass %**: Low-frequency energy content
- **Transient**: Attack/punch strength
- **Harmonic %**: Musical vs noise content
- **Pitch**: Pitch clarity and strength
- **Brightness**: How bright/harsh the sound is
- **Energy**: Overall loudness/energy level
- **Noise**: How noisy vs tonal the sound is
- **Confidence**: ML model confidence with progress bars

### **Search & Organization**
- **Real-time search** by filename
- **Category filtering** dropdown
- **Statistics summary** showing file counts per category
- **Export to CSV** functionality

### **Audio Playback**
- **Play/Stop controls** for listening to samples
- **Auto-stop** when playback finishes
- **Visual feedback** during playback

### **Accessibility**
- **Dark mode** interface
- **Colorblind-friendly** category colors
- **High-contrast** design for easy reading

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