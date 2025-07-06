# Hathor

A powerful tool for analyzing, organizing, and classifying audio samples using machine learning.

## Features

- **Audio Analysis**: Extract 14 key audio features for comprehensive analysis
- **Machine Learning**: Pre-trained model for accurate classification
- **Modern GUI**: Beautiful interface built with CustomTkinter
- **Audio Playback**: Listen to samples directly in the GUI with play/stop controls
- **Smart Sorting**: Click any column header to sort by any feature
- **Search & Filter**: Find samples by filename or filter by category
- **Enhanced Features**: View brightness, energy, noise levels, and more
- **Confidence Scores**: See ML model confidence with visual progress bars
- **Export Capabilities**: Save results to CSV for further analysis
- **Colorblind-Friendly**: High-contrast colors for easy category identification

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/awstoner/hathor.git
cd hathor
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

## Quick Start

### Using the Pre-trained Model

The repository includes a pre-trained model that you can use immediately:

```bash
# Launch GUI to analyze and organize samples
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib --gui

# Organize files into categorized folders
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib --output-dir ./organized --move

# Analyze samples without moving files
python organizer.py --input-dir ./example_samples --model audio_classifier.joblib
```

## How It Works

### Audio Analysis
The system analyzes each audio file and extracts 14 comprehensive features:

**Basic Features:**
- **Duration**: Length in seconds
- **Bass %**: Low-frequency energy content
- **Transient**: Attack/punch strength
- **Harmonic %**: Musical vs noise content
- **Pitch**: Pitch clarity and strength

**Enhanced Features:**
- **Brightness**: How bright/harsh the sound is
- **Energy**: Overall loudness/energy level
- **Noise**: How noisy vs tonal the sound is

### Machine Learning Classification
- **Pre-trained Model**: Uses a Random Forest classifier trained on diverse audio samples
- **9 Categories**: kick, 808, bass, snare, hi_hat, clap, drum_loop, vocal_loop, other
- **Confidence Scores**: Each prediction includes a confidence percentage
- **Real-time Processing**: Fast classification for immediate results

### GUI Interface
- **Interactive Table**: View all samples with their features and classifications
- **Audio Playback**: Click any row to play the sample
- **Smart Sorting**: Click column headers to sort by any feature
- **Search & Filter**: Find specific samples or filter by category
- **Export**: Save results to CSV for further analysis

## Audio Categories

The system classifies audio into these categories:
- **kick**: Short, punchy kick drum samples
- **808**: Bass-heavy 808 samples with longer tails
- **bass**: Other bass samples and sub-bass
- **snare**: Snare drum samples with high transients
- **hi_hat**: High-frequency hi-hat and cymbal samples
- **clap**: Short, sharp clap and percussion samples
- **drum_loop**: Full drum loops and rhythmic patterns
- **vocal_loop**: Vocal samples with clear pitch and harmonic content
- **other**: Everything else that doesn't fit the above categories

## Repository Structure

```
hathor/
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

## Troubleshooting

### Common Issues

1. **"No .mp3/.wav files found"**
   - Check your input directory path
   - Ensure files have correct extensions

2. **Import errors**
   - Activate virtual environment: `source venv/bin/activate`
   - Reinstall dependencies: `pip install -r requirements.txt`

3. **ML model not found**
   - Ensure `audio_classifier.joblib` exists in the project directory
   - The model is included in the repository by default

### Performance Tips

- For large datasets (>1000 files), consider batch processing
- Use SSD storage for faster file operations
- Close other applications when running analysis

## Contributing

Feel free to:
- Add new audio features
- Improve the GUI interface
- Add audio playback functionality
- Enhance the ML model

## License

This project is open source. Use it for your audio organization needs!

---