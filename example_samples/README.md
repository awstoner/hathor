# Example Samples

This directory is for example audio files to test the Audio Sample Organizer.

## What to add here:

- A few `.wav` or `.mp3` files for testing
- Mix of different audio types (kicks, snares, loops, etc.)
- Keep files small (< 1MB each) for repository size

## Example structure:
```
example_samples/
├── kick_sample.wav
├── snare_sample.wav
├── hihat_sample.wav
├── vocal_loop.wav
└── drum_loop.wav
```

## Usage:
```bash
python organizer.py --input-dir ./example_samples --gui
```

**Note:** Don't add large audio libraries here - this is just for testing the tool functionality. 