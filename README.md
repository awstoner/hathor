# 🥁 Audio Sample Organizer

This project analyzes .mp3/.wav samples, classifies them using simple audio features (like duration, bass content, and transients), and helps you sort them — either automatically into folders, or visually using a lightweight GUI.

---

## 📁 Usage Examples

### Basic Analysis (No File Moves)

```bash
python organizer.py --input-dir ./samples
```

### Sort Files Into Category Folders

```bash
python organizer.py --input-dir ./samples --output-dir ./organized --move
```

### GUI Preview (Optional)

```bash
python organizer.py --input-dir ./samples --gui
```

> The GUI lets you inspect classification results without moving files.

---

## 🧠 How It Works

* Uses `librosa` to extract audio features:

  * Duration
  * Bass frequency energy
  * Transient strength
* Applies rule-based classification to label samples:

  * Kick/Bass One‑Shots
  * Drum Loops
  * Other

---

## 🛠 Customize It

* Tweak thresholds at the top of `organizer.py` to better fit your sample library
* Expand `categorise()` with more rules or ML models
* Upgrade the GUI or integrate into a DAW workflow

---

## ✅ To Do

* [ ] Add support for FLAC / AIFF
* [ ] Train a ML model based on labeled folders
* [ ] Drag-and-drop GUI?

---

## 📄 License

MIT

---