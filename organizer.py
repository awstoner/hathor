"""
Audio Sample Organizer

A utility to analyse .mp3/.wav samples, classify them with simple audio
heuristics (kick/bass oneshot, drum loop, other) and optionally move them
into category folders or present the results in a lightweight GUI.

Dependencies
------------
- librosa               # analysis
- soundfile             # audio I/O (librosa backend)
- numpy                 # numeric helpers
- customtkinter         # modern GUI
- scikit-learn          # ML models
- pandas                # data handling

Install with:
    pip install librosa soundfile numpy customtkinter scikit-learn pandas

Usage examples
--------------
CLI report only:
    python organizer.py --input-dir ./samples

Move files into ./organized/<category>:
    python organizer.py --input-dir ./samples --output-dir ./organized --move

Launch a GUI inspector (no moves):
    python organizer.py --input-dir ./samples --gui

Export features for ML training:
    python organizer.py --input-dir ./samples --export-features features.csv

Use ML model for classification:
    python organizer.py --input-dir ./samples --model audio_classifier.joblib
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf  # noqa: F401

try:
    import customtkinter as ctk
    from tkinter import ttk
    import tkinter as tk
    GUI_AVAILABLE = True
except ImportError:  # pragma: no cover – GUI optional
    GUI_AVAILABLE = False

try:
    import pandas as pd
    import joblib
    ML_AVAILABLE = True
except ImportError:  # pragma: no cover – ML optional
    ML_AVAILABLE = False

# -----------------------------
# Tunable categorisation rules
# -----------------------------
MAX_ONESHOT_DURATION = 2.0  # seconds
BASS_FREQUENCY_CUTOFF = 200  # Hz (energy below this counts as "bass")
BASS_ENERGY_RATIO_THRESHOLD = 0.30  # ≥ 30 % energy in bass region → bass‑heavy
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

# -----------------------------
# Feature extraction helpers
# -----------------------------

def extract_features(file_path: Path) -> dict[str, float]:
    """Load *file_path* and compute a set of audio descriptors for ML."""
    y, sr = librosa.load(file_path, sr=None, mono=True)  # keep native SR

    # Duration (s)
    duration = librosa.get_duration(y=y, sr=sr)

    # Spectral centroid – proxy for perceived brightness
    centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Low‑end energy ratio (≤ *BASS_FREQUENCY_CUTOFF*)
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = stft[freqs <= BASS_FREQUENCY_CUTOFF, :]
    low_energy_ratio = float(np.sum(low_band) / (np.sum(stft) + 1e-9))

    # Average onset strength – rough proxy for "transient-iness"
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


# -----------------------------
# I/O helpers
# -----------------------------

def maybe_move(src: Path, dest_root: Path, category: str) -> None:
    """Move *src* into *dest_root/category/*. Keeps directory flat."""
    dest = dest_root / CATEGORY_FOLDERS[category]
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dest / src.name)


# -----------------------------
# Modern GUI with CustomTkinter
# -----------------------------

def launch_gui(records):
    """Launch a modern GUI using CustomTkinter."""
    
    # Set modern appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    app = ctk.CTk()
    app.title("Hathor - Audio Sample Organizer")
    app.geometry("1600x900")
    app.minsize(1200, 700)
    
    # Audio playback variables
    current_audio = None
    playing_row = None
    pygame_initialized = False
    
    # Sorting variables
    sort_column = None
    sort_reverse = False
    
    # Create modern gradient background
    main_frame = ctk.CTkFrame(app, fg_color=("gray95", "gray10"))
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Modern title with Hathor branding
    title_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    title_frame.pack(fill="x", pady=(0, 20))
    
    title_label = ctk.CTkLabel(
        title_frame, 
        text="𓃭 Hathor", 
        font=ctk.CTkFont(size=36, weight="bold"),
        text_color=("#FFD700", "#FFA500")  # Egyptian gold
    )
    title_label.pack(side="left", padx=(0, 10))
    
    subtitle_label = ctk.CTkLabel(
        title_frame,
        text="Audio Sample Organizer",
        font=ctk.CTkFont(size=18, weight="normal"),
        text_color=("gray40", "gray60")
    )
    subtitle_label.pack(side="left", pady=(8, 0))
    # Modern stats display
    stats_frame = ctk.CTkFrame(main_frame, fg_color=("gray90", "gray20"), corner_radius=12)
    stats_frame.pack(fill="x", padx=10, pady=(0, 15))
    
    total_files = len(records)
    categories = {}
    for record in records:
        category = record[1]
        categories[category] = categories.get(category, 0) + 1
    
    stats_label = ctk.CTkLabel(
        stats_frame, 
        text=f"📊 {total_files} samples analyzed", 
        font=ctk.CTkFont(size=16, weight="bold"),
        text_color=("gray20", "white")
    )
    stats_label.pack(pady=(15, 5))
    
    # Category breakdown
    category_text = " | ".join([f"{cat.replace('_', ' ').title()}: {count}" for cat, count in categories.items()])
    category_label = ctk.CTkLabel(
        stats_frame, 
        text=category_text, 
        font=ctk.CTkFont(size=14),
        text_color=("gray40", "gray70")
    )
    category_label.pack(pady=(0, 15))
    
    # Modern search and filter controls
    search_frame = ctk.CTkFrame(main_frame, fg_color=("gray92", "gray18"), corner_radius=10)
    search_frame.pack(fill="x", padx=10, pady=(0, 15))
    
    # Left side controls
    left_controls = ctk.CTkFrame(search_frame, fg_color="transparent")
    left_controls.pack(side="left", padx=15, pady=10)
    
    # Search box with modern styling
    search_label = ctk.CTkLabel(left_controls, text="🔍 Search:", font=ctk.CTkFont(size=14, weight="bold"))
    search_label.pack(side="left", padx=(0, 8))
    
    search_var = tk.StringVar()
    search_entry = ctk.CTkEntry(
        left_controls, 
        textvariable=search_var,
        placeholder_text="Search files...", 
        width=250,
        height=32,
        corner_radius=8
    )
    search_entry.pack(side="left", padx=(0, 15))
    
    # Category filter with modern styling
    filter_label = ctk.CTkLabel(left_controls, text="🎯 Filter:", font=ctk.CTkFont(size=14, weight="bold"))
    filter_label.pack(side="left", padx=(0, 8))
    
    categories = ["All"] + list(set([record[1] if len(record) == 4 else record[1] for record in records]))
    filter_var = tk.StringVar(value="All")
    filter_dropdown = ctk.CTkOptionMenu(
        left_controls, 
        variable=filter_var, 
        values=categories, 
        width=150,
        height=32,
        corner_radius=8
    )
    filter_dropdown.pack(side="left", padx=(0, 15))
    
    # Audio controls with modern styling
    audio_label = ctk.CTkLabel(left_controls, text="🎵 Audio:", font=ctk.CTkFont(size=14, weight="bold"))
    audio_label.pack(side="left", padx=(0, 8))
    
    stop_button = ctk.CTkButton(
        left_controls, 
        text="⏹️ Stop", 
        width=80, 
        height=32,
        corner_radius=8,
        command=lambda: stop_audio()
    )
    stop_button.pack(side="left", padx=(0, 15))
    
    # Audio playback functions
    def init_pygame():
        nonlocal pygame_initialized
        if not pygame_initialized:
            try:
                import pygame
                pygame.mixer.init()
                pygame_initialized = True
            except Exception as e:
                print(f"Could not initialize pygame: {e}")
                return False
        return True
    
    def play_audio(file_path):
        nonlocal current_audio, playing_row
        if not init_pygame():
            return False
        try:
            import pygame
            pygame.mixer.music.load(str(file_path))
            pygame.mixer.music.play()
            current_audio = file_path
            # Schedule auto-stop check
            app.after(100, check_audio_finished)
            return True
        except Exception as e:
            print(f"Could not play {file_path}: {e}")
            return False
    
    def check_audio_finished():
        nonlocal current_audio, playing_row
        try:
            import pygame
            if not pygame.mixer.music.get_busy() and current_audio:
                stop_audio()
            else:
                # Check again in 100ms
                app.after(100, check_audio_finished)
        except:
            pass
    
    def stop_audio():
        nonlocal current_audio, playing_row
        try:
            import pygame
            pygame.mixer.music.stop()
            current_audio = None
            if playing_row:
                # Reset row background only
                playing_row.configure(fg_color=playing_row.original_fg)
                # Restore category label color
                for widget in playing_row.winfo_children():
                    if isinstance(widget, ctk.CTkLabel) and hasattr(widget, 'original_fg'):
                        widget.configure(fg_color=widget.original_fg)
                playing_row = None
        except:
            pass
    
    def on_row_click(event, row_frame, file_path):
        nonlocal playing_row
        # Stop current audio
        stop_audio()
        # Play new audio
        if play_audio(file_path):
            # Highlight row background only
            row_frame.configure(fg_color=("#444", "#222"))
            # Restore category label color
            for widget in row_frame.winfo_children():
                if isinstance(widget, ctk.CTkLabel) and hasattr(widget, 'original_fg'):
                    widget.configure(fg_color=widget.original_fg)
            playing_row = row_frame
    
    # Search and filter functions
    def sort_records(column_index):
        nonlocal sort_column, sort_reverse, records
        
        # If clicking the same column, reverse sort order
        if sort_column == column_index:
            sort_reverse = not sort_reverse
        else:
            sort_column = column_index
            sort_reverse = False
        
        # Sort the records
        def get_sort_key(record):
            if len(record) == 4:
                file_path, category, features, confidence = record
            else:
                file_path, category, features = record
                confidence = None
            
            if column_index == 0:  # File name
                return file_path.name.lower()
            elif column_index == 1:  # Category
                return category.lower()
            elif column_index == 2:  # Duration
                return features['duration']
            elif column_index == 3:  # Bass %
                return features['low_energy_ratio']
            elif column_index == 4:  # Transient
                return features['transient_strength']
            elif column_index == 5:  # Harmonic %
                return features['harmonic_ratio']
            elif column_index == 6:  # Pitch
                return features['pitch_strength']
            elif column_index == 7:  # Brightness
                return features['centroid_mean']
            elif column_index == 8:  # Energy
                return features['rms_mean']
            elif column_index == 9:  # Noise
                return features['spectral_flatness']
            elif column_index == 10:  # Confidence
                return confidence if confidence is not None else 0.0
            return 0
        
        records = sorted(records, key=get_sort_key, reverse=sort_reverse)
        
        # Rebuild the table
        rebuild_table()
        
        # Update header appearance
        update_header_appearance()
    
    def update_header_appearance():
        # Reset all headers to normal appearance
        for i, header_label in enumerate(header_labels):
            header_label.configure(fg_color="transparent", text_color=("gray20", "white"))
        
        # Highlight current sort column
        if sort_column is not None:
            sort_indicator = " ↓" if sort_reverse else " ↑"
            header_labels[sort_column].configure(
                fg_color=("gray80", "gray25"),
                text_color=("gray20", "white")
            )
            # Update text to show sort indicator
            original_text = headers[sort_column]
            header_labels[sort_column].configure(text=original_text + sort_indicator)
    
    def rebuild_table():
        # Clear existing data rows (keep headers)
        for widget in scrollable_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                widget.destroy()
        
        # Rebuild data rows
        for row_idx, record in enumerate(records, start=1):
            # Support both 3-tuple and 4-tuple (with confidence)
            if len(record) == 4:
                file_path, category, features, confidence = record
            else:
                file_path, category, features = record
                confidence = None
            # Create a modern frame for this row to handle clicks
            row_frame = ctk.CTkFrame(scrollable_frame, fg_color=("gray85", "gray25"), corner_radius=6)
            row_frame.grid(row=row_idx, column=0, columnspan=len(headers), padx=2, pady=1, sticky="ew")
            row_frame.original_fg = ("gray85", "gray25")  # Store original color
            
            file_name = file_path.name
            if len(file_name) > 40:
                file_name = file_name[:37] + "..."
            file_label = ctk.CTkLabel(
                row_frame, 
                text=file_name, 
                width=col_widths[0],
                anchor="w"
            )
            file_label.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
            
            category_label = ctk.CTkLabel(
                row_frame,
                text=category.replace("_", " ").title(),
                fg_color=category_colors.get(category, "#95A5A6"),
                text_color="white",
                corner_radius=8,
                width=col_widths[1],
                anchor="w",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            category_label.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
            category_label.original_fg = category_colors.get(category, "#95A5A6")  # Always keep this
            
            values = [
                f"{features['duration']:.2f}",
                f"{features['low_energy_ratio']*100:.1f}%",
                f"{features['transient_strength']:.2f}",
                f"{features['harmonic_ratio']*100:.1f}%",
                f"{features['pitch_strength']:.2f}",
                f"{features['centroid_mean']:.0f}",
                f"{features['rms_mean']:.2f}",
                f"{features['spectral_flatness']:.2f}"
            ]
            
            for col_idx, (value, width) in enumerate(zip(values, col_widths[2:-1]), start=2):
                label = ctk.CTkLabel(row_frame, text=value, width=width, anchor="w")
                label.grid(row=0, column=col_idx, padx=2, pady=2, sticky="ew")
            
            # Confidence column
            if confidence is not None:
                conf_percent = f"{confidence*100:.1f}%"
                conf_label = ctk.CTkLabel(row_frame, text=conf_percent, width=60, anchor="w")
                conf_label.grid(row=0, column=len(headers)-1, padx=(2, 0), pady=2, sticky="ew")
                conf_bar = ctk.CTkProgressBar(row_frame, width=60, height=16)
                conf_bar.set(confidence)
                conf_bar.grid(row=0, column=len(headers), padx=(0, 2), pady=2, sticky="ew")
            else:
                label = ctk.CTkLabel(row_frame, text="-", width=col_widths[-1], anchor="w")
                label.grid(row=0, column=len(headers)-1, padx=2, pady=2, sticky="ew")
            
            # Configure grid weights for the row
            for i in range(len(headers)):
                row_frame.grid_columnconfigure(i, weight=1)
            
            # Add click handler to the entire row
            row_frame.bind("<Button-1>", lambda event, rf=row_frame, fp=file_path: on_row_click(event, rf, fp))
            for widget in row_frame.winfo_children():
                widget.bind("<Button-1>", lambda event, rf=row_frame, fp=file_path: on_row_click(event, rf, fp))
    
    def filter_records():
        search_term = search_var.get().lower()
        category_filter = filter_var.get()
        
        # Get all row frames
        row_frames = []
        for widget in scrollable_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                row_frames.append(widget)
        
        # Hide all rows first
        for frame in row_frames:
            frame.grid_remove()
        
        # Show matching rows
        visible_count = 0
        for i, record in enumerate(records):
            if len(record) == 4:
                file_path, category, features, confidence = record
            else:
                file_path, category, features = record
                confidence = None
            file_name = file_path.name.lower()
            matches_search = search_term in file_name
            matches_category = category_filter == "All" or category == category_filter
            
            if matches_search and matches_category and i < len(row_frames):
                row_frames[i].grid()
                visible_count += 1
        
        # Update stats
        stats_text = f"Showing {visible_count} of {len(records)} samples"
        stats_label.configure(text=stats_text)
    
    # Bind search and filter events
    search_var.trace("w", lambda *args: filter_records())
    filter_var.trace("w", lambda *args: filter_records())
    # Modern table
    table_frame = ctk.CTkFrame(main_frame, fg_color=("gray90", "gray20"), corner_radius=12)
    table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 15))
    headers = ["File", "Category", "Duration (s)", "Bass %", "Transient", "Harmonic %", "Pitch", "Brightness", "Energy", "Noise", "Confidence"]
    col_widths = [250, 120, 100, 100, 100, 120, 100, 100, 100, 100, 140]
    # Scrollable canvas
    canvas = ctk.CTkCanvas(table_frame, highlightthickness=0)
    scrollbar = ctk.CTkScrollbar(table_frame, orientation="vertical", command=canvas.yview)
    scrollable_frame = ctk.CTkFrame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
    scrollbar.pack(side="right", fill="y", pady=5)
    # Modern header row - make clickable for sorting
    header_labels = []
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        label = ctk.CTkLabel(
            scrollable_frame, 
            text=header, 
            font=ctk.CTkFont(size=13, weight="bold"),
            width=width,
            anchor="w",
            cursor="hand2",  # Show hand cursor on hover
            fg_color=("#4169E1", "#1E90FF"),  # Royal blue
            text_color="white",
            corner_radius=6
        )
        label.grid(row=0, column=i, padx=2, pady=3, sticky="ew")
        label.bind("<Button-1>", lambda event, col=i: sort_records(col))
        header_labels.append(label)
        scrollable_frame.grid_columnconfigure(i, minsize=width, weight=1)
    # Colorblind-friendly category colors with good contrast
    category_colors = {
        "kick": "#E74C3C",      # Red - high contrast
        "808": "#9B59B6",       # Purple - distinct from red
        "bass": "#E67E22",      # Orange - warm tone
        "drum_loop": "#3498DB", # Blue - cool tone
        "hi_hat": "#1ABC9C",    # Teal - distinct blue-green
        "snare": "#F39C12",     # Yellow - high contrast
        "clap": "#2ECC71",      # Green - distinct from teal
        "vocal_loop": "#E91E63", # Pink - distinct from red
        "other": "#95A5A6"      # Gray - neutral
    }
    
    # Build initial table
    rebuild_table()
    # Modern bottom buttons
    button_frame = ctk.CTkFrame(main_frame, fg_color=("gray92", "gray18"), corner_radius=10)
    button_frame.pack(fill="x", padx=10, pady=(0, 15))
    
    def export_results():
        import csv
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for file_path, category, features, confidence in records:
                    writer.writerow([
                        str(file_path),
                        category,
                        f"{features['duration']:.2f}",
                        f"{features['low_energy_ratio']*100:.1f}%",
                        f"{features['transient_strength']:.2f}",
                        f"{features['harmonic_ratio']*100:.1f}%",
                        f"{features['pitch_strength']:.2f}",
                        f"{features['centroid_mean']:.0f}",
                        f"{features['rms_mean']:.2f}",
                        f"{features['spectral_flatness']:.2f}",
                        f"{confidence*100:.1f}%" if confidence else "N/A"
                    ])
            status_label.configure(text=f"📤 Exported to {filename}")
    
    # Left side buttons
    left_buttons = ctk.CTkFrame(button_frame, fg_color="transparent")
    left_buttons.pack(side="left", padx=15, pady=10)
    
    export_button = ctk.CTkButton(
        left_buttons, 
        text="📤 Export to CSV", 
        command=export_results,
        width=140,
        height=32,
        corner_radius=8,
        fg_color=("#4169E1", "#1E90FF"),  # Royal blue
        hover_color=("#2E4B8F", "#1874CD")
    )
    export_button.pack(side="left", padx=(0, 10))
    
    status_label = ctk.CTkLabel(
        left_buttons, 
        text="✨ Ready", 
        font=ctk.CTkFont(size=14),
        text_color=("gray40", "gray70")
    )
    status_label.pack(side="left", padx=10)
    
    # Right side buttons
    right_buttons = ctk.CTkFrame(button_frame, fg_color="transparent")
    right_buttons.pack(side="right", padx=15, pady=10)
    
    close_button = ctk.CTkButton(
        right_buttons, 
        text="✖️ Close", 
        command=app.destroy,
        width=100,
        height=32,
        corner_radius=8
    )
    close_button.pack(side="right")
    app.mainloop()


# -----------------------------
# ML functionality
# -----------------------------




def use_model(records, model_path):
    """Use trained ML model for classification."""
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


# -----------------------------
# CLI entry-point
# -----------------------------

def main():  # noqa: C901 – simple script
    parser = argparse.ArgumentParser(description="Hathor - Analyse & organise audio samples using ML classification")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with .mp3/.wav files")
    parser.add_argument("--output-dir", type=Path, help="Destination root for organised samples")
    parser.add_argument("--move", action="store_true", help="Actually move files instead of reporting only")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (requires customtkinter)")
    parser.add_argument("--model", type=Path, default="audio_classifier.joblib", help="ML model path (default: audio_classifier.joblib)")
    args = parser.parse_args()

    if args.gui and not GUI_AVAILABLE:
        sys.exit("customtkinter not installed. Install via: pip install customtkinter")

    audio_files = [*args.input_dir.rglob("*.mp3"), *args.input_dir.rglob("*.wav")]
    if not audio_files:
        sys.exit("No .mp3/.wav files found under input directory.")

    records = []  # (path, category, feats)
    for path in audio_files:
        try:
            feats = extract_features(path)
            records.append((path, feats))
            if args.move and args.output_dir:
                maybe_move(path, args.output_dir, feats)
        except Exception as exc:  # broad – quick script
            print(f"⚠️ {path.name}: {exc}", file=sys.stderr)

    # CLI summary
    for p, feats in records:
        print(
            f"{p.name:<40} | {feats['duration']:.2f}s | bass:{feats['low_energy_ratio']*100:.1f}% | trans:{feats['transient_strength']:.2f}"
        )

    # Use ML model for classification
    ml_records = use_model(records, args.model)

    # Launch GUI if requested
    if args.gui:
        launch_gui(ml_records)


if __name__ == "__main__":
    main()
