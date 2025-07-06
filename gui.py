"""
GUI functionality for Hathor.

This module contains the CustomTkinter GUI implementation for displaying
and interacting with audio analysis results.
"""

import customtkinter as ctk
import pygame
import tkinter as tk
from pathlib import Path

def launch_gui(records):
    """Launch a modern GUI using CustomTkinter."""
    
    # Set modern appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    app = ctk.CTk()
    app.title("Hathor - Audio Sample Analysis")
    app.geometry("1400x800")
    app.minsize(1000, 600)
    
    # Audio playback variables
    current_audio = None
    playing_row = None
    pygame_initialized = False
    
    # Sorting variables
    sort_column = None
    sort_reverse = False
    
    # Create gradient-like effect with multiple frames
    main_frame = ctk.CTkFrame(app, fg_color=("gray90", "gray15"))
    main_frame.pack(fill="both", expand=True, padx=15, pady=15)
    
    # Title with better styling
    title_label = ctk.CTkLabel(
        main_frame, 
        text="🎵 Hathor - Audio Sample Analysis", 
        font=ctk.CTkFont(size=28, weight="bold"),
        text_color=("gray20", "white")
    )
    title_label.pack(pady=(15, 25))
    
    # Stats frame
    stats_frame = ctk.CTkFrame(main_frame)
    stats_frame.pack(fill="x", padx=10, pady=(0, 10))
    total_files = len(records)
    categories = {}
    for record in records:
        category = record[1]
        categories[category] = categories.get(category, 0) + 1
    stats_text = f"📊 Total Files: {total_files} | "
    stats_text += " | ".join([f"{cat}: {count}" for cat, count in categories.items()])
    stats_label = ctk.CTkLabel(stats_frame, text=stats_text, font=ctk.CTkFont(size=14))
    stats_label.pack(pady=10)
    
    # Search and filter controls
    search_frame = ctk.CTkFrame(main_frame)
    search_frame.pack(fill="x", padx=10, pady=(0, 10))
    
    # Search box
    search_label = ctk.CTkLabel(search_frame, text="🔍 Search:", font=ctk.CTkFont(size=12, weight="bold"))
    search_label.pack(side="left", padx=(10, 5))
    
    search_var = tk.StringVar()
    search_entry = ctk.CTkEntry(search_frame, textvariable=search_var, width=200, placeholder_text="Search files...")
    search_entry.pack(side="left", padx=(0, 10))
    
    # Category filter
    filter_label = ctk.CTkLabel(search_frame, text="📂 Category:", font=ctk.CTkFont(size=12, weight="bold"))
    filter_label.pack(side="left", padx=(10, 5))
    
    categories = ["All"] + list(set([record[1] if len(record) == 4 else record[1] for record in records]))
    filter_var = tk.StringVar(value="All")
    filter_dropdown = ctk.CTkOptionMenu(search_frame, variable=filter_var, values=categories, width=120)
    filter_dropdown.pack(side="left", padx=(0, 10))
    
    # Audio controls
    audio_label = ctk.CTkLabel(search_frame, text="🎵 Audio:", font=ctk.CTkFont(size=12, weight="bold"))
    audio_label.pack(side="left", padx=(10, 5))
    
    stop_button = ctk.CTkButton(search_frame, text="⏹️ Stop", width=80, command=lambda: stop_audio())
    stop_button.pack(side="left", padx=(0, 10))
    
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
            # Create a frame for this row to handle clicks
            row_frame = ctk.CTkFrame(scrollable_frame, fg_color=("gray70", "gray30"))
            row_frame.grid(row=row_idx, column=0, columnspan=len(headers), padx=2, pady=1, sticky="ew")
            row_frame.original_fg = ("gray70", "gray30")  # Store original color
            
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
                font=ctk.CTkFont(size=11, weight="bold")
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
        stats_text = f"📊 Showing {visible_count} of {len(records)} samples"
        stats_label.configure(text=stats_text)
    
    # Bind search and filter events
    search_var.trace("w", lambda *args: filter_records())
    filter_var.trace("w", lambda *args: filter_records())
    
    # Table
    table_frame = ctk.CTkFrame(main_frame)
    table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
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
    
    # Header row - make clickable for sorting
    header_labels = []
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        label = ctk.CTkLabel(
            scrollable_frame, 
            text=header, 
            font=ctk.CTkFont(size=12, weight="bold"),
            width=width,
            anchor="w",
            cursor="hand2"  # Show hand cursor on hover
        )
        label.grid(row=0, column=i, padx=2, pady=2, sticky="ew")
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
    
    # Bottom buttons
    button_frame = ctk.CTkFrame(main_frame)
    button_frame.pack(fill="x", padx=10, pady=(0, 10))
    
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
    
    export_button = ctk.CTkButton(
        button_frame, 
        text="📤 Export to CSV", 
        command=export_results,
        width=120
    )
    export_button.pack(side="left", padx=(0, 10))
    
    status_label = ctk.CTkLabel(button_frame, text="✨ Ready")
    status_label.pack(side="left", padx=10)
    
    close_button = ctk.CTkButton(
        button_frame, 
        text="✖️ Close", 
        command=app.destroy,
        width=80
    )
    close_button.pack(side="right")
    
    app.mainloop()

# Add any other GUI helpers here 