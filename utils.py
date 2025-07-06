# utils.py
# Shared helpers/utilities for Hathor 

"""
Utility functions for Hathor.

This module contains shared helper functions used across the application.
"""

import shutil
from pathlib import Path

# Category folder mapping
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

def maybe_move(src: Path, dest_root: Path, category: str) -> None:
    """Move *src* into *dest_root/category/*. Keeps directory flat."""
    dest = dest_root / CATEGORY_FOLDERS[category]
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dest / src.name) 