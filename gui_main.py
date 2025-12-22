#!/usr/bin/env python3
"""
BAROx GUI - Formula Student Lap Time Simulator

Graphical user interface for the BAROx simulation tool.
Run this file to launch the GUI application.

Usage:
    python gui_main.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import main

if __name__ == "__main__":
    main()
