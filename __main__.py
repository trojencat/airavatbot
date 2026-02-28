#!/usr/bin/env python3
"""
Entry point wrapper for PyInstaller builds.
This allows the relative imports in src.server to work correctly.
"""
import sys
from src.server import main

if __name__ == "__main__":
    main()
