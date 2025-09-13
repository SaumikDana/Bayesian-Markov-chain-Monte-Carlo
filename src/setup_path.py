"""
Python Path Setup Module

Adds the project root directory to sys.path to enable imports from anywhere
within the project structure. Simply import this module at the beginning
of any script to ensure consistent import behavior.

Usage:
    import setup_path  # Must be first import
    from src.imports import *  # Now works from any location
"""

import os
import sys

# Determine the path of the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add the parent directory to sys.path if not already present
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)