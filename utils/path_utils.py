# Path utilities for managing import paths

import os
import sys
import importlib

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

def ensure_package_path():
    """
    Ensures the Agent-4o parent directory is in the Python path
    to enable reliable imports across the package
    """
    # Get the directory containing the utils module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get parent directory (should be Agent-4o)
    parent_dir = os.path.dirname(current_dir)
    
    # Add to path if not already present
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    return parent_dir

def import_module(module_name):
    """
    Safely import a module and return None if it fails
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None