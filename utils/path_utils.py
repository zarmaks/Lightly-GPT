# Path utilities for managing import paths and dependency checks

import os
import sys
import importlib

# Dictionary to track dependency availability
DEPENDENCIES = {
    'sklearn': None,
    'matplotlib': None,
    'torch': None,
    'transformers': None,
    'langchain': None,
    'chromadb': None,
}

def ensure_package_path():
    """
    Ensures the project parent directory is in the Python path
    to enable reliable imports across the package
    """
    # Get the directory containing the utils module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get parent directory
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

def check_dependency(name):
    """
    Check if a dependency is available and update the DEPENDENCIES dictionary
    Returns True if available, False otherwise
    """
    global DEPENDENCIES
    
    if DEPENDENCIES[name] is None:
        module = import_module(name)
        DEPENDENCIES[name] = module is not None
    
    return DEPENDENCIES[name]

def check_all_dependencies():
    """Check all dependencies and return a dictionary of results"""
    results = {}
    for dep in DEPENDENCIES:
        results[dep] = check_dependency(dep)
    return results

def get_missing_dependencies():
    """Return a list of missing dependencies"""
    missing = []
    for dep, available in DEPENDENCIES.items():
        if available is False:  # Only include explicitly False (checked and missing)
            missing.append(dep)
    return missing