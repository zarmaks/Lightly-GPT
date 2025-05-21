# Utility functions for the Agent-4o application
# Provides CLIP model utilities, image processing, and UI components

import os
import sys
import warnings
import streamlit as st
import numpy as np
from PIL import Image

# Common warnings to suppress
warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Utility functions for path management and common checks
def setup_project_path():
    """Add the project root to Python path for more reliable imports"""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    return current_dir

def validate_session_requirements():
    """Check if required session state variables exist"""
    if not hasattr(st.session_state, 'uploaded_images') or not st.session_state.uploaded_images:
        return False, "No images uploaded. Please upload images first."
    
    if not hasattr(st.session_state, 'processed') or not st.session_state.processed:
        return False, "Images not processed. Please process images first."
        
    if not hasattr(st.session_state, 'collection') or st.session_state.collection is None:
        return False, "No image collection found. Please process images first."
    
    return True, "Requirements met"

# Function decorator for checking session state requirements
def requires_session_setup(func):
    """Decorator to check if the session state is properly set up before running a function"""
    def wrapper(*args, **kwargs):
        is_valid, message = validate_session_requirements()
        if not is_valid:
            return message
        return func(*args, **kwargs)
    return wrapper

# Initialize the paths
setup_project_path()