# Image analysis tools for color, BW detection, etc.

import os
import sys
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageStat

# Import from our utils package
from utils import setup_project_path, validate_session_requirements, requires_session_setup
from utils.error_utils import handle_tool_errors
from utils.path_utils import check_dependency
from utils.session_utils import get_active_indices

# Ensure path is set up correctly
setup_project_path()

# Check if sklearn is available
SKLEARN_AVAILABLE = check_dependency("sklearn")
if not SKLEARN_AVAILABLE:
    st.warning("scikit-learn not found. Color analysis will be limited.")
else:
    from sklearn.cluster import KMeans

def rgb_to_hex(rgb):
    """Convert RGB color to hex string"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

@handle_tool_errors
@requires_session_setup
def analyze_image_colors(image_indices_str="use last filtered set"):
    """
    Extract dominant colors from specified images
    Args:
        image_indices_str: Comma-separated list of image indices to analyze, or 'use last filtered set'
    Returns:
        String with dominant color information
    """
    # Check if sklearn is available
    if not SKLEARN_AVAILABLE:
        return "Color analysis requires scikit-learn. Please install it with 'pip install scikit-learn'."

    # Support 'use last filtered set' or explicit indices
    if isinstance(image_indices_str, str) and image_indices_str.strip().lower() in ["use last filtered set", "use last", "previous", "last"]:
        indices = get_active_indices()
        if not indices:
            return "No previous filtered set found. Please run a filter or search first."
    elif isinstance(image_indices_str, str) and ',' in image_indices_str:
        indices = [int(idx.strip()) for idx in image_indices_str.split(',') if idx.strip().isdigit()]
    elif isinstance(image_indices_str, str) and image_indices_str.strip().isdigit():
        indices = [int(image_indices_str.strip())]
    else:
        indices = get_active_indices()
        if not indices:
            return "No valid image indices provided."

    # Validate indices
    valid_indices = [idx for idx in indices if 0 <= idx < len(st.session_state.uploaded_images)]
    if not valid_indices:
        return "No valid image indices provided."

    # Store filtered indices in session state for conversational memory
    st.session_state.last_filtered_indices = valid_indices

    results = []
    
    # Process each valid image
    for idx in valid_indices:
        img_file = st.session_state.uploaded_images[idx]
        img_file.seek(0)
        img = Image.open(img_file).convert('RGB')
        
        # Resize for faster processing
        img = img.resize((150, 150))
        
        # Convert to numpy array
        pixels = np.array(img).reshape(-1, 3)
        
        # Use k-means to find dominant colors
        n_colors = 5
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        
        # Convert to hex
        hex_colors = [rgb_to_hex(color) for color in colors]
        
        # Add to results
        results.append({
            "index": idx,
            "filename": img_file.name,
            "colors": hex_colors
        })
      # Generate response
    response = "Dominant colors analysis:\n\n"
    
    for result in results:
        response += f"Image {result['index']} ({result['filename']}):\n"
        for i, color in enumerate(result['colors']):
            response += f"  Color {i+1}: {color}\n"
        response += "\n"
        
    return response

@handle_tool_errors
@requires_session_setup
def detect_bw_images(image_indices_str="use last filtered set"):
    """
    Detect which images are black and white or grayscale
    Args:
        image_indices_str: Comma-separated list of image indices to analyze, or 'use last filtered set'
    Returns:
        String listing black & white images
    """
    # Support 'use last filtered set' or explicit indices
    if isinstance(image_indices_str, str) and image_indices_str.strip().lower() in ["use last filtered set", "use last", "previous", "last"]:
        indices = get_active_indices()
    elif isinstance(image_indices_str, str) and ',' in image_indices_str:
        indices = [int(idx.strip()) for idx in image_indices_str.split(',') if idx.strip().isdigit()]
    elif isinstance(image_indices_str, str) and image_indices_str.strip().isdigit():
        indices = [int(image_indices_str.strip())]
    else:
        indices = get_active_indices()

    bw_images = []
    filtered_indices = []
    for idx in indices:
        if 0 <= idx < len(st.session_state.uploaded_images):
            img_file = st.session_state.uploaded_images[idx]
            img_file.seek(0)
            img = Image.open(img_file).convert('RGB')
            # Calculate standard deviation of each color channel
            stat = ImageStat.Stat(img)
            r_std = stat.stddev[0]
            g_std = stat.stddev[1]
            b_std = stat.stddev[2]
            # Calculate difference between channels
            rg_diff = abs(stat.mean[0] - stat.mean[1])
            rb_diff = abs(stat.mean[0] - stat.mean[2])
            gb_diff = abs(stat.mean[1] - stat.mean[2])
            # If color channels are very similar, it's likely B&W
            if rg_diff < 5 and rb_diff < 5 and gb_diff < 5:
                bw_images.append({
                    "index": idx,
                    "filename": img_file.name,
                    "confidence": "high" if max(r_std, g_std, b_std) < 50 else "medium"
                })
                filtered_indices.append(idx)
    # Store filtered indices in session state for conversational memory
    st.session_state.last_filtered_indices = filtered_indices

    # Generate response
    if not bw_images:
        return "No black and white images detected in your collection."

    response = f"Found {len(bw_images)} black and white images:\n\n"
    for img in bw_images:
        response += f"Image {img['index']}: {img['filename']} (Confidence: {img['confidence']})\n"
    return response