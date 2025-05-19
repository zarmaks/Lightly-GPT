# Image analysis tools for color, BW detection, etc.

import io
import numpy as np
import streamlit as st
import sys
import os
from PIL import Image, ImageStat

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Try to import sklearn with error handling
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    st.warning("scikit-learn not found. Color analysis will be disabled.")
    SKLEARN_AVAILABLE = False

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rgb_to_hex(rgb):
    """Convert RGB color to hex string"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def analyze_image_colors(image_indices_str):
    """
    Extract dominant colors from specified images
    
    Args:
        image_indices_str: Comma-separated list of image indices to analyze
        
    Returns:
        String with dominant color information
    """
    # Check if sklearn is available
    if not SKLEARN_AVAILABLE:
        return "Color analysis requires scikit-learn. Please install it with 'pip install scikit-learn'."
    
    # Check if session state has necessary variables
    if not hasattr(st.session_state, 'uploaded_images') or not st.session_state.uploaded_images:
        return "No images uploaded. Please upload images first."
    
    try:
        # Parse indices
        if ',' in image_indices_str:
            indices = [int(idx.strip()) for idx in image_indices_str.split(',')]
        else:
            indices = [int(image_indices_str.strip())]
        
        # Validate indices
        valid_indices = []
        for idx in indices:
            if 0 <= idx < len(st.session_state.uploaded_images):
                valid_indices.append(idx)
        
        if not valid_indices:
            return "No valid image indices provided."
        
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
        
    except Exception as e:
        return f"An error occurred while analyzing colors: {str(e)}"

def detect_bw_images(dummy=""):
    """
    Detect which images are black and white or grayscale
    
    Returns:
        String listing black & white images
    """
    # Check if session state has necessary variables
    if not hasattr(st.session_state, 'uploaded_images') or not st.session_state.uploaded_images:
        return "No images uploaded. Please upload images first."
    
    try:
        bw_images = []
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
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
        
        # Generate response
        if not bw_images:
            return "No black and white images detected in your collection."
        
        response = f"Found {len(bw_images)} black and white images:\n\n"
        
        for img in bw_images:
            response += f"Image {img['index']}: {img['filename']} (Confidence: {img['confidence']})\n"
            
        return response
        
    except Exception as e:
        return f"An error occurred while detecting B&W images: {str(e)}"