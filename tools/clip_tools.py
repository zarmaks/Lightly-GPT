# CLIP-based image search tools

import streamlit as st
import sys
import os

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.clip_utils import generate_text_embedding
from utils.ui_utils import display_image_grid
from utils.session_utils import get_active_indices

def clip_image_search_tool(query):
    """
    Search for images matching a text description using CLIP embeddings
    
    Args:
        query: Text description to search for
        
    Returns:
        String describing the search results with image indices
    """
    # Check if session state has necessary variables
    if not hasattr(st.session_state, 'processed') or not st.session_state.processed:
        return "No processed images available. Please process images first."
    
    if not hasattr(st.session_state, 'collection') or st.session_state.collection is None:
        return "No image collection available. Please process images first."
    
    if not hasattr(st.session_state, 'uploaded_images') or not st.session_state.uploaded_images:
        return "No images uploaded. Please upload images first."
    
    try:
        # Generate embedding for the query
        text_embedding = generate_text_embedding(query)
        if text_embedding is None:
            return "Failed to generate text embedding for your query."
        
        # Search for similar images in ChromaDB (top 10 most similar for more flexibility)
        results = st.session_state.collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=10
        )
        
        if not results["ids"] or not results["ids"][0]:
            return f"No images matching '{query}' were found."
        
        # Debug: print/log distances
        distances = results.get("distances", [None])[0]
        st.write(f"DEBUG: distances returned: {distances}")
        
        # Apply similarity threshold (cosine distance, lower is more similar)
        similarity_threshold = 0.8  # Default threshold, can be made user-adjustable
        filtered_indices = []
        filtered_distances = []
        ids = results["ids"][0]
        for i, dist in enumerate(distances):
            if dist is not None and dist < similarity_threshold:
                idx = int(ids[i].split("_")[1])
                filtered_indices.append(idx)
                filtered_distances.append(dist)
        st.session_state.last_filtered_indices = filtered_indices
        
        # Use get_active_indices for search scope (if you want to restrict search to filtered set)
        active_indices = get_active_indices()
        
        # After getting filtered_indices, intersect with active_indices if you want chaining
        filtered_indices = [idx for idx in filtered_indices if idx in active_indices]
        st.session_state.last_filtered_indices = filtered_indices
        
        response = f"I found {len(filtered_indices)} images matching '{query}' (distance < {similarity_threshold}):\n\n"
        matching_images = []
        captions = []
        for i, idx in enumerate(filtered_indices):
            if 0 <= idx < len(st.session_state.uploaded_images):
                img = st.session_state.uploaded_images[idx]
                matching_images.append(img)
                filename = img.name
                captions.append(f"Image {idx}: {filename} (distance: {filtered_distances[i]:.2f})")
                response += f"Image {idx}: {filename} (distance: {filtered_distances[i]:.2f})\n"
        
        if matching_images:
            st.write("### Matching Images")
            display_image_grid(matching_images, num_columns=3, captions=captions)
        else:
            response += "No images passed the similarity threshold."
        
        return response
    
    except Exception as e:
        return f"An error occurred during image search: {str(e)}"