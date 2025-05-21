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
        
        # Search for similar images in ChromaDB
        results = st.session_state.collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=5
        )
        
        if not results["ids"] or not results["ids"][0]:
            return f"No images matching '{query}' were found."
        
        # Extract results
        image_indices = [int(img_id.split("_")[1]) for img_id in results["ids"][0]]
        
        # Store found indices in session state for conversational memory
        st.session_state.last_filtered_indices = image_indices

        # Generate response with image descriptions
        response = f"I found {len(image_indices)} images matching '{query}':\n\n"
        
        # Prepare lists for image display
        matching_images = []
        captions = []
        
        for i, idx in enumerate(image_indices):
            if 0 <= idx < len(st.session_state.uploaded_images):
                # Add image to the list for display
                img = st.session_state.uploaded_images[idx]
                matching_images.append(img)
                
                # Create caption with filename
                filename = img.name
                captions.append(f"Image {idx}: {filename}")
                
                # Also include in text response
                response += f"Image {idx}: {filename}\n"
        
        # Display the matching images in a grid
        if matching_images:
            st.write("### Matching Images")
            display_image_grid(matching_images, num_columns=3, captions=captions)
        
        return response
    
    except Exception as e:
        return f"An error occurred during image search: {str(e)}"