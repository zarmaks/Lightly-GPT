# Session state management utilities

import os
import streamlit as st
import chromadb

def initialize_session_state():
    """
    Initialize all session state variables in a centralized function
    to avoid scattered initialization throughout the code
    """
    # Initialize image storage
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
        
    # Processing flag  
    if 'processed' not in st.session_state:
        st.session_state.processed = False
        
    # ChromaDB client
    if 'chroma_client' not in st.session_state:
        # Create a data directory for ChromaDB
        home_dir = os.path.expanduser("~")
        chroma_data_dir = os.path.join(home_dir, "agent4o_clip_data")
        os.makedirs(chroma_data_dir, exist_ok=True)
        st.session_state.chroma_client = chromadb.PersistentClient(
            path=chroma_data_dir,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        
    # Image collection
    if 'collection' not in st.session_state:
        st.session_state.collection = None
        
    # Image metadata
    if 'image_metadata' not in st.session_state:
        st.session_state.image_metadata = {}
        
    # CLIP model components
    if 'clip_model' not in st.session_state:
        st.session_state.clip_model = None
        
    if 'clip_processor' not in st.session_state:
        st.session_state.clip_processor = None
        
    # Chat components
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'agent' not in st.session_state:
        st.session_state.agent = None

def validate_session_files():
    """Clean up session state to ensure only files with valid extensions remain"""
    if 'uploaded_images' in st.session_state and st.session_state.uploaded_images:
        valid_files = []
        for file in st.session_state.uploaded_images:
            try:
                # Check if file is valid and has valid extension
                if hasattr(file, 'name'):
                    ext = os.path.splitext(file.name)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        valid_files.append(file)
            except:
                # Skip any files that cause errors
                pass
        
        # Update session state with only valid files
        st.session_state.uploaded_images = valid_files

def get_active_indices():
    """Return indices of currently active (filtered) images"""
    if hasattr(st.session_state, "last_filtered_indices") and st.session_state.last_filtered_indices:
        return st.session_state.last_filtered_indices
    return list(range(len(st.session_state.uploaded_images)))
