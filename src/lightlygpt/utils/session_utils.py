# Session state management utilities

import os
import sys

# Fix for Streamlit Cloud SQLite compatibility issue with ChromaDB
# This must be done before importing chromadb
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
import streamlit as st


def initialize_session_state():
    """
    Initialize all session state variables in a centralized function
    to avoid scattered initialization throughout the code
    """
    # Initialize image storage
    if "uploaded_images" not in st.session_state:
        st.session_state["uploaded_images"] = []

    # Processing flag
    if "processed" not in st.session_state:
        st.session_state["processed"] = False    # ChromaDB client
    if "chroma_client" not in st.session_state:
        try:
            # Create a data directory for ChromaDB  
            home_dir = os.path.expanduser("~")
            chroma_data_dir = os.path.join(home_dir, "agent4o_clip_data")
            os.makedirs(chroma_data_dir, exist_ok=True)
            st.session_state["chroma_client"] = chromadb.PersistentClient(
                path=chroma_data_dir, settings=chromadb.Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            st.error(f"⚠️ ChromaDB initialization failed: {str(e)}")
            st.info("💡 This is usually due to SQLite version compatibility in Streamlit Cloud. The app will continue with limited functionality.")
            st.session_state["chroma_client"] = None

    # Image collection
    if "collection" not in st.session_state:
        st.session_state["collection"] = None

    # Image metadata
    if "image_metadata" not in st.session_state:
        st.session_state["image_metadata"] = {}

    # CLIP model components
    if "clip_model" not in st.session_state:
        st.session_state["clip_model"] = None

    if "clip_processor" not in st.session_state:
        st.session_state["clip_processor"] = None

    # Chat components
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "agent" not in st.session_state:
        st.session_state["agent"] = None


def validate_session_files():
    """Clean up session state to ensure only files with valid extensions remain"""
    if "uploaded_images" in st.session_state and st.session_state.uploaded_images:
        valid_files = []
        for file in st.session_state.uploaded_images:
            try:
                # Check if file is valid and has valid extension
                if hasattr(file, "name"):
                    ext = os.path.splitext(file.name)[1].lower()
                    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                        valid_files.append(file)
            except Exception:
                # Skip any files that cause errors
                pass

        # Update session state with only valid files
        st.session_state.uploaded_images = valid_files


def get_active_indices():
    """Return indices of currently active (filtered) images"""
    if (
        hasattr(st.session_state, "last_filtered_indices")
        and st.session_state.last_filtered_indices
    ):
        return st.session_state.last_filtered_indices
    return list(range(len(st.session_state.uploaded_images)))


def check_chromadb_status():
    """Check if ChromaDB is properly initialized and working"""
    try:
        if not hasattr(st.session_state, "chroma_client") or st.session_state.chroma_client is None:
            return False, "ChromaDB client not initialized"
        
        # Try a simple operation to verify it's working
        test_collections = st.session_state.chroma_client.list_collections()
        return True, "ChromaDB is working properly"
        
    except Exception as e:
        return False, f"ChromaDB error: {str(e)}"


def display_chromadb_status():
    """Display ChromaDB status in the sidebar for debugging"""
    is_working, message = check_chromadb_status()
    
    if is_working:
        st.sidebar.success(f"✅ {message}")
    else:
        st.sidebar.error(f"❌ {message}")
        if "sqlite" in message.lower():
            st.sidebar.info("💡 Try redeploying the app or check SQLite compatibility")
