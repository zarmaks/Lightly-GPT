# CLIP model loading and embedding utilities

import torch
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

def load_clip_model():
    """Load CLIP model and processor for image embeddings"""
    try:
        with st.spinner("Loading CLIP model..."):
            # Using Hugging Face's CLIP implementation
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            st.session_state.clip_model = model
            st.session_state.clip_processor = processor
            st.success("✅ CLIP model loaded successfully")
            return True
    except Exception as e:
        st.error(f"Error loading CLIP model: {str(e)}")
        return False

def ensure_clip_model_loaded():
    """Ensure CLIP model is loaded, loading it if necessary"""
    if st.session_state.clip_model is None or st.session_state.clip_processor is None:
        return load_clip_model()
    return True

def generate_clip_embedding_generic(input_data, is_image=True):
    """Generate CLIP embedding for either image or text"""
    if not ensure_clip_model_loaded():
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        with torch.no_grad():
            if is_image:
                # Reset file position
                input_data.seek(0)
                img = Image.open(input_data).convert('RGB')
                inputs = st.session_state.clip_processor(images=img, return_tensors="pt", padding=True).to(device)
                features = st.session_state.clip_model.get_image_features(**inputs)
            else:
                inputs = st.session_state.clip_processor(text=input_data, return_tensors="pt", padding=True).to(device)
                features = st.session_state.clip_model.get_text_features(**inputs)
                
            # Normalize and convert to numpy
            features = features / features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy().flatten()
            return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def generate_clip_embedding(img_file):
    """Generate CLIP embedding for an image"""
    return generate_clip_embedding_generic(img_file, is_image=True)

def generate_text_embedding(query_text):
    """Generate CLIP embedding for a text query"""
    return generate_clip_embedding_generic(query_text, is_image=False)