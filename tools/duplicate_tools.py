# Tools for finding duplicate images

import streamlit as st
import sys
import os
from PIL import Image

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import compute_image_hash

def find_duplicate_images(threshold_str="5"):
    """
    Find duplicate or very similar images in the collection
    
    Args:
        threshold_str: Similarity threshold (1-10, where 10 is most strict)
        
    Returns:
        String describing duplicate images found
    """
    try:
        # Parse threshold (1-10 scale, convert to actual hash difference threshold)
        threshold = max(1, min(10, int(threshold_str)))
        hash_threshold = 16 - threshold  # Convert to hash difference (0-16 scale)
        
        # Calculate image hashes
        hashes = []
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            try:
                img_file.seek(0)
                img = Image.open(img_file).convert('RGB')
                
                # Use perceptual hash with error handling
                img_hash = compute_image_hash(img)
                if img_hash is None:
                    continue
                
                hashes.append({
                    "index": idx,
                    "filename": img_file.name,
                    "hash": img_hash
                })
            except Exception as e:
                st.warning(f"Couldn't process image {idx}: {str(e)}")
        
        if not hashes:
            return "No images could be processed for duplicate detection."
        
        # Find duplicates
        duplicates = []
        
        for i in range(len(hashes)):
            for j in range(i+1, len(hashes)):
                hash_distance = hashes[i]["hash"] - hashes[j]["hash"]
                
                if hash_distance <= hash_threshold:
                    duplicates.append({
                        "image1": hashes[i],
                        "image2": hashes[j],
                        "similarity": 100 - (hash_distance * 6.25)  # Convert to percentage
                    })
        
        # Generate response
        if not duplicates:
            return f"No duplicate images found with similarity threshold {threshold}/10."
        
        duplicates.sort(key=lambda x: x["similarity"], reverse=True)
        
        response = f"Found {len(duplicates)} potential duplicate image pairs:\n\n"
        
        for i, dup in enumerate(duplicates):
            response += f"Pair {i+1}: {dup['similarity']:.1f}% similar\n"
            response += f"  Image {dup['image1']['index']}: {dup['image1']['filename']}\n"
            response += f"  Image {dup['image2']['index']}: {dup['image2']['filename']}\n\n"
            
        return response
        
    except Exception as e:
        return f"An error occurred while finding duplicates: {str(e)}"