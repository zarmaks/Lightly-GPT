# CLIP-based image search tools

import warnings

import numpy as np
import streamlit as st

from ..utils import setup_project_path
from ..utils.clip_utils import generate_text_embedding
from ..utils.session_utils import get_active_indices
from ..utils.ui_utils import display_image_grid

warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Ensure project path is set up
setup_project_path()


def apply_temperature_scaling(distances, temperature=0.1):
    """
    Apply temperature scaling to improve contrast between relevant and irrelevant results

    Args:
        distances: Array of cosine distances
        temperature: Temperature parameter (lower = more contrast, default 0.1)

    Returns:
        Tuple of (scaled_distances, softmax_scores)
    """

    # Convert distances to similarities (1 - distance)
    similarities = 1 - np.array(distances)

    # Apply temperature scaling to logits (similarities)
    scaled_similarities = similarities / temperature

    # Apply softmax to get probability-like scores
    exp_similarities = np.exp(scaled_similarities - np.max(scaled_similarities))  # Subtract max for numerical stability
    softmax_scores = exp_similarities / np.sum(exp_similarities)

    # Convert back to distances but with enhanced contrast
    # Use a different scaling that preserves the ranking but increases separation
    scaled_distances = 1 - (softmax_scores / np.max(softmax_scores))

    return scaled_distances, softmax_scores


def clip_image_search_tool(query, threshold=0.8, use_dynamic=True, percentile=70, use_temperature_scaling=True, temperature=0.15):
    """
    Search for images matching a text description using CLIP embeddings with enhanced contrast

    Args:
        query: Text description to search for
        threshold: Maximum distance threshold for similarity
        use_dynamic: Whether to use dynamic threshold calculation
        percentile: Percentile for dynamic threshold (default 70)
        use_temperature_scaling: Whether to apply temperature scaling for better contrast
        temperature: Temperature parameter for scaling (lower = more contrast)
    """
    # Validate parameters
    try:
        threshold = float(threshold)
        percentile = int(percentile)
        use_dynamic = bool(use_dynamic)
        temperature = float(temperature)

        if not (0.0 <= threshold <= 1.0):
            return "Error: threshold must be between 0.0 and 1.0"

        if not (1 <= percentile <= 99):
            return "Error: percentile must be between 1 and 99"

        if not (0.01 <= temperature <= 1.0):
            return "Error: temperature must be between 0.01 and 1.0"

    except (ValueError, TypeError) as e:
        return f"Error: Invalid parameter types - {str(e)}"

    # Check if session state has necessary variables
    if not hasattr(st.session_state, "processed") or not st.session_state.processed:
        return "No processed images available. Please process images first."

    if (
        not hasattr(st.session_state, "collection")
        or st.session_state.collection is None
    ):
        return "No image collection available. Please process images first."

    if (
        not hasattr(st.session_state, "uploaded_images")
        or not st.session_state.uploaded_images
    ):
        return "No images uploaded. Please upload images first."

    try:
        # Generate embedding for the query
        text_embedding = generate_text_embedding(query)
        if text_embedding is None:
            return "Failed to generate text embedding for your query."

        # Search for similar images in ChromaDB (get all images)
        results = st.session_state.collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=len(st.session_state.uploaded_images),
        )

        if not results["ids"] or not results["ids"][0]:
            return f"No images matching '{query}' were found."

        # Get distances and apply temperature scaling for better contrast
        distances = results.get("distances", [None])[0]
        st.write(f"DEBUG: Raw distances (first 10): {[f'{d:.3f}' for d in distances[:10] if d is not None]}")

        # Apply temperature scaling to improve result quality
        if use_temperature_scaling:
            valid_distances = [d for d in distances if d is not None]
            if len(valid_distances) > 0:
                scaled_distances, softmax_scores = apply_temperature_scaling(valid_distances, temperature)

                # Replace None values in original distances with scaled ones
                final_distances = []
                valid_idx = 0
                for d in distances:
                    if d is not None:
                        final_distances.append(scaled_distances[valid_idx])
                        valid_idx += 1
                    else:
                        final_distances.append(None)

                distances = final_distances
                st.write(f"DEBUG: Scaled distances (first 10): {[f'{d:.3f}' for d in distances[:10] if d is not None]}")
                st.write(f"DEBUG: Softmax scores (first 10): {[f'{s:.4f}' for s in softmax_scores[:10]]}")
            else:
                st.warning("No valid distances found for temperature scaling")

        if use_dynamic:
            import numpy as np

            # Remove None values for threshold calculation
            valid_distances = [d for d in distances if d is not None]

            if len(valid_distances) < 5:
                final_threshold = threshold
                st.write(f"DEBUG: Too few images ({len(valid_distances)}), using static threshold: {final_threshold:.3f}")
            else:
                # Calculate percentile-based dynamic threshold
                dynamic_threshold = np.percentile(valid_distances, percentile)

                # Apply safety limits
                min_threshold = 0.1  # More aggressive minimum after scaling
                final_threshold = min(max(dynamic_threshold, min_threshold), threshold)

                st.write(f"""
                DEBUG: Enhanced dynamic threshold calculation:
                - Valid distances: {len(valid_distances)} images
                - {percentile}th percentile: {dynamic_threshold:.3f}
                - User max threshold: {threshold:.3f}
                - Final threshold: {final_threshold:.3f}
                - Temperature scaling: {'ON' if use_temperature_scaling else 'OFF'} (T={temperature})
                """)
        else:
            final_threshold = threshold
            st.write(f"DEBUG: Using static threshold: {final_threshold:.3f}")

        # Apply the calculated threshold
        filtered_indices = []
        filtered_distances = []
        ids = results["ids"][0]
        for i, dist in enumerate(distances):
            if dist is not None and dist < final_threshold:
                idx = int(ids[i].split("_")[1])
                filtered_indices.append(idx)
                filtered_distances.append(dist)

        # Sort by distance (best matches first)
        if filtered_indices:
            sorted_pairs = sorted(zip(filtered_indices, filtered_distances), key=lambda x: x[1])
            filtered_indices, filtered_distances = zip(*sorted_pairs)
            filtered_indices = list(filtered_indices)
            filtered_distances = list(filtered_distances)

        st.session_state.last_filtered_indices = filtered_indices

        # Use get_active_indices for search scope
        active_indices = get_active_indices()
        filtered_indices = [idx for idx in filtered_indices if idx in active_indices]
        st.session_state.last_filtered_indices = filtered_indices

        response = f"I found {len(filtered_indices)} images matching '{query}' (using {'dynamic' if use_dynamic else 'static'} threshold {final_threshold:.3f}):\n\n"
        matching_images = []
        captions = []

        for i, idx in enumerate(filtered_indices):
            if 0 <= idx < len(st.session_state.uploaded_images):
                img = st.session_state.uploaded_images[idx]
                matching_images.append(img)
                filename = img.name

                # Convert distance to similarity percentage with enhanced scaling
                if use_temperature_scaling:
                    # For temperature-scaled results, use a different similarity calculation
                    similarity_pct = (1 - filtered_distances[i]) * 100
                    confidence = "HIGH" if similarity_pct > 80 else "MEDIUM" if similarity_pct > 60 else "LOW"
                else:
                    similarity_pct = (1 - filtered_distances[i]) * 100
                    confidence = "NORMAL"

                captions.append(
                    f"Image {idx}: {filename} ({similarity_pct:.1f}% - {confidence})"
                )
                response += (
                    f"Image {idx}: {filename} (similarity: {similarity_pct:.1f}%, confidence: {confidence}, distance: {filtered_distances[i]:.3f})\n"
                )

        if matching_images:
            st.write("### 🎯 Best Matching Images (Enhanced Contrast)")
            display_image_grid(matching_images, num_columns=3, captions=captions)

            if use_temperature_scaling:
                st.info(f"🔥 **Temperature scaling applied** (T={temperature}) for enhanced result quality. "
                       f"Found {len(matching_images)} high-confidence matches!")
        else:
            response += f"No images passed the similarity threshold ({final_threshold:.3f}). "
            if use_temperature_scaling:
                response += "Try increasing the search sensitivity in the sidebar or disabling temperature scaling."
            else:
                response += "Try increasing the search sensitivity in the sidebar."

        return response

    except Exception as e:
        return f"An error occurred during image search: {str(e)}"
