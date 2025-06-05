# Tools for finding duplicate images

import warnings

import streamlit as st
from PIL import Image

from ..utils import setup_project_path
from ..utils.image_utils import compute_image_hash, resize_image

warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Ensure project path is set up
setup_project_path()


def find_duplicate_images(threshold_str="5"):
    """
    Find duplicate or very similar images in the collection and display them

    Args:
        threshold_str: Similarity threshold (1-10, where 10 is most strict)

    Returns:
        String describing duplicate images found
    """
    try:
        # Check if we have uploaded images
        if not hasattr(st.session_state, 'uploaded_images') or not st.session_state.uploaded_images:
            st.warning("No images have been uploaded yet. Please upload images first.")
            return "No images have been uploaded yet. Please upload images first to find duplicates."
          # Parse threshold (1-10 scale, convert to actual hash difference threshold)
        threshold = max(1, min(10, int(threshold_str)))
        hash_threshold = 16 - threshold  # Convert to hash difference (0-16 scale)

        st.info(f"Processing {len(st.session_state.uploaded_images)} images for duplicate detection...")

        # Calculate image hashes
        hashes = []
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            try:
                # Use perceptual hash with error handling
                img_hash = compute_image_hash(img_file)
                if img_hash is None:
                    continue

                hashes.append(
                    {
                        "index": idx,
                        "filename": img_file.name,
                        "hash": img_hash,
                        "file": img_file,
                    }
                )
            except Exception as e:
                st.warning(f"Couldn't process image {idx}: {str(e)}")

        if not hashes:
            st.warning("No images could be processed for duplicate detection.")
            return "No images could be processed for duplicate detection."

        # Find duplicates
        duplicates = []

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                hash_distance = hashes[i]["hash"] - hashes[j]["hash"]

                if hash_distance <= hash_threshold:
                    duplicates.append(
                        {
                            "image1": hashes[i],
                            "image2": hashes[j],
                            "similarity": 100
                            - (hash_distance * 6.25),  # Convert to percentage
                        }
                    )

        # Display results
        if not duplicates:
            st.info(
                f"🎉 No duplicate images found with similarity threshold {threshold}/10."
            )
            return (
                f"No duplicate images found with similarity threshold {threshold}/10."
            )

        duplicates.sort(key=lambda x: x["similarity"], reverse=True)

        # Display duplicate pairs with images
        st.subheader(f"🔍 Found {len(duplicates)} Duplicate Image Pairs")
        st.write(f"*Similarity threshold: {threshold}/10*")

        for i, dup in enumerate(duplicates):
            with st.expander(
                f"Duplicate Pair {i + 1}: {dup['similarity']:.1f}% Similar",
                expanded=True,
            ):
                col1, col2 = st.columns(2)
                # Display first image
                with col1:
                    st.write(
                        f"**Image {dup['image1']['index']}**: {dup['image1']['filename']}"
                    )
                    try:
                        dup["image1"]["file"].seek(0)
                        img1 = Image.open(dup["image1"]["file"]).convert("RGB")
                        # Resize for better display
                        img1_resized = resize_image(img1, max_size=400)
                        st.image(
                            img1_resized,
                            caption=f"Index: {dup['image1']['index']}",
                            use_column_width=True,
                        )
                    except Exception as e:
                        st.error(f"Could not display image: {str(e)}")

                # Display second image
                with col2:
                    st.write(
                        f"**Image {dup['image2']['index']}**: {dup['image2']['filename']}"
                    )
                    try:
                        dup["image2"]["file"].seek(0)
                        img2 = Image.open(dup["image2"]["file"]).convert("RGB")
                        # Resize for better display
                        img2_resized = resize_image(img2, max_size=400)
                        st.image(
                            img2_resized,
                            caption=f"Index: {dup['image2']['index']}",
                            use_column_width=True,
                        )
                    except Exception as e:
                        st.error(f"Could not display image: {str(e)}")
                # Show similarity percentage prominently
                col_metric, col_actions = st.columns([1, 2])
                with col_metric:
                    st.metric("Similarity", f"{dup['similarity']:.1f}%")

                with col_actions:
                    st.write("**Actions:**")
                    if st.button(
                        f"Remove Image {dup['image1']['index']}", key=f"remove_{i}_1"
                    ):
                        st.info(
                            f"Would remove image {dup['image1']['index']}: {dup['image1']['filename']}"
                        )
                        st.write(
                            "*(Feature can be implemented to actually remove from session state)*"
                        )

                    if st.button(
                        f"Remove Image {dup['image2']['index']}", key=f"remove_{i}_2"
                    ):
                        st.info(
                            f"Would remove image {dup['image2']['index']}: {dup['image2']['filename']}"
                        )
                        st.write(
                            "*(Feature can be implemented to actually remove from session state)*"
                        )

        # Return summary text for the agent
        response = f"Found and displayed {len(duplicates)} duplicate image pairs with similarity threshold {threshold}/10. "
        response += "The images are shown above with their similarity percentages."

        return response

    except Exception as e:
        error_msg = f"An error occurred while finding duplicates: {str(e)}"
        st.error(error_msg)
        return error_msg
