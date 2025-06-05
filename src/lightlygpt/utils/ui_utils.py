# UI helper functions for Agent_4o

import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from .image_utils import image_to_base64

warnings.filterwarnings("ignore", message=".*use_column_width.*")


def display_image_grid(images, num_columns=4, captions=None):
    """
    Display a grid of images in Streamlit

    Args:
        images: List of file-like objects containing image data
        num_columns: Number of columns in the grid
        captions: Optional list of captions for each image
    """
    columns = st.columns(num_columns)

    for i, img_file in enumerate(images):
        col_idx = i % num_columns

        with columns[col_idx]:
            try:
                img_file.seek(0)
                caption = (
                    captions[i] if captions and i < len(captions) else img_file.name
                )
                st.image(img_file, use_container_width=True, caption=caption)
            except Exception as e:
                st.error(f"Could not display image: {str(e)}")


def display_image_with_info(img_file, metadata=None):
    """
    Display an image with its metadata information

    Args:
        img_file: File-like object containing image data
        metadata: Dictionary of metadata to display
    """
    col1, col2 = st.columns([1, 1])

    with col1:
        img_file.seek(0)
        st.image(img_file, use_container_width=True, caption=img_file.name)

    with col2:
        st.subheader("Image Information")

        if metadata:
            for key, value in metadata.items():
                if key.lower() not in ["embedding", "vector"]:  # Skip large vector data
                    st.write(f"**{key}:** {value}")
        else:
            st.write("No metadata available")


def create_thumbnail_gallery(images, on_click_func=None, size=(150, 150)):
    """
    Create a gallery of clickable image thumbnails

    Args:
        images: List of file-like objects containing image data
        on_click_func: Function to call when thumbnail is clicked (passed index)
        size: Size of thumbnails (width, height)
    """
    # Create a container for the gallery
    gallery = st.container()

    with gallery:
        cols = st.columns(5)  # 5 thumbnails per row

        for i, img_file in enumerate(images):
            col_idx = i % 5

            with cols[col_idx]:
                img_file.seek(0)
                img = Image.open(img_file).convert("RGB")
                img.thumbnail(size)

                # Convert to base64 for HTML
                img_str = image_to_base64(img)

                # Create clickable thumbnail
                if on_click_func:
                    if st.button("📷", key=f"thumb_{i}"):
                        on_click_func(i)

                st.markdown(                    f"""
                    <div style="text-align: center">
                        <img src="data:image/jpeg;base64,{img_str}"
                             style="max-height: {size[1]}px; max-width: {size[0]}px; margin: 5px; border-radius: 5px;">
                        <p style="font-size: 0.8em; margin-top: 0;">{i}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def styled_container(
    content_function, border_color="#eeeeee", padding="1em", border_radius="0.5em"
):
    """
    Create a styled container for content

    Args:
        content_function: Function that defines the content inside the container
        border_color: Color of the container border
        padding: Padding inside the container
        border_radius: Border radius of the container
    """
    container = st.container()

    with container:
        st.markdown(            f"""
            <div style="border: 1px solid {border_color}; padding: {padding};
                      border-radius: {border_radius}; margin-bottom: 1em;">
            </div>
            """,
            unsafe_allow_html=True,
        )

        content_function()


def show_agent_thinking(thinking_text):
    """
    Display the agent's thinking process in a styled container

    Args:
        thinking_text: The thinking process to display
    """
    with st.expander("🤔 Agent Thinking Process", expanded=False):
        st.code(thinking_text, language="markdown")


def display_image_comparison(img1_file, img2_file, labels=("Image 1", "Image 2")):
    """
    Display two images side by side for comparison

    Args:
        img1_file: First image file
        img2_file: Second image file
        labels: Tuple of labels for the images
    """
    col1, col2 = st.columns(2)

    with col1:
        img1_file.seek(0)
        st.image(img1_file, use_container_width=True, caption=labels[0])

    with col2:
        img2_file.seek(0)
        st.image(img2_file, use_container_width=True, caption=labels[1])


def format_agent_response(response):
    """
    Format the agent's response with enhanced styling

    Args:
        response: Raw response from the agent

    Returns:
        Formatted response with markdown styling
    """
    # Apply formatting based on content
    parts = response.split("\n\n")
    formatted_parts = []

    for part in parts:
        # Add headers for sections
        if ":" in part and len(part.split(":")[0]) < 30:
            title = part.split(":")[0].strip()
            content = ":".join(part.split(":")[1:]).strip()
            formatted_parts.append(f"### {title}\n{content}")
        else:
            formatted_parts.append(part)

    return "\n\n".join(formatted_parts)


def display_color_palette(colors, labels=None):
    """
    Display a color palette with hex codes

    Args:
        colors: List of RGB tuples or hex strings
        labels: Optional labels for each color
    """
    # Convert RGB tuples to hex if needed
    hex_colors = []
    for color in colors:
        if isinstance(color, tuple) and len(color) == 3:
            hex_colors.append(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
        else:
            hex_colors.append(color)    # Create HTML for color palette
    html = '<div style="display: flex; flex-wrap: wrap;">'

    for i, color in enumerate(hex_colors):
        label = labels[i] if labels and i < len(labels) else color
        html += f"""
        <div style="margin: 10px; text-align: center;">
            <div style="width: 80px; height: 80px; background-color: {color};
                       border-radius: 5px; border: 1px solid #ddd;"></div>
            <p style="margin-top: 5px; font-size: 0.8em;">{label}</p>
        </div>
        """

    html += "</div>"

    # Display in Streamlit
    st.markdown(html, unsafe_allow_html=True)


def create_progress_tracker(steps, current_step):
    """
    Create a visual progress tracker for multi-step processes

    Args:
        steps: List of step names
        current_step: Index of current step (0-based)
    """
    # Create HTML for progress tracker
    html = '<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">'

    for i, step in enumerate(steps):
        status = (
            "completed"
            if i < current_step
            else "current"
            if i == current_step
            else "upcoming"
        )
        color = (
            "#4CAF50"
            if status == "completed"
            else "#2196F3"
            if status == "current"
            else "#ddd"
        )

        html += f"""
        <div style="text-align: center; flex: 1;">
            <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {color};
                      display: flex; align-items: center; justify-content: center; margin: 0 auto;
                      color: white; font-weight: bold;">{i + 1}</div>
            <p style="margin-top: 5px; font-size: 0.8em; color: {color};">{step}</p>
        </div>
        """

        # Add connecting line except for the last item
        if i < len(steps) - 1:
            line_color = "#4CAF50" if i < current_step else "#ddd"
            html += f'<div style="flex-grow: 0.5; height: 2px; background-color: {line_color}; margin-top: 15px;"></div>'

    html += "</div>"

    # Display in Streamlit
    st.markdown(html, unsafe_allow_html=True)


def plot_image_clusters(embeddings, cluster_labels, images=None, figsize=(12, 10)):
    """
    Create an interactive plot of image clusters with visible thumbnails

    Args:
        embeddings: 2D array of reduced embeddings (e.g., t-SNE result)
        cluster_labels: Array of cluster assignments
        images: Optional list of image files to show thumbnails
        figsize: Figure size as (width, height)
    """
    # Create figure with much larger size
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Get unique clusters
    clusters = set(cluster_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

    # Create custom plotting area with normalized coordinates
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # Normalize embeddings to 0-1 range for better control
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    x_range = max(0.1, x_max - x_min)
    y_range = max(0.1, y_max - y_min)

    norm_embeddings = np.zeros_like(embeddings)
    norm_embeddings[:, 0] = (embeddings[:, 0] - x_min) / x_range
    norm_embeddings[:, 1] = (embeddings[:, 1] - y_min) / y_range

    # Plot points first (very small and transparent)
    for cluster_id, color in zip(clusters, colors):
        mask = cluster_labels == cluster_id
        ax.scatter(
            norm_embeddings[mask, 0],
            norm_embeddings[mask, 1],
            c=[color],
            label=f"Cluster {cluster_id}",
            alpha=0.2,
            s=25,
        )

    # Add image thumbnails with fixed size
    if images:
        thumbnail_size = 0.05  # Fixed size in normalized coordinates

        for i, (x, y) in enumerate(norm_embeddings):
            if i < len(images):
                try:
                    # Reset file pointer
                    images[i].seek(0)

                    # Open image and convert to RGB
                    img = Image.open(images[i]).convert("RGB")

                    # Create a thumbnail
                    img.thumbnail((50, 50))  # Much larger thumbnail
                    img_array = np.array(img)

                    # Calculate a small offset based on index to avoid overlaps
                    jitter_x = ((i % 5) - 2) * 0.01
                    jitter_y = ((i // 5) % 5 - 2) * 0.01

                    # Add the image to the plot with fixed size
                    extent = [
                        x - thumbnail_size + jitter_x,
                        x + thumbnail_size + jitter_x,
                        y - thumbnail_size + jitter_y,
                        y + thumbnail_size + jitter_y,
                    ]
                    ax.imshow(img_array, extent=extent, zorder=100 + i)

                    # Add a border around the image in cluster color
                    rect = plt.Rectangle(
                        (extent[0], extent[2]),
                        extent[1] - extent[0],
                        extent[3] - extent[2],
                        linewidth=1.5,
                        edgecolor=colors[cluster_labels[i]],
                        facecolor="none",
                        zorder=100 + i + 0.5,
                    )
                    ax.add_patch(rect)

                    # Add small text with image index
                    ax.text(
                        x,
                        y - thumbnail_size - 0.02,
                        f"{i}",
                        ha="center",
                        va="top",
                        fontsize=8,
                        bbox=dict(
                            facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"
                        ),
                    )
                except Exception as e:
                    print(f"Error displaying thumbnail {i}: {str(e)}")

    # Add legend and labels
    ax.legend(loc="upper right")
    ax.set_title("Image Cluster Visualization", fontsize=16)
    ax.grid(alpha=0.2)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    # Hide axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Display with Streamlit
    st.pyplot(fig)


def create_empty_file_placeholder():
    """
    Create a placeholder for file upload when no files are present
    """
    st.markdown(
        """
        <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 2em;
                  text-align: center; margin-bottom: 1em;">
            <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#ccc" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="12" y1="12" x2="12" y2="18"></line>
                <line x1="9" y1="15" x2="15" y2="15"></line>
            </svg>
            <p style="color: #666; margin-top: 1em;">No images uploaded yet</p>
            <p style="color: #999; font-size: 0.8em;">Drag and drop images here or click to upload</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_image_card(img_file, title=None, metadata=None):
    """
    Create a card-style display for an image with metadata

    Args:
        img_file: File-like object containing image data
        title: Optional title for the card
        metadata: Dictionary of metadata to display
    """
    img_file.seek(0)
    img = Image.open(img_file)

    # Get image dimensions
    width, height = img.size

    # Create card container
    st.markdown(
        """
        <div style="border: 1px solid #ddd; border-radius: 10px; overflow: hidden;
                  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); margin-bottom: 1em;">
        """,
        unsafe_allow_html=True,
    )

    # Display image
    st.image(img_file, use_container_width=True)

    # Display title and metadata
    with st.container():
        if title:
            st.markdown(
                f"<h3 style='margin-top: 0.5em;'>{title}</h3>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h3 style='margin-top: 0.5em;'>{img_file.name}</h3>",
                unsafe_allow_html=True,
            )

        st.markdown(f"<p>Dimensions: {width}x{height}</p>", unsafe_allow_html=True)

        if metadata:
            st.write("**Metadata:**")
            for key, value in metadata.items():
                if key.lower() not in ["embedding", "vector"]:  # Skip large data
                    st.write(f"• **{key}:** {value}")

    # Close card container
    st.markdown("</div>", unsafe_allow_html=True)
