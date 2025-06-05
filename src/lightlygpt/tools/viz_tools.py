# Tools for visualizing image relationships

import warnings

import numpy as np
import streamlit as st

from ..utils import setup_project_path
from ..utils.session_utils import get_active_indices
from ..utils.ui_utils import plot_image_clusters

warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Ensure project path is set up
setup_project_path()

# Try to import visualization dependencies with error handling
try:
    # import matplotlib.pyplot as plt  # Removed unused import to resolve F401
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    st.warning("matplotlib not found. Visualization features will be disabled.")
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    st.warning(
        "scikit-learn not found. Clustering and visualization features will be disabled."
    )
    SKLEARN_AVAILABLE = False


def create_tsne_visualization(selection_or_n_clusters="3"):
    """
    Create t-SNE visualization of image relationships

    Args:
        selection_or_n_clusters: Can be a number of clusters (default), 'use last filtered set', or a comma-separated list of indices

    Returns:
        String with embedded visualization
    """
    # Check for dependencies
    if not MATPLOTLIB_AVAILABLE:
        return "Visualization requires matplotlib. Please install it with 'pip install matplotlib'."

    if not SKLEARN_AVAILABLE:
        return "t-SNE visualization requires scikit-learn. Please install it with 'pip install scikit-learn'."

    # Check session state
    if not hasattr(st.session_state, "processed") or not st.session_state.processed:
        return "Please process the images first."

    if (
        not hasattr(st.session_state, "collection")
        or st.session_state.collection is None
    ):
        return "No image collection found. Please process images first."

    # Check if ChromaDB client is available
    if (
        not hasattr(st.session_state, "chroma_client")
        or st.session_state.chroma_client is None
    ):
        return "ChromaDB client not available. Please check your SQLite version compatibility."

    if (
        not hasattr(st.session_state, "uploaded_images")
        or not st.session_state.uploaded_images
    ):
        return "No images uploaded. Please upload images first."

    try:
        # Determine if the input is a special reference or a number
        # Use get_active_indices for filtered set
        if isinstance(
            selection_or_n_clusters, str
        ) and selection_or_n_clusters.strip().lower() in [
            "use last filtered set",
            "use last",
            "previous",
            "last",
        ]:
            filtered_indices = get_active_indices()
            n_clusters = 3
        elif isinstance(selection_or_n_clusters, str) and all(
            x.strip().isdigit() for x in selection_or_n_clusters.split(",")
        ):
            filtered_indices = [
                int(x.strip())
                for x in selection_or_n_clusters.split(",")
                if x.strip().isdigit()
            ]
            n_clusters = 3
        else:
            filtered_indices = get_active_indices()
            try:
                n_clusters = min(10, max(1, int(selection_or_n_clusters)))
            except Exception:
                n_clusters = 3

        # Get all embeddings from ChromaDB
        results = st.session_state.collection.get(
            include=["embeddings", "metadatas"], where={"source": "current_session"}
        )

        if not results or len(results["embeddings"]) == 0:
            return "No embeddings found. Please process images first."

        # Extract embeddings and metadata
        embeddings = np.array(results["embeddings"])
        metadatas = results["metadatas"]

        if filtered_indices is not None and len(filtered_indices) > 0:
            embeddings = embeddings[filtered_indices]
            metadatas = [metadatas[i] for i in filtered_indices]
            images = [st.session_state.uploaded_images[i] for i in filtered_indices]
        elif filtered_indices is not None and len(filtered_indices) == 0:
            return "No images to visualize from the previous filter. Please run a filter tool first."
        else:
            images = st.session_state.uploaded_images

        # Create t-SNE projection
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(3, len(embeddings) - 1)),
            random_state=42,
            learning_rate=200,
        )
        projections = tsne.fit_transform(embeddings)

        # Apply KMeans clustering if n_clusters > 1
        if n_clusters > 1:
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10
            )
            clusters = kmeans.fit_predict(
                embeddings
            )  # Use original embeddings for better clustering
        else:
            # Single cluster case
            clusters = np.zeros(len(projections), dtype=int)

        # Use our UI utility to create an interactive plot with thumbnails
        plot_image_clusters(projections, clusters, images)

        return f"t-SNE visualization created with {n_clusters} clusters. Images that are closer together are more similar visually."

    except Exception as e:
        return f"An error occurred creating visualization: {str(e)}"


def create_image_clusters(n_clusters_str="4"):
    """
    Group similar images into clusters

    Args:
        n_clusters_str: Number of clusters to create (2-10)

    Returns:
        String describing image clusters
    """
    # Check for dependencies
    if not SKLEARN_AVAILABLE:
        return "Clustering requires scikit-learn. Please install it with 'pip install scikit-learn'."

    # Check session state
    if not hasattr(st.session_state, "processed") or not st.session_state.processed:
        return "Please process the images first."

    if (
        not hasattr(st.session_state, "collection")
        or st.session_state.collection is None
    ):
        return "No image collection found. Please process images first."

    # Check if ChromaDB client is available
    if (
        not hasattr(st.session_state, "chroma_client")
        or st.session_state.chroma_client is None
    ):
        return "ChromaDB client not available. Please check your SQLite version compatibility."

    if (
        not hasattr(st.session_state, "uploaded_images")
        or not st.session_state.uploaded_images
    ):
        return "No images uploaded. Please upload images first."

    try:
        # Parse number of clusters
        try:
            n_clusters = min(10, max(2, int(n_clusters_str)))
        except Exception:
            n_clusters = 4

        # Get all embeddings from ChromaDB
        results = st.session_state.collection.get(
            include=["embeddings", "metadatas"], where={"source": "current_session"}
        )

        if not results or len(results["embeddings"]) == 0:
            return "No embeddings found. Please process images first."

        # Extract embeddings and metadata
        embeddings = np.array(results["embeddings"])
        metadatas = results["metadatas"]
        # Filter embeddings and metadatas to active_indices
        active_indices = get_active_indices()
        embeddings = embeddings[active_indices]
        metadatas = [metadatas[i] for i in active_indices]

        # Apply KMeans clustering
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10
        )
        clusters = kmeans.fit_predict(embeddings)

        # Group images by cluster
        clustered_images = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_images:
                clustered_images[cluster_id] = []

            idx = int(metadatas[i]["index"])
            filename = st.session_state.uploaded_images[idx].name
            clustered_images[cluster_id].append((idx, filename))

        # Create a t-SNE projection for visualization
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(3, len(embeddings) - 1)),
            random_state=42,
            learning_rate=200,
        )
        projections = tsne.fit_transform(embeddings)

        # Use our UI utility to create an interactive plot
        images = [
            st.session_state.uploaded_images[int(metadatas[i]["index"])]
            for i in range(len(metadatas))
        ]
        plot_image_clusters(projections, clusters, images)

        # Generate response
        response = f"Images grouped into {len(clustered_images)} clusters:\n\n"

        for cluster_id, images in clustered_images.items():
            response += f"Cluster {cluster_id + 1}:\n"
            for idx, filename in images:
                response += f"  Image {idx}: {filename}\n"
            response += "\n"

        return response

    except Exception as e:
        return f"An error occurred creating clusters: {str(e)}"
