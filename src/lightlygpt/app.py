# Main application file for Agent_4o - CLIP-based Image Analysis with LangChain Agent

import os
import sys
import io
import warnings
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from lightlygpt.utils.clip_utils import ensure_clip_model_loaded, generate_clip_embedding_generic
from lightlygpt.utils.ui_utils import (
    display_image_grid,
    show_agent_thinking,
    format_agent_response,
)
from lightlygpt.utils.image_utils import show_dependency_warnings
from lightlygpt.utils.session_utils import initialize_session_state, validate_session_files
from lightlygpt.utils.config_utils import initialize_agent_tools

# Import torch with error handling to prevent Streamlit watcher issues
try:
    import torch

    # Disable torch JIT to avoid Streamlit compatibility issues
    torch.jit._state.disable()
except Exception:
    pass

warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Load environment variables
load_dotenv()

# Show dependency warnings early
show_dependency_warnings()

# Configure page
st.set_page_config(
    page_title="LightlyGPT - Agentic AI tool for Image Analysis",
    page_icon="🔆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize all session state variables centrally
initialize_session_state()

# Validate files in session state
validate_session_files()

# Main title
st.title("🔆 LightlyGPT - Agentic AI tool for Image Analysis")

# API Key Section
st.subheader("🔑 OpenAI API Key")
# Check if OpenAI API key exists in environment variables
api_key = os.environ.get("OPENAI_API_KEY")

col1, col2 = st.columns([3, 1])
with col1:
    api_key_input = st.text_input(
        "Enter your OpenAI API key:",
        value=api_key if api_key else "",
        type="password",
        placeholder="sk-...",
    )

with col2:
    if st.button("Validate Key"):
        if api_key_input and api_key_input.startswith("sk-"):
            # Test the API key by making a simple request
            try:
                import openai

                openai.api_key = api_key_input
                # Test with a minimal request
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                )
                st.success("✅ Valid and working API key")
            except Exception as e:
                st.error(f"❌ API key error: {str(e)}")
        else:
            st.error("❌ Invalid API key format")

# Stop app execution if no API key is provided
if not api_key_input:
    st.info(
        "👆 You need an OpenAI API key to use this app. Visit https://platform.openai.com/api-keys to get one."
    )
    st.stop()
else:
    # Set API key in environment variables
    os.environ["OPENAI_API_KEY"] = api_key_input

# File uploader section
st.subheader("📁 Upload Images")
uploaded_files = st.file_uploader(
    "Upload your images (JPG, PNG, etc.)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

# Store uploaded files in session state
if uploaded_files:
    # Add any new files to session state
    new_files = []
    existing_filenames = [f.name for f in st.session_state.uploaded_images]

    for file in uploaded_files:
        if file.name not in existing_filenames:
            new_files.append(file)

    # Add new files to existing ones
    st.session_state.uploaded_images.extend(new_files)

    # Reset processed flag if new files are added
    if new_files:
        st.session_state.processed = False

# Show active filtered set count
if st.session_state.uploaded_images:
    active_count = (
        len(st.session_state.last_filtered_indices)
        if hasattr(st.session_state, "last_filtered_indices")
        and st.session_state.last_filtered_indices
        else len(st.session_state.uploaded_images)
    )
    st.info(f"Active images: {active_count} / {len(st.session_state.uploaded_images)}")

# Display uploaded images in a grid
if st.session_state.uploaded_images:
    st.subheader(f"📊 Uploaded Images ({len(st.session_state.uploaded_images)})")

    # Use our custom UI utility for displaying images in a grid
    display_image_grid(st.session_state.uploaded_images, num_columns=4)

    # Process button
    if not st.session_state.processed:
        if st.button("Process Images with CLIP"):
            with st.spinner("Processing images with CLIP..."):
                # Ensure CLIP model is loaded
                if ensure_clip_model_loaded():
                    # Create or get collection
                    collection_name = "clip_images"
                    try:
                        # Always delete the collection to ensure correct metric
                        st.session_state.chroma_client.delete_collection(
                            collection_name
                        )
                    except Exception:
                        pass  # Collection may not exist yet
                    st.session_state.collection = (
                        st.session_state.chroma_client.create_collection(
                            name=collection_name,
                            embedding_function=None,  # We'll provide our own embeddings
                            metadata={
                                "hnsw:space": "cosine"
                            },  # Use cosine distance for CLIP
                        )
                    )

                    # Process each image
                    for i, img_file in enumerate(st.session_state.uploaded_images):
                        try:
                            # Generate CLIP embedding
                            embedding = generate_clip_embedding_generic(
                                img_file, is_image=True
                            )

                            # Extract image metadata
                            img_file.seek(0)
                            img = Image.open(img_file)

                            # Store in ChromaDB
                            st.session_state.collection.add(
                                ids=[f"img_{i}"],
                                embeddings=[embedding.tolist()],
                                metadatas=[
                                    {
                                        "filename": img_file.name,
                                        "index": i,
                                        "source": "current_session",
                                    }
                                ],
                            )

                            # Update progress
                            st.progress((i + 1) / len(st.session_state.uploaded_images))

                        except Exception as e:
                            st.error(f"Error processing {img_file.name}: {str(e)}")
                    st.session_state.processed = True
                    st.success("✅ All images processed and indexed!")

                    # Initialize the agent after processing
                    initialize_agent_tools()

                    # Rerun to update the UI
                    st.rerun()

# Initialize agent if images are processed but agent isn't ready
if st.session_state.processed and st.session_state.agent is None:
    initialize_agent_tools()

# Chat interface - only show if images have been processed
if st.session_state.processed:
    st.subheader("💬 Chat with Your Images")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about your images..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing images..."):
                try:
                    # Check API key before making requests
                    if not api_key_input or not api_key_input.startswith("sk-"):
                        raise Exception("Invalid API key format")

                    # Store the agent's verbose output for visualization
                    with io.StringIO() as thinking_buffer:
                        # Create a temporary print function that writes to our buffer
                        original_print = print

                        def verbose_print(*args, **kwargs):
                            try:
                                thinking_buffer.write(" ".join(map(str, args)) + "\n")
                            except ValueError:
                                pass  # Ignore if buffer is closed
                            original_print(*args, **kwargs)

                        # Replace print temporarily to capture thinking
                        import builtins

                        builtins.print = verbose_print
                        # Run the agent
                        response = st.session_state.agent.run(prompt)
                        # Restore original print
                        builtins.print = original_print
                        # Get the thinking process
                        thinking_text = thinking_buffer.getvalue()
                    # Show the agent thinking in an expander
                    show_agent_thinking(thinking_text)
                    # Show formatted response
                    formatted_response = format_agent_response(response)
                    st.markdown(
                        formatted_response
                    )  # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    error_msg = str(e)
                    if "401" in error_msg or "not_authorized" in error_msg:
                        st.error("🚨 **API Key Issue**")
                        st.error(
                            "Your OpenAI API key is invalid, expired, or the project has been archived."
                        )
                        st.info("**Solutions:**")
                        st.info(
                            "1. Get a new API key from https://platform.openai.com/api-keys"
                        )
                        st.info("2. Check your OpenAI account billing status")
                        st.info("3. Make sure your project is active (not archived)")
                    else:
                        st.error(f"An error occurred: {error_msg}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"I encountered an error: {error_msg}",
                        }
                    )

# Add explanation in sidebar
with st.sidebar:
    # Add company logo at the top of the sidebar
    logo_path = os.path.join("assets", "Lightly_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)

    st.subheader("LightlyGPT")
    st.markdown("""
    **LightlyGPT** is an AI-powered image analysis tool that combines:

    1. **CLIP Model**: For understanding image content
    2. **gpt-4o-mini**: For intelligent reasoning about images
    3. **LangChain**: For managing complex workflows
    4. **ChromaDB**: For efficient image storage and retrieval
    5. **Agent Architecture**: To choose the right tool for each task
    """)

    # Add search settings section only if images are processed
    if st.session_state.processed:
        st.markdown("---")
        st.subheader("🔍 Search Settings")

        # Enhanced contrast toggle
        use_enhanced_contrast = st.checkbox(
            "🔥 Enhanced Contrast Mode",
            value=True,
            help="Uses temperature scaling to dramatically improve result quality"
        )
        
        if use_enhanced_contrast:
            temperature = st.select_slider(
                "Contrast Level:",
                options=[0.05, 0.1, 0.15, 0.2, 0.3],
                value=0.15,
                format_func=lambda x: {
                    0.05: "Maximum Contrast",
                    0.1: "High Contrast", 
                    0.15: "Balanced Contrast",
                    0.2: "Moderate Contrast",
                    0.3: "Low Contrast"
                }[x],
                help="Lower values = more dramatic contrast between relevant/irrelevant images"
            )
        else:
            temperature = 1.0  # No scaling
        
        # Threshold mode selection
        threshold_mode = st.radio(
            "Search Mode:",
            ["Dynamic (Recommended)", "Static"],
            help="Dynamic automatically adjusts based on results quality"
        )
        
        if threshold_mode == "Dynamic (Recommended)":
            # Dynamic mode with user-friendly options
            search_sensitivity = st.select_slider(
                "Search Sensitivity:",
                options=["Very Strict", "Strict", "Balanced", "Relaxed", "Very Relaxed"],
                value="Balanced",
                help="Controls how many results you get"
            )
            
            # Map user choices to technical parameters  
            sensitivity_mapping = {
                "Very Strict": {"percentile": 40, "max_threshold": 0.6},
                "Strict": {"percentile": 50, "max_threshold": 0.7},
                "Balanced": {"percentile": 60, "max_threshold": 0.8},
                "Relaxed": {"percentile": 70, "max_threshold": 0.85},
                "Very Relaxed": {"percentile": 80, "max_threshold": 0.9}
            }
            
            params = sensitivity_mapping[search_sensitivity]
            
            st.session_state.search_settings = {
                'use_dynamic': True,
                'percentile': params["percentile"],
                'threshold': params["max_threshold"],
                'mode': 'dynamic',
                'sensitivity': search_sensitivity,
                'use_temperature_scaling': use_enhanced_contrast,
                'temperature': temperature
            }
            
            # Show what this means
            expected_results = 100 - params['percentile']
            st.caption(f"💡 **{search_sensitivity}**: Expects ~{expected_results}% of best matching images")
            
            if use_enhanced_contrast:
                st.caption(f"🔥 **Enhanced contrast**: Temperature {temperature} for better separation")
            
        else:
            # Static mode with threshold slider
            static_threshold = st.slider(
                "Similarity Threshold:",
                0.3, 0.95, 0.8, 0.05,
                help="Lower = more strict (fewer results), Higher = more relaxed (more results)"
            )
            
            st.session_state.search_settings = {
                'use_dynamic': False,
                'threshold': static_threshold,
                'percentile': 70,
                'mode': 'static',
                'use_temperature_scaling': use_enhanced_contrast,
                'temperature': temperature
            }
            
            # Show similarity percentage
            similarity_pct = (1 - static_threshold) * 100
            st.caption(f"💡 **Minimum similarity**: {similarity_pct:.0f}%")
            
            if use_enhanced_contrast:
                st.caption(f"🔥 **Enhanced contrast**: Temperature {temperature} for better separation")
        
        st.markdown("---")
        st.markdown("""
    ### Available Capabilities:

    - **Find images** matching natural language descriptions
    - **Analyze colors** in your images
    - **Detect black & white** images automatically
    - **Find duplicate images** to clean up your collection
    - **Filter by date & location** using image metadata
    - **Visualize relationships** between images using t-SNE
    - **Group similar images** into meaningful clusters

    ### Try asking:

    - "Find images with people smiling"
    - "Search for cats"
    - "Find dogs"
    - "Look for cars"
    - "Search for flowers"
    - "What are the dominant colors in my collection?"
    - "Show me a visualization of my image collection"
    """)

    # Add reset filters button in sidebar
    if st.session_state.uploaded_images and st.button("🔄 Reset filters"):
        st.session_state.last_filtered_indices = list(
            range(len(st.session_state.uploaded_images))
        )
        st.success("Filters reset. All images are now active.")

    # In the sidebar, add a debug toggle
    if st.session_state.processed:
        st.markdown("---")
        debug_mode = st.checkbox("🔧 Debug Mode", help="Show agent decision-making process")
        if debug_mode:
            st.session_state.debug_mode = True
        else:
            st.session_state.debug_mode = False

def main():
    """Main entry point for the LightlyGPT application."""
    # The main app logic is already in the global scope above
    pass


if __name__ == "__main__":
    main()
