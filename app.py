# Main application file for Agent_4o - CLIP-based Image Analysis with LangChain Agent

import os
import io
import base64
import torch
import streamlit as st
import chromadb
from datetime import datetime
from PIL import Image, ExifTags
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Add the project root to Python path for more reliable imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our custom modules
from utils.clip_utils import ensure_clip_model_loaded, generate_clip_embedding_generic
from utils.ui_utils import display_image_grid, show_agent_thinking, format_agent_response, create_image_card
from tools.clip_tools import clip_image_search_tool
from tools.analysis_tools import analyze_image_colors, detect_bw_images
from tools.duplicate_tools import find_duplicate_images
from tools.exif_tools import filter_by_datetime, filter_by_location
from tools.viz_tools import create_tsne_visualization, create_image_clusters

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="LightlyGPT - Agentic AI tool for Image Analysis",
    page_icon="🔆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chroma_client' not in st.session_state:
    # Create a data directory for ChromaDB
    home_dir = os.path.expanduser("~")
    chroma_data_dir = os.path.join(home_dir, "agent4o_clip_data")
    os.makedirs(chroma_data_dir, exist_ok=True)
    st.session_state.chroma_client = chromadb.PersistentClient(
        path=chroma_data_dir,
        settings=chromadb.Settings(anonymized_telemetry=False)
    )
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'image_metadata' not in st.session_state:
    st.session_state.image_metadata = {}
if 'clip_model' not in st.session_state:
    st.session_state.clip_model = None
if 'clip_processor' not in st.session_state:
    st.session_state.clip_processor = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None

# Function to validate file extensions in session state
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

# Validate files in session state
validate_session_files()

# Function to initialize agent tools
def initialize_agent_tools():
    """Initialize LangChain agent with tools for image analysis"""
    
    # Define tools
    tools = [
        Tool(
            name="ImageSearch",
            func=clip_image_search_tool,
            description="Search for images matching a description or concept using CLIP embeddings. Input should be a natural language description of what you're looking for."
        ),
        Tool(
            name="ColorAnalysis",
            func=analyze_image_colors,
            description="Analyze the dominant colors in specified images. Input should be a comma-separated list of image indices."
        ),
        Tool(
            name="BWDetection",
            func=detect_bw_images,
            description="Detect which images are black and white or grayscale. No input required."
        ),
        Tool(
            name="DuplicateDetection",
            func=find_duplicate_images,
            description="Find duplicate or very similar images. Input should be a similarity threshold (1-10, where 10 is most strict)."
        ),
        Tool(
            name="DateTimeFilter",
            func=filter_by_datetime,
            description="Filter images by date/time they were taken. Input should be a date range in format 'YYYY-MM-DD to YYYY-MM-DD'."
        ),
        Tool(
            name="LocationFilter",
            func=filter_by_location,
            description="Filter images by location. Input should be either coordinates or a place name."
        ),
        Tool(
            name="ImageVisualization",
            func=create_tsne_visualization,
            description="Create a t-SNE visualization showing relationships between images. No input required."
        ),
        Tool(
            name="ImageClustering",
            func=create_image_clusters,
            description="Group similar images into clusters. Input should be the number of clusters (2-10)."
        ),
    ]
    
    # Setup memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize the agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    
    st.session_state.agent = agent
    return agent

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
        placeholder="sk-..."
    )

with col2:
    if st.button("Validate Key"):
        if api_key_input and api_key_input.startswith("sk-"):
            st.success("✅ Valid API key format")
        else:
            st.error("❌ Invalid API key format")

# Stop app execution if no API key is provided
if not api_key_input:
    st.info("👆 You need an OpenAI API key to use this app. Visit https://platform.openai.com/api-keys to get one.")
    st.stop()
else:
    # Set API key in environment variables
    os.environ["OPENAI_API_KEY"] = api_key_input

# File uploader section
st.subheader("📁 Upload Images")
uploaded_files = st.file_uploader(
    "Upload your images (JPG, PNG, etc.)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True
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
                        st.session_state.collection = st.session_state.chroma_client.get_collection(collection_name)
                        # Clear the collection to start fresh
                        st.session_state.collection.delete(where={"source": "current_session"})
                    except:
                        # Create a new collection if it doesn't exist
                        st.session_state.collection = st.session_state.chroma_client.create_collection(
                            name=collection_name,
                            embedding_function=None  # We'll provide our own embeddings
                        )
                    
                    # Process each image
                    for i, img_file in enumerate(st.session_state.uploaded_images):
                        try:
                            # Generate CLIP embedding
                            embedding = generate_clip_embedding_generic(img_file, is_image=True)
                            
                            # Extract image metadata
                            img_file.seek(0)
                            img = Image.open(img_file)
                            
                            # Store in ChromaDB
                            st.session_state.collection.add(
                                ids=[f"img_{i}"],
                                embeddings=[embedding.tolist()],
                                metadatas=[{
                                    "filename": img_file.name,
                                    "index": i,
                                    "source": "current_session"
                                }]
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
                    # Store the agent's verbose output for visualization
                    with io.StringIO() as thinking_buffer:
                        # Create a temporary print function that writes to our buffer
                        original_print = print
                        def verbose_print(*args, **kwargs):
                            thinking_buffer.write(" ".join(map(str, args)) + "\n")
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
                    st.markdown(formatted_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"I encountered an error: {str(e)}"})

# Add explanation in sidebar
with st.sidebar:
    # Add company logo at the top of the sidebar
    st.image(r"C:\Users\zarma\Downloads\Lightly_logo_small.png", width=200)
    
    st.subheader("LightlyGPT")
    st.markdown("""
    **LightlyGPT** is an AI-powered image analysis tool that combines:
    
    1. **CLIP Model**: For understanding image content
    2. **GPT-4o**: For intelligent reasoning about images
    3. **LangChain**: For managing complex workflows
    4. **ChromaDB**: For efficient image storage and retrieval  
    5. **Agent Architecture**: To choose the right tool for each task
    
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
    - "What are the dominant colors in my collection?"
    - "Are there any black and white photos?"
    - "Find duplicate images with threshold 5"
    - "Show me a visualization of my image collection"
    - "Group similar images into 3 clusters"
    """)
