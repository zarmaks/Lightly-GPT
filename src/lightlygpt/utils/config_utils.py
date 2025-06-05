# Configuration utilities for LightlyGPT

import importlib

import streamlit as st
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI

from ..tools.clip_tools import clip_image_search_tool

# Tool configuration - makes it easier to add/remove tools
TOOLS_CONFIG = {
    "image_search": {
        "name": "clip_image_search",
        "func_name": "clip_image_search_tool",
        "description": "Search for images containing specific objects, people, animals, scenes, or activities using CLIP embeddings. Use this for ANY content-based search including: people, faces, animals, objects, activities, scenes, etc. Examples: 'people', 'cats', 'cars', 'mountains', 'food', 'beaches'. This is the PRIMARY tool for finding image content.",
        "import_from": "lightlygpt.tools.clip_tools",
    },
    "color_analysis": {
        "name": "ColorAnalysis",
        "func_name": "analyze_image_colors",
        "description": "Analyze and extract the dominant colors from images. Use this ONLY when user specifically asks about colors, color palette, or color analysis. Examples: 'what colors', 'dominant colors', 'color scheme'.",
        "import_from": "lightlygpt.tools.analysis_tools",
    },
    "bw_detection": {
        "name": "BWDetection",
        "func_name": "detect_bw_images",
        "description": "Detect which images are black and white or grayscale. Use this ONLY when user specifically asks to identify/detect/find black and white, grayscale, or monochrome images. Do NOT use for any other content searches.",
        "import_from": "lightlygpt.tools.analysis_tools",
    },
    "duplicate_detection": {
        "name": "DuplicateDetection",
        "func_name": "find_duplicate_images",
        "description": "Find duplicate or very similar images in the collection. Use this ONLY when user asks to find duplicates, similar images, or wants to clean up their collection.",
        "import_from": "lightlygpt.tools.duplicate_tools",
    },
    "datetime_filter": {
        "name": "DateTimeFilter",
        "func_name": "filter_by_datetime",
        "description": "Filter images by the date/time they were taken (from EXIF data). Use this when user asks to filter by date, time period, or specific dates.",
        "import_from": "lightlygpt.tools.exif_tools",
    },
    "location_filter": {
        "name": "LocationFilter",
        "func_name": "filter_by_location",
        "description": "Filter images by the location where they were taken (from GPS EXIF data). Use this when user asks to filter by location, place names, or coordinates.",
        "import_from": "lightlygpt.tools.exif_tools",
    },
    "image_visualization": {
        "name": "ImageVisualization",
        "func_name": "create_tsne_visualization",
        "description": "Create a t-SNE visualization showing relationships between images. Use this when user asks to visualize, plot, or see relationships between images.",
        "import_from": "lightlygpt.tools.viz_tools",
    },
    "image_clustering": {
        "name": "ImageClustering",
        "func_name": "create_image_clusters",
        "description": "Group similar images into clusters using K-means. Use this when user asks to group, cluster, or organize similar images together.",
        "import_from": "lightlygpt.tools.viz_tools",
    },
}


def enhanced_clip_search_wrapper(query):
    """Enhanced wrapper that uses UI settings automatically"""
    # Get settings from UI (sidebar)
    if hasattr(st.session_state, 'search_settings'):
        settings = st.session_state.search_settings
        threshold = settings.get('threshold', 0.8)
        use_dynamic = settings.get('use_dynamic', True)
        percentile = settings.get('percentile', 70)
        use_temperature_scaling = settings.get('use_temperature_scaling', True)
        temperature = settings.get('temperature', 0.15)

        # Show user what settings are being used
        if settings.get('mode') == 'dynamic':
            mode_info = f"**{settings.get('sensitivity', 'Balanced')}** search sensitivity"
        else:
            mode_info = f"**static threshold**: {threshold:.2f}"

        contrast_info = f" with {'**Enhanced Contrast**' if use_temperature_scaling else 'standard contrast'}"
        st.info(f"🔍 Using {mode_info}{contrast_info}")

    else:
        # Fallback defaults
        threshold = 0.8
        use_dynamic = True
        percentile = 70
        use_temperature_scaling = True
        temperature = 0.15
        st.info("🔍 Using **default** search settings with **Enhanced Contrast**")

    return clip_image_search_tool(query, threshold, use_dynamic, percentile, use_temperature_scaling, temperature)


def initialize_agent_tools():
    """
    Initialize LangChain agent with tools for image analysis using the configuration
    """
    tools = []

    # Create tools from configuration
    for tool_key, config in TOOLS_CONFIG.items():
        if tool_key == "image_search":
            # Use enhanced wrapper for image search
            tools.append(
                Tool(
                    name="clip_image_search",
                    description=(
                        "🔍 PRIMARY CONTENT SEARCH TOOL: Search for images containing ANY type of content using CLIP embeddings. "
                        "Use this for people, animals, objects, scenes, activities, etc. "
                        "The tool automatically uses optimized search settings from the sidebar. "
                        "Simply provide a text description of what you're looking for. "
                        "Examples: 'people', 'cats', 'dogs', 'cars', 'mountains', 'food', 'beaches', 'smiling faces'"
                    ),
                    func=enhanced_clip_search_wrapper
                )
            )
        else:
            # Import the module containing the tool function
            module = importlib.import_module(config["import_from"])
            # Get the actual function
            func = getattr(module, config["func_name"])
            # Create the tool
            tools.append(
                Tool(name=config["name"], func=func, description=config["description"])
            )

    # Setup memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize the agent with the loaded tools and enhanced prompt
    try:
        models_to_try = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-4o-mini",
            "gpt-4.1-nano",
        ]
        llm = None
        for model in models_to_try:
            try:
                llm = ChatOpenAI(model=model, temperature=0.1)  # Lower temperature for more consistent tool selection
                llm.predict("test")
                st.success(f"✅ Successfully initialized with model: {model}")
                break
            except Exception as model_error:
                st.warning(f"Model {model} failed: {str(model_error)}")
                continue

        if llm is None:
            raise Exception("No working models available")

        # Create agent with enhanced system message
        system_message = get_agent_prompt()

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": system_message
            }
        )

        st.session_state.agent = agent
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None


def get_agent_prompt():
    """Enhanced agent prompt with clear tool selection guidelines"""
    return """
    You are an intelligent image analysis assistant. Use these tools correctly:

    **PRIMARY CONTENT SEARCH TOOL:**
    - **clip_image_search**: Use this for ANY content-based searches including people, animals, objects, scenes, activities, etc.
      Examples: "people", "cats", "dogs", "cars", "mountains", "food", "smiling faces", "red cars"

    **SPECIALIZED TOOLS:**
    - **BWDetection**: ONLY for finding black & white/grayscale images
    - **ColorAnalysis**: ONLY for analyzing colors/color palettes
    - **DuplicateDetection**: ONLY for finding duplicate images
    - **DateTimeFilter**: ONLY for filtering by date/time
    - **LocationFilter**: ONLY for filtering by location
    - **ImageVisualization**: ONLY for creating t-SNE plots
    - **ImageClustering**: ONLY for grouping similar images

    **DECISION RULES:**
    - If user asks about ANY image content (people, animals, objects, etc.) → Use clip_image_search
    - If user asks "do I have people?" → Use clip_image_search("people")
    - If user asks "show me cats" → Use clip_image_search("cats")
    - If user asks "find cars" → Use clip_image_search("cars")
    - Only use specialized tools when explicitly requested

    **Examples:**
    - "Do I have any people?" → clip_image_search("people")
    - "Show me animals" → clip_image_search("animals")
    - "Find black and white images" → BWDetection()
    - "What colors are in my images?" → ColorAnalysis()
    """
