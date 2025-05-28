# Configuration utilities for LightlyGPT

from langchain.agents import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Tool configuration - makes it easier to add/remove tools
TOOLS_CONFIG = {
    "image_search": {
        "name": "ImageSearch",
        "func_name": "clip_image_search_tool",
        "description": "Search for images matching a description or concept using CLIP embeddings. Input should be a natural language description of what you're looking for.",
        "import_from": "tools.clip_tools"
    },
    "color_analysis": {
        "name": "ColorAnalysis",
        "func_name": "analyze_image_colors",
        "description": "Analyze the dominant colors in specified images. Input should be a comma-separated list of image indices.",
        "import_from": "tools.analysis_tools"
    },
    "bw_detection": {
        "name": "BWDetection",
        "func_name": "detect_bw_images",
        "description": "Detect which images are black and white or grayscale. No input required.",
        "import_from": "tools.analysis_tools"
    },
    "duplicate_detection": {
        "name": "DuplicateDetection",
        "func_name": "find_duplicate_images",
        "description": "Find duplicate or very similar images. Input should be a similarity threshold (1-10, where 10 is most strict).",
        "import_from": "tools.duplicate_tools"
    },
    "datetime_filter": {
        "name": "DateTimeFilter",
        "func_name": "filter_by_datetime",
        "description": "Filter images by date/time they were taken. Input should be a date range in format 'YYYY-MM-DD to YYYY-MM-DD'.",
        "import_from": "tools.exif_tools"
    },
    "location_filter": {
        "name": "LocationFilter",
        "func_name": "filter_by_location",
        "description": "Filter images by location. Input should be either coordinates or a place name.",
        "import_from": "tools.exif_tools"
    },
    "image_visualization": {
        "name": "ImageVisualization",
        "func_name": "create_tsne_visualization",
        "description": "Create a t-SNE visualization showing relationships between images. No input required.",
        "import_from": "tools.viz_tools"
    },
    "image_clustering": {
        "name": "ImageClustering",
        "func_name": "create_image_clusters",
        "description": "Group similar images into clusters. Input should be the number of clusters (2-10).",
        "import_from": "tools.viz_tools"
    }
}

def initialize_agent_tools():
    """
    Initialize LangChain agent with tools for image analysis using the configuration
    """
    # Import tool functions dynamically based on the configuration
    tools = []
    import importlib
    
    for tool_id, config in TOOLS_CONFIG.items():
        try:
            # Import the module containing the tool function
            module = importlib.import_module(config["import_from"])
            # Get the actual function
            func = getattr(module, config["func_name"])
            # Create the tool
            tools.append(Tool(
                name=config["name"],
                func=func,
                description=config["description"]
            ))
        except (ImportError, AttributeError) as e:
            st.warning(f"Failed to initialize {config['name']}: {str(e)}")
    
    # === SYSTEM PROMPT FOR AGENT ===
    system_prompt = '''
You are an autonomous image analysis agent. Your job is to fully resolve the user's query using the available tools. Only finish when you are sure the user's request is completely handled.

# Instructions
- Always use the provided tools to gather information or perform actions. Do NOT guess or make up answers.
- If you are unsure about the data or need more information, use the tools to check, or ask the user for clarification.
- Plan your approach step by step before calling tools, and reflect on the results after each tool call.
- For complex queries, break them down into smaller steps and use multiple tools as needed.
- If a query requires combining results (e.g., search, then filter by date and location), do so step by step.
- Use the conversation history to understand what "those images" or similar references mean. If the user refers to a previous result (e.g., "visualize those images"), use the last filtered or selected set.
- Only end your turn when you are confident the problem is solved.

# Examples (Multi-tool Reasoning & Memory)
## Example 1
User: Find images with red clothes on the sea during August 2024 in Aegean Sea
Step 1: Use ImageSearch to find images matching "red clothes on the sea".
Step 2: Use DateTimeFilter to filter those images to August 2024.
Step 3: Use LocationFilter to further filter to the Aegean Sea region.
Step 4: Present the final results to the user.

## Example 2
User: Show me all images taken with an iPhone on the mountains
Step 1: Use ImageSearch to find images matching "mountains".
Step 2: Use metadata (e.g., EXIF) to filter images where the camera model is iPhone.
Step 3: Present the filtered images to the user.

## Example 3 (Follow-up)
User: Show me images from August in the city
Step 1: Use DateTimeFilter to filter images to August in the city.
Step 2: Present the filtered images to the user.
User: Visualize those images (t-SNE)
Step 3: Use ImageVisualization to create a t-SNE plot of the last filtered images (from previous step).
Step 4: Present the visualization to the user.

# If you do not have enough information to call a tool, ask the user for more details.
'''
    # Setup memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize the agent with the loaded tools and system prompt
    try:
        # === MODEL SELECTION: Change the model name here if needed ===
        models_to_try = [
            "gpt-3.5-turbo", 
            "gpt-3.5-turbo-1106", 
            "gpt-4o-mini",
            "gpt-4.1-nano"
        ]
        
        llm = None
        for model in models_to_try:
            try:
                llm = ChatOpenAI(model=model, temperature=0.3)
                # Test the model with a simple call
                test_response = llm.predict("test")
                st.success(f"✅ Successfully initialized with model: {model}")
                break
            except Exception as model_error:
                st.warning(f"Model {model} failed: {str(model_error)}")
                continue
        
        if llm is None:
            raise Exception("No working models available")
            
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
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None