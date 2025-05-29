# CLIP-based image analysis tools package
# These tools provide various image analysis functions for use with LangChain agents

from utils import setup_project_path
from utils.path_utils import check_dependency

# Ensure the project root is in the Python path
setup_project_path()

# Define available tools categories
TOOL_CATEGORIES = {
    "search": ["ImageSearch"],
    "analysis": ["ColorAnalysis", "BWDetection"],
    "organization": ["DuplicateDetection"],
    "metadata": ["DateTimeFilter", "LocationFilter"],
    "visualization": ["ImageVisualization", "ImageClustering"],
}

# Check dependencies for each tool category
CATEGORY_DEPENDENCIES = {
    "search": ["torch", "transformers"],
    "analysis": ["sklearn", "numpy"],
    "organization": ["numpy"],
    "metadata": ["PIL"],
    "visualization": ["matplotlib", "sklearn"],
}


def get_available_tools():
    """
    Returns a dictionary of available tools based on installed dependencies
    """
    available_categories = {}

    for category, deps in CATEGORY_DEPENDENCIES.items():
        # Check if all dependencies for this category are available
        all_available = all(check_dependency(dep) for dep in deps)
        available_categories[category] = all_available

    return available_categories
