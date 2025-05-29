"""
LightlyGPT - AI-powered image analysis tool

A sophisticated image analysis application that combines CLIP embeddings,
OpenAI's GPT models, and LangChain agents for intelligent image exploration.

Features:
- Semantic image search using CLIP
- Conversational AI interface with GPT
- Color analysis and duplicate detection
- Image clustering and visualization
- Metadata-based filtering
"""

__version__ = "1.0.0"
__author__ = "Konstantinos Zarmakoupis"
__email__ = "zarmaks@gmail.com"
__license__ = "MIT"

from utils import setup_project_path

# Ensure project path is set up for imports
setup_project_path()

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
