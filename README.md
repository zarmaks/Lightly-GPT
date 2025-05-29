# LightlyGPT

LightlyGPT is an AI-powered image analysis tool that combines advanced vision and language models to help you explore, search, and analyze your image collections interactively.

*This project was developed in collaboration with [Lightly.ai](https://lightly.ai) as a capstone project for the Data Science Bootcamp at Big Blue Data Academy.*

## Features

- **CLIP Model**: Understands image content for semantic search and analysis
- **gpt-4.1-nano**: Provides intelligent, conversational reasoning about your images
- **LangChain**: Manages complex workflows and tool selection
- **ChromaDB**: Efficient storage and retrieval of image embeddings
- **Agent Architecture**: Dynamically chooses the right tool for each task

### Available Capabilities
- Find images matching natural language descriptions
- Analyze dominant colors in your images
- Detect black & white images automatically
- Find duplicate or similar images
- Filter by date & location using image metadata
- Visualize relationships between images (t-SNE)
- Group similar images into clusters

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Quick Install

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LightlyGPT.git
   cd LightlyGPT
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install the package:**
   ```bash
   # Install in development mode
   pip install -e .
   
   # Or install from PyPI (when published)
   pip install lightlygpt
   ```

### Development Installation

For contributing to the project:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install individual dev tools
pip install -e ".[test]"  # Just testing tools
```

## Usage

1. Set your OpenAI API key as an environment variable or enter it in the app when prompted.
   - To set it in your environment:
     ```sh
     set OPENAI_API_KEY=sk-...
     ```
2. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
3. Upload your images and start chatting with your collection!

## Example Queries
- "Find images with people smiling"
- "What are the dominant colors in my collection?"
- "Are there any black and white photos?"
- "Find duplicate images with threshold 5"
- "Show me a visualization of my image collection"
- "Group similar images into 3 clusters"

## About

This project was developed as a capstone project for the Data Science Bootcamp at Big Blue Data Academy over a 7-week period, in collaboration with [Lightly.ai](https://lightly.ai). The development process included weekly mentorship sessions with Lionel Peer, who provided valuable guidance and advice throughout the project.

## Acknowledgments

- **Lightly.ai** - For their collaboration and support in making this project possible
- **Big Blue Data Academy** - For providing the Data Science Bootcamp framework
- **Lionel Peer** - For mentorship and weekly guidance sessions during development

