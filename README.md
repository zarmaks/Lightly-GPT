# LightlyGPT

An AI-powered image analysis tool that combines CLIP, GPT-4, and LangChain to help you explore and analyze your image collections through natural conversation.

*Developed in collaboration with [Lightly.ai](https://lightly.ai) as a capstone project for the Data Science Bootcamp at Big Blue Data Academy.*

## Features

- **Semantic Image Search**: Find images using natural language descriptions
- **Intelligent Analysis**: Analyze colors, detect duplicates, extract metadata
- **Interactive Chat**: Conversational interface powered by GPT-4
- **Visual Clustering**: Group and visualize similar images
- **Efficient Storage**: ChromaDB for fast embedding retrieval

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone https://github.com/your-username/LightlyGPT.git
   cd LightlyGPT
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -e .
   ```

2. **Set your OpenAI API key:**
   ```bash
   set OPENAI_API_KEY=sk-your-key-here
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Example Queries

- "Find images with people smiling"
- "What are the dominant colors in my collection?"
- "Show me a visualization of my image collection"
- "Find duplicate images with threshold 5"

## Development

To contribute to the project:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .
ruff check . --fix
```

## Acknowledgments

This project was developed in collaboration with **Lightly.ai** as a capstone project for the Data Science Bootcamp at Big Blue Data Academy, with mentorship from Lionel Peer.

