# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-05-29

### Added
- Initial release of LightlyGPT
- CLIP-based semantic image search functionality
- OpenAI GPT integration for conversational AI
- LangChain agent architecture for tool selection
- ChromaDB vector storage for efficient image embeddings
- Color analysis tools for dominant color extraction
- Duplicate image detection with configurable similarity thresholds
- Black & white image detection
- Image clustering and visualization (t-SNE)
- Metadata-based filtering (date, location)
- Streamlit web interface
- Comprehensive test suite
- Modern packaging with pyproject.toml

### Features
- **Image Search**: Natural language queries to find specific images
- **Color Analysis**: Extract and analyze dominant colors in image collections
- **Duplicate Detection**: Find similar or duplicate images with adjustable sensitivity
- **Visualization**: t-SNE plots and clustering for image collection exploration
- **Metadata Filtering**: Filter images by date, location, and other EXIF data
- **Agent Architecture**: Intelligent tool selection based on user queries
- **Chat Interface**: Conversational interaction with your image collection

### Technical Stack
- Python 3.8+ support
- Streamlit for web interface
- CLIP (Contrastive Language-Image Pre-training) for embeddings
- OpenAI GPT models for natural language processing
- LangChain for agent orchestration
- ChromaDB for vector storage
- PyTorch and Transformers for deep learning
- scikit-learn for machine learning utilities

### Security
- Environment variable support for API keys
- Input validation and error handling
- Secure handling of uploaded images
