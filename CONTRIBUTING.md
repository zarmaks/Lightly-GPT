# Contributing to LightlyGPT

Thank you for your interest in contributing to LightlyGPT! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- OpenAI API key for testing

### Setting up the development environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/LightlyGPT.git
   cd LightlyGPT
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

5. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all formatting tools:
```bash
# Format code
black .
isort .

# Check linting
flake8 .

# Type checking
mypy .
```

## Testing

We use pytest for testing. Make sure all tests pass before submitting a PR:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=utils --cov=tools

# Run specific test files
pytest tests/test_clip_tools.py
```

### Writing Tests
- Write tests for all new functionality
- Aim for at least 80% code coverage
- Use descriptive test names
- Mock external dependencies (OpenAI API, file operations)

## Submitting Changes

### Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in the imperative mood
- Keep the first line under 50 characters
- Add more details in the body if needed

Examples:
```
Add duplicate image detection tool
Fix CLIP model loading issue
Update documentation for installation
```

## Areas for Contribution

### High Priority
- Performance optimizations for large image collections
- Additional image analysis tools
- Improved error handling and user feedback
- Documentation improvements

### New Features
- Support for additional image formats
- Video analysis capabilities
- Batch processing improvements
- Advanced visualization options

### Bug Reports
- Check existing issues before creating new ones
- Provide detailed reproduction steps
- Include system information and error messages
- Add screenshots for UI issues

## Code Review Guidelines

- Be respectful and constructive
- Focus on the code, not the person
- Suggest specific improvements
- Test the changes locally when possible

## Questions or Need Help?

- Open an issue for bug reports or feature requests
- Start a discussion for questions about contributing
- Check existing issues and discussions before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
