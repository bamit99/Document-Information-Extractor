# Development Guide

## Commands
- Run app: `streamlit run app.py`
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest` (when tests are implemented)
- Format code: `black .` (Black formatter is recommended)
- Check types: `mypy app.py python.py`
- Linting: `flake8 .`

## Code Style Guidelines
- **Indentation**: 4 spaces
- **Line length**: 100-120 characters max
- **Imports**: Standard library first, third-party second, local imports last
- **Types**: Use type hints consistently (parameters and return values)
- **Naming**:
  - Classes: PascalCase
  - Functions/Methods/Variables: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Error handling**: Use specific exception types with descriptive messages
- **Docstrings**: Include for all classes and functions
- **Logging**: Use Python's built-in logging module with appropriate levels

## Class Structure
- Use abstract base classes where appropriate
- Follow proper OOP principles with clear inheritance hierarchies
- Maintain encapsulation of implementation details

This project is a Streamlit web application for document processing using various LLM providers (OpenAI, Ollama, Deepseek, and custom providers). The app extracts information from different document formats including PDF, Excel, Word, and text files.