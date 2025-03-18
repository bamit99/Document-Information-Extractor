![image](https://github.com/user-attachments/assets/06c27d13-16ea-4c43-9d12-86f182a199bd)
# Document Inforamtion Extractor

A sophisticated Python-based application that leverages Large Language Models (LLMs) to automatically process large documents based on various system prompts included in the program. This tool is designed to extract meaningful information and create structured data based on the system promppt selected. The solution is simple. Configure your LLM provider, upload your documents, and select the prompt you need. The application will process the documents and provide you with the results based on the selected System Prompt. Inspired by daniel miessler's h___s://github.com/danielmiessler/fabric

## Features

- **Multiple Document Format Support**:
  - PDF documents (*.pdf)
  - Excel files (*.xlsx, *.xls)
  - Word documents (*.docx)
  - Text files (*.txt, *.csv, *.json, *.xml, *.md)

- **LLM Provider Integration**:
  - OpenAI (GPT-3.5, GPT-4)
  - Ollama (local models)
  - Deepseek
  - Support for OpenAI API-compatible services
    - Azure OpenAI
    - Mistral AI
    - Together AI
    - Anyscale
    - OpenRouter

- **Key Features**:
  - Smart text extraction from multiple file formats
  - Customizable prompts for different question types
  - Batch processing capabilities
  - Robust error handling with retries
  - Web-based interface using Streamlit
  - Markdown output formatting

## System Requirements

- Python 3.8 or higher
- Windows/Linux/MacOS
- Internet connection for cloud-based LLM providers
- Sufficient disk space for document processing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bamit99/Document-Information-Extractor.git
   cd Document-Information-Extractor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root and configure your LLM providers:

   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=your_api_key_here

   # For other providers, use the format:
   PROVIDER_<NAME>_API_KEY=your_api_key_here
   PROVIDER_<NAME>_BASE_URL=provider_api_url
   PROVIDER_<NAME>_MODEL=model_name
   ```

2. Customize prompts in the `Prompts` directory:
   - Each prompt template has its own directory
   - `system.md`: Contains system instructions
   - `user.md`: Contains the user prompt template

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Through the web interface:
   - Select your preferred LLM provider
   - Configure provider settings
   - Upload documents for processing
   - Choose prompt templates
   - Process files and view results

3. Using the Python API:
   ```python
   from python import DocumentProcessor, OpenAIProvider

   # Initialize provider
   provider = OpenAIProvider(api_key="your_api_key")
   
   # Create processor
   processor = DocumentProcessor(provider)
   
   # Process files
   processor.process_files("input_path", "output_folder")
   ```

## Project Structure

- `app.py`: Streamlit web application
- `python.py`: Core processing logic and provider implementations
- `Prompts/`: Directory containing prompt templates
- `requirements.txt`: Python dependencies

## Error Handling

The application implements robust error handling:
- Automatic retries for API calls with exponential backoff
- Comprehensive logging
- User-friendly error messages
- Fallback mechanisms for provider connections

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI and other LLM providers for their APIs
- Streamlit for the web interface framework
- PyMuPDF, python-docx, and other document processing libraries
