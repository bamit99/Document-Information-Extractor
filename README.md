# PDF Q&A Extractor

A powerful tool that extracts questions and answers from PDF documents using various LLM providers (OpenAI, Ollama, Deepseek). Built with Streamlit for an intuitive user interface.

## Features

- 🔄 Multiple LLM Provider Support
  - OpenAI (GPT-3.5, GPT-4)
  - Ollama (Local models)
  - Deepseek
- 📝 Customizable Prompts
  - System prompt customization
  - User prompt template editing
- 📁 Batch Processing
  - Process multiple PDFs at once
  - Size limit of 10MB per file
- 💾 Structured Output
  - Markdown formatted Q&A pairs
  - Organized output directory structure

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd langflow_workflows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables (optional):
```bash
# Create a .env file
OPENAI_API_KEY=your_api_key
DEEPSEEK_API_KEY=your_api_key
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Configure your preferred LLM provider:
   - OpenAI: Enter your API key
   - Ollama: Ensure local instance is running (default: http://localhost:11434)
   - Deepseek: Enter API key and base URL

3. Customize prompts (optional):
   - Modify system prompt to change AI behavior
   - Edit user prompt template for custom formatting
   - Use {text} placeholder for PDF content in template

4. Upload PDFs and process:
   - Select one or more PDF files
   - Click "Process PDFs"
   - Find results in the output_qa directory

## Project Structure

```
langflow_workflows/
├── app.py              # Streamlit UI and main application
├── python.py           # Core logic and LLM providers
├── requirements.txt    # Project dependencies
├── temp_pdfs/         # Temporary storage for uploaded PDFs
└── output_qa/         # Generated Q&A markdown files
```

## LLM Provider Setup

### OpenAI
- Requires API key
- Supports GPT-3.5 and GPT-4 models
- Set key in UI or .env file

### Ollama
- Requires local installation
- Run Ollama server before using
- Supports various open-source models

### Deepseek
- Requires API key and base URL
- Supports multiple model variants
- Set credentials in UI or .env file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
