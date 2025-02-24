# Document Information Extractor

A powerful document analysis tool that extracts information from various document types using Large Language Models (LLMs). This application supports multiple LLM providers and can process various document formats to generate structured information.

## Features

- **Multiple Document Format Support**:
  - PDF documents (*.pdf)
  - Excel files (*.xlsx, *.xls)
  - Word documents (*.docx)
  - Text files (*.txt, *.csv, *.json, *.xml, *.md)

- **LLM Provider Support**:
  - OpenAI (GPT-3.5, GPT-4)
  - Azure OpenAI
  - OpenRouter
  - Mistral AI
  - Together AI
  - Anyscale
  - Ollama (local models)
  - Deepseek
  - Any OpenAI API-compatible provider

- **Features**:
  - Dynamic model listing for each provider
  - Customizable system and user prompts
  - Batch processing support
  - Markdown output format
  - Beautiful Streamlit web interface

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Document-Information-Extractor.git
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

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Configure your providers in the `.env` file:

   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here

   # Azure OpenAI Configuration
   PROVIDER_AZURE_NAME=Azure OpenAI
   PROVIDER_AZURE_API_KEY=your_azure_api_key_here
   PROVIDER_AZURE_BASE_URL=https://your-resource.openai.azure.com
   PROVIDER_AZURE_MODEL=gpt-35-turbo
   PROVIDER_AZURE_API_VERSION=2024-02-15-preview

   # Add more providers as needed
   ```

### Adding a New Provider

To add a new OpenAI-compatible provider, add the following to your `.env` file:

```bash
PROVIDER_<NAME>_NAME=Display Name
PROVIDER_<NAME>_API_KEY=your_api_key_here
PROVIDER_<NAME>_BASE_URL=https://api.provider.com/v1
PROVIDER_<NAME>_MODEL=default-model-name
PROVIDER_<NAME>_API_VERSION=api-version  # Optional
```

Example for Mistral AI:
```bash
PROVIDER_MISTRAL_NAME=Mistral AI
PROVIDER_MISTRAL_API_KEY=your_mistral_api_key_here
PROVIDER_MISTRAL_BASE_URL=https://api.mistral.ai/v1
PROVIDER_MISTRAL_MODEL=mistral-medium
```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. In the sidebar:
   - Select your LLM provider
   - Configure the provider settings
   - Choose a model from the available options

4. Upload one or more supported documents

5. Click 'Process Files' to start the extraction

The application will process each file and generate structured information in Markdown format.

## Customizing Prompts

You can customize the system and user prompts for each provider in the Prompts directory:

- `Prompts/<template_name>/system.md`: Contains the system instructions
- `Prompts/<template_name>/user.md`: Contains the user prompt template

## Error Handling

The application includes robust error handling:
- Retries for API calls with exponential backoff
- Fallback mechanisms for model listing
- Informative error messages in the UI
- Comprehensive logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the ChatGPT API
- Streamlit for the web interface
- All supported LLM providers for their APIs
