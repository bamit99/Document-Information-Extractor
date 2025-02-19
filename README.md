# Document Information Extractor

A Streamlit-based application that extracts and processes information from various document types using Large Language Models (LLMs).

## Features

- **Multi-Format Support**
  - PDF documents (.pdf)
  - Excel files (.xlsx, .xls)
  - Word documents (.docx)
  - Text files (.txt, .csv, .json, .xml, .md)

- **Multiple LLM Providers**
  - OpenAI (GPT-3.5, GPT-4)
  - Ollama (Local models)
  - Deepseek

- **Rich Document Processing**
  - Automatic file type detection
  - Comprehensive Excel processing (all sheets, formulas, metadata)
  - PDF text extraction
  - Word document parsing
  - Batch file processing

- **Customizable Prompts**
  - Configurable system prompts
  - Custom user prompt templates
  - Dynamic prompt variables

- **User-Friendly Interface**
  - Progress tracking
  - Error handling
  - Batch download of results
  - Provider configuration
  - Model selection

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd document-extractor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_key
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=your_deepseek_url
OLLAMA_BASE_URL=http://localhost:11434
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Configure the application:
   - Select an LLM provider
   - Enter necessary API keys
   - Choose a model
   - Customize prompts if needed

3. Upload files:
   - Select one or more supported files
   - Click "Process Files"
   - Download the results

## File Processing Details

### Excel Files
- Processes all sheets, including renamed ones
- Extracts document metadata
- Preserves table structure
- Handles merged cells
- Provides statistical summaries for numeric data

### PDF Files
- Extracts text from all pages
- Maintains document structure
- Handles various PDF formats

### Word Documents
- Processes text content
- Extracts tables and lists
- Maintains document structure

### Text Files
- Supports various encodings
- Preserves formatting
- Handles structured formats (JSON, XML)

## Output Format

Results are provided in markdown format, making them easy to read and process further. The output structure is determined by your prompt and the LLM's response.

## Dependencies

- `streamlit`: Web interface
- `pandas`: Data processing
- `openpyxl`: Excel file handling
- `python-docx`: Word document processing
- `PyMuPDF`: PDF processing
- `python-dotenv`: Environment management
- `requests`: API communication
- `openai`: OpenAI API integration

## Error Handling

- File size limits (10MB per file)
- Invalid file formats
- API connection issues
- Processing errors
- Missing configurations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here]
