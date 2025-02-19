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
  - Structured text processing

- **Customizable Prompts**
  - Pre-defined prompt templates in `Prompts` directory
  - Custom prompt creation
  - System and User prompt separation
  - Default text placeholder support

- **Automatic File Management**
  - Automatic saving to `processed_files` directory
  - Timestamp-based unique filenames
  - Preview processed content in UI
  - Bulk download as ZIP
  - Markdown output format

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Question-Answer
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```plaintext
   OPENAI_API_KEY=your_openai_key
   DEEPSEEK_API_KEY=your_deepseek_key
   DEEPSEEK_BASE_URL=your_deepseek_url
   OLLAMA_BASE_URL=your_ollama_url
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Configure the application:
   - Select your preferred LLM provider
   - Configure provider settings (API keys, model selection)
   - Choose or create prompts for processing

3. Process documents:
   - Upload one or more supported files
   - Click "Process Files"
   - View results in the UI
   - Find processed files in the `processed_files` directory

## Prompt Templates

The application supports customizable prompt templates located in the `Prompts` directory:

```
Prompts/
├── analyze_paper/
│   ├── system.md
│   └── user.md
├── analyze_patent/
│   └── ...
└── ...
```

Each template contains:
- `system.md`: Instructions for the AI's behavior
- `user.md`: Template for processing document content

To create a new template:
1. Create a new directory in `Prompts/`
2. Add `system.md` and `user.md` files
3. Use `{text}` placeholder in `user.md` for document content

## Output Format

Processed files are saved as Markdown files with:
- Original filename
- Timestamp
- Extracted information
- Formatted content based on prompts

Example: `document_20250219_183000.md`

## Dependencies

- Python 3.8+
- See `requirements.txt` for full list

## Error Handling

The application includes:
- Input validation
- Error reporting
- Progress tracking
- Automatic file cleanup

## Notes

- Files are automatically saved in the `processed_files` directory
- Each processed file has a unique timestamp
- Preview processed content before downloading
- Download individual files or all as ZIP
- Empty user prompts default to `{text}` placeholder
