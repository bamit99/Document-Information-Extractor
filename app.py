import os
import streamlit as st
from pathlib import Path
from python import PDFProcessor, OpenAIProvider, OllamaProvider, DeepseekProvider, DocumentProcessor
from dotenv import load_dotenv
import tempfile
import zipfile

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Document Information Extractor",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Document Information Extractor")
st.markdown("""
This app extracts information from various document types using LLM providers.
Supported formats:
- PDF documents
- Excel files (XLSX, XLS)
- Word documents (DOCX)
- Text files (TXT, CSV, JSON, XML, MD)
""")

def get_provider_models(provider_name: str, config: dict) -> list:
    """Get available models for the selected provider by querying their APIs"""
    try:
        if provider_name == "OpenAI":
            if not config.get("api_key"):
                st.sidebar.warning("OpenAI API key is required to fetch available models")
                return []
            provider = OpenAIProvider(config["api_key"])
            models = provider.list_models()
            if not models:
                st.sidebar.warning("No models found. Please check your API key.")
            return models
            
        elif provider_name == "Ollama":
            if not config.get("base_url"):
                st.sidebar.warning("Ollama base URL is required to fetch available models")
                return []
            provider = OllamaProvider(config["base_url"])
            models = provider.list_models()
            if not models:
                st.sidebar.warning("No models found. Please check if Ollama is running and has models pulled.")
            return models
            
        elif provider_name == "Deepseek":
            if not config.get("api_key") or not config.get("base_url"):
                st.sidebar.warning("Deepseek API key and base URL are required to fetch available models")
                return []
            provider = DeepseekProvider(config["api_key"], config["base_url"])
            models = provider.list_models()
            if not models:
                st.sidebar.warning("No models found. Please check your API credentials.")
            return models
            
    except Exception as e:
        st.sidebar.error(f"Error fetching models from {provider_name}: {str(e)}")
        st.sidebar.info("Please configure your provider settings correctly to see available models.")
        return []
    
    return []

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # LLM Provider Selection
    provider_name = st.selectbox(
        "Select LLM Provider",
        ["OpenAI", "Ollama", "Deepseek"],
        help="Choose the AI provider to process your documents"
    )
    
    # Provider-specific configuration
    if provider_name == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Your OpenAI API key"
        )
        llm_config = {"api_key": api_key}
        
        # Fetch and display available models
        models = get_provider_models(provider_name, llm_config)
        selected_model = st.selectbox(
            "Select Model",
            models,
            index=0 if models else -1,
            help="Choose the OpenAI model to use"
        )
        llm_config["model"] = selected_model
        
    elif provider_name == "Ollama":
        base_url = st.text_input(
            "Ollama Base URL",
            value="http://localhost:11434",
            help="Base URL for Ollama API"
        )
        llm_config = {"base_url": base_url}
        
        # Fetch and display available models
        if base_url:
            models = get_provider_models(provider_name, llm_config)
            selected_model = st.selectbox(
                "Select Model",
                models,
                index=0 if models else -1,
                help="Choose the Ollama model to use"
            )
            llm_config["model"] = selected_model
            
            st.info("Make sure Ollama is running and the selected model is pulled.")
        
    elif provider_name == "Deepseek":
        api_key = st.text_input(
            "Deepseek API Key",
            type="password",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            help="Your Deepseek API key"
        )
        base_url = st.text_input(
            "Deepseek Base URL",
            value="https://api.deepseek.com",
            help="Base URL for Deepseek API"
        )
        llm_config = {"api_key": api_key, "base_url": base_url}
        
        # Display available models
        models = get_provider_models(provider_name, llm_config)
        selected_model = st.selectbox(
            "Select Model",
            models,
            index=0 if models else -1,
            help="Choose the Deepseek model to use"
        )
        llm_config["model"] = selected_model
    
    st.info("Your configuration is securely stored for this session only.")
    
    # Prompt customization
    st.header("Prompt Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value="Extract information from the given text.",
        help="The instruction given to the AI about its task"
    )
    user_prompt_template = st.text_area(
        "User Prompt Template",
        value="Extract information from the following text:\n\n{text}\n\nFormat as: ...",
        help="The template for processing text. Use {text} as placeholder for the document content"
    )
    
    # Add usage instructions
    st.markdown("### How to use")
    st.markdown("""
    1. Select your preferred LLM provider
    2. Configure the provider settings
    3. Choose a model from the available options
    4. Upload one or more files
    5. Click 'Process Files'
    6. Download the generated information
    """)

# Main content area
def main():
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload document files",
        type=["pdf", "xlsx", "xls", "docx", "txt", "csv", "json", "xml", "md"],
        accept_multiple_files=True,
        help="Select one or more files to process"
    )
    
    if uploaded_files:
        # Validate configuration
        if provider_name == "OpenAI" and not llm_config["api_key"]:
            st.warning("Please enter your OpenAI API key in the sidebar.")
            return
        elif provider_name == "Deepseek" and (not llm_config["api_key"] or not llm_config["base_url"]):
            st.warning("Please enter your Deepseek API key and base URL in the sidebar.")
            return
        elif provider_name == "Ollama" and not llm_config["base_url"]:
            st.warning("Please enter the Ollama base URL in the sidebar.")
            return
        
        if st.button("Process Files", type="primary"):
            # Initialize appropriate provider
            try:
                if provider_name == "OpenAI":
                    llm_provider = OpenAIProvider(llm_config["api_key"], llm_config["model"])
                elif provider_name == "Ollama":
                    llm_provider = OllamaProvider(llm_config["base_url"], llm_config["model"])
                elif provider_name == "Deepseek":
                    llm_provider = DeepseekProvider(llm_config["api_key"], llm_config["base_url"], llm_config["model"])
                
                # Set custom prompts
                llm_provider.set_prompts(system_prompt, user_prompt_template)
                
                processor = DocumentProcessor(llm_provider)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    output_dir = temp_dir / "output"
                    output_dir.mkdir()
                    
                    # Initialize progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each file
                    for i, uploaded_file in enumerate(uploaded_files):
                        progress = (i) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        temp_path = temp_dir / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract text
                        text = processor.process_file(str(temp_path))
                        if text:
                            # Process with LLM
                            result = processor.extract_information(text)
                            if result:
                                # Save to markdown
                                output_path = output_dir / f"{temp_path.stem}.md"
                                if processor.format_to_markdown(result, str(output_path)):
                                    st.success(f"Successfully processed {uploaded_file.name}")
                                else:
                                    st.error(f"Failed to save results for {uploaded_file.name}")
                            else:
                                st.error(f"Failed to process {uploaded_file.name}")
                        else:
                            st.error(f"Failed to extract text from {uploaded_file.name}")
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Create zip file of results
                    zip_path = temp_dir / "results.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for markdown_file in output_dir.glob('*.md'):
                            zipf.write(markdown_file, markdown_file.name)
                    
                    # Offer download
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            "Download Results",
                            f,
                            file_name="extracted_information.zip",
                            mime="application/zip",
                            help="Download all processed files as a ZIP archive"
                        )
            
            except Exception as e:
                st.error(f"Error initializing provider: {str(e)}")
    
    # Add documentation
    with st.expander("📖 Documentation"):
        st.markdown("""
        ### About this app
        
        This application uses various LLM providers to extract information from document files.
        
        **Supported Providers:**
        - OpenAI (GPT-3.5, GPT-4)
        - Ollama (Local models)
        - Deepseek
        
        **Supported File Types:**
        - PDF documents (.pdf)
        - Excel files (.xlsx, .xls)
        - Word documents (.docx)
        - Text files (.txt, .csv, .json, .xml, .md)
        
        **Features:**
        - Multiple LLM provider support
        - Dynamic model selection
        - Upload multiple files
        - Automatic text extraction
        - AI-powered information extraction
        - Markdown output format
        
        **Limitations:**
        - File size limit: 10MB
        - Processing time depends on:
          - File size and content
          - Selected provider and model
          - Network speed
        
        ### Output Format
        The generated information will be in markdown format:
        
        ```markdown
        # Extracted Information
        
        ...
        ```
        """)

if __name__ == "__main__":
    main()
