import os
import streamlit as st
from pathlib import Path
from python import PDFProcessor, OpenAIProvider, OllamaProvider, DeepseekProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="PDF Q&A Extractor",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 PDF Q&A Extractor")
st.markdown("""
This app extracts questions and answers from PDF documents using various LLM providers.
Upload a PDF file or multiple files and get structured Q&A pairs in markdown format.
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
    provider = st.selectbox(
        "Select LLM Provider",
        ["OpenAI", "Ollama", "Deepseek"],
        help="Choose the AI model provider"
    )
    
    # Provider-specific configuration
    if provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Your OpenAI API key"
        )
        llm_config = {"api_key": api_key}
        
        # Fetch and display available models
        models = get_provider_models(provider, llm_config)
        selected_model = st.selectbox(
            "Select Model",
            models,
            index=0 if models else -1,
            help="Choose the OpenAI model to use"
        )
        llm_config["model"] = selected_model
        
    elif provider == "Ollama":
        base_url = st.text_input(
            "Ollama Base URL",
            value="http://localhost:11434",
            help="Base URL for Ollama API"
        )
        llm_config = {"base_url": base_url}
        
        # Fetch and display available models
        if base_url:
            models = get_provider_models(provider, llm_config)
            selected_model = st.selectbox(
                "Select Model",
                models,
                index=0 if models else -1,
                help="Choose the Ollama model to use"
            )
            llm_config["model"] = selected_model
            
            st.info("Make sure Ollama is running and the selected model is pulled.")
        
    elif provider == "Deepseek":
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
        models = get_provider_models(provider, llm_config)
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
        value="Extract questions and answers from the given text.",
        help="The instruction given to the AI about its task"
    )
    user_prompt_template = st.text_area(
        "User Prompt Template",
        value="Extract all questions and answers from the following text:\n\n{text}\n\nFormat as: Question: ... Answer: ...",
        help="The template for processing text. Use {text} as placeholder for the PDF content"
    )
    
    # Add usage instructions
    st.markdown("### How to use")
    st.markdown("""
    1. Select your preferred LLM provider
    2. Configure the provider settings
    3. Choose a model from the available options
    4. Upload one or more PDF files
    5. Click 'Process PDFs'
    6. Download the generated Q&A
    """)

# Main content area
def main():
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Validate configuration
        if provider == "OpenAI" and not llm_config["api_key"]:
            st.warning("Please enter your OpenAI API key in the sidebar.")
            return
        elif provider == "Deepseek" and (not llm_config["api_key"] or not llm_config["base_url"]):
            st.warning("Please enter your Deepseek API key and base URL in the sidebar.")
            return
        elif provider == "Ollama" and not llm_config["base_url"]:
            st.warning("Please enter the Ollama base URL in the sidebar.")
            return
        
        if st.button("Process PDFs"):
            # Initialize appropriate provider
            try:
                if provider == "OpenAI":
                    llm_provider = OpenAIProvider(llm_config["api_key"], llm_config["model"])
                elif provider == "Ollama":
                    llm_provider = OllamaProvider(llm_config["base_url"], llm_config["model"])
                elif provider == "Deepseek":
                    llm_provider = DeepseekProvider(llm_config["api_key"], llm_config["base_url"], llm_config["model"])
                
                # Set custom prompts
                llm_provider.set_prompts(system_prompt, user_prompt_template)
                
                processor = PDFProcessor(llm_provider)
                
                # Create directories
                temp_dir = Path("temp_pdfs")
                output_dir = Path("output_qa")
                temp_dir.mkdir(exist_ok=True)
                output_dir.mkdir(exist_ok=True)
                
                # Process each uploaded file
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Save uploaded file temporarily
                        progress = (i) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        temp_path = temp_dir / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract text
                        text = processor.extract_text_from_pdf(str(temp_path))
                        if text:
                            # Extract Q&A
                            qa = processor.extract_questions_answers(text)
                            if qa:
                                # Save markdown
                                output_path = output_dir / f"{uploaded_file.name.rsplit('.', 1)[0]}.md"
                                if processor.format_to_markdown(qa, str(output_path)):
                                    st.success(f"Successfully processed {uploaded_file.name}")
                                    
                                    # Display the Q&A
                                    with st.expander(f"Q&A from {uploaded_file.name}", expanded=True):
                                        st.markdown(qa)
                                    
                                    # Add download button
                                    with open(output_path, "r", encoding="utf-8") as f:
                                        st.download_button(
                                            label=f"Download Q&A for {uploaded_file.name}",
                                            data=f.read(),
                                            file_name=output_path.name,
                                            mime="text/markdown"
                                        )
                                else:
                                    st.error(f"Failed to save results for {uploaded_file.name}")
                            else:
                                st.error(f"Failed to extract Q&A from {uploaded_file.name}")
                        else:
                            st.error(f"Failed to extract text from {uploaded_file.name}")
                        
                        # Clean up temporary file
                        temp_path.unlink()
                    
                    progress_bar.progress(1.0)
                    status_text.text("All files processed!")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                finally:
                    # Clean up temporary directory
                    if temp_dir.exists():
                        for file in temp_dir.iterdir():
                            file.unlink()
                        temp_dir.rmdir()
            
            except Exception as e:
                st.error(f"Failed to initialize LLM provider: {str(e)}")
    
    # Add documentation
    with st.expander("📖 Documentation"):
        st.markdown("""
        ### About this app
        
        This application uses various LLM providers to extract questions and answers from PDF documents.
        
        **Supported Providers:**
        
        1. **OpenAI**
           - Uses GPT models (3.5-turbo, GPT-4)
           - Requires API key
           - Best for accuracy and reliability
        
        2. **Ollama**
           - Uses local models (llama2, mistral, etc.)
           - Requires running Ollama server
           - Free and private processing
        
        3. **Deepseek**
           - Multiple specialized models
           - Requires API key
           - Good for specific use cases
        
        **Features:**
        - Multiple LLM provider support
        - Dynamic model selection
        - Upload multiple PDF files
        - Automatic text extraction
        - AI-powered Q&A generation
        - Markdown output format
        - Download results
        
        **Limitations:**
        - PDF file size limit: 10MB
        - Processing time depends on:
          - File size and content
          - Selected provider and model
          - Network conditions
        
        **Output Format:**
        The generated Q&A pairs are formatted in Markdown with the following structure:
        ```markdown
        # Extracted Questions and Answers
        
        ## Question: ...
        Answer: ...
        ```
        """)

if __name__ == "__main__":
    main()
