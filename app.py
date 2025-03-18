import os
import streamlit as st
from pathlib import Path
from python import PDFProcessor, OpenAIProvider, OllamaProvider, DeepseekProvider, DocumentProcessor, OpenAICompatibleProvider, FileProcessor
from dotenv import load_dotenv
import tempfile
import zipfile
from datetime import datetime, timedelta
import io
import tiktoken  # Add tiktoken for token counting
import time
import logging
import base64

# Configure logging to show in Streamlit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Document Information Extractor",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for errors
if 'errors' not in st.session_state:
    st.session_state.errors = []
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = []

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate count if model-specific tokenizer not available
        return len(text.split()) * 1.3  # Rough approximation

def check_token_limit(text: str, model: str, provider: str) -> tuple[bool, int]:
    """Check if text exceeds token limit for given model"""
    token_count = count_tokens(text, model)
    
    # Define token limits for different providers/models
    token_limits = {
        "OpenAI": {
            "gpt-3.5-turbo": 16385,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "default": 16385
        },
        "Deepseek": {
            "default": 65536
        },
        "default": 16385
    }
    
    # Get token limit for the model
    if provider in token_limits:
        limit = token_limits[provider].get(model, token_limits[provider]["default"])
    else:
        limit = token_limits["default"]
    
    return token_count <= limit, token_count

def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format"""
    duration = timedelta(seconds=seconds)
    
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60
    milliseconds = duration.microseconds // 1000
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    elif seconds > 0:
        return f"{seconds}s {milliseconds}ms"
    else:
        return f"{milliseconds}ms"

def estimate_remaining_time(processed_count: int, total_count: int, elapsed_time: float) -> float:
    """Estimate remaining time based on current progress"""
    if processed_count == 0:
        return 0
    avg_time_per_file = elapsed_time / processed_count
    remaining_files = total_count - processed_count
    return avg_time_per_file * remaining_files

class ProcessingTimer:
    """Helper class to track processing phases"""
    def __init__(self):
        self.start_time = time.time()
        self.phases = {}
        self.current_phase = None
        self._phase_start = None
        
    def start_phase(self, phase_name: str):
        """Start timing a new phase"""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        if phase_name not in self.phases:
            self.phases[phase_name] = 0
        self._phase_start = time.time()
    
    def end_phase(self, phase_name: str = None):
        """End timing the current phase or specified phase"""
        if phase_name:
            if phase_name in self.phases and self._phase_start:
                self.phases[phase_name] += time.time() - self._phase_start
                if self.current_phase == phase_name:
                    self.current_phase = None
        elif self.current_phase and self._phase_start:
            self.phases[self.current_phase] += time.time() - self._phase_start
            self.current_phase = None
    
    def get_total_time(self) -> float:
        """Get total elapsed time"""
        return time.time() - self.start_time
    
    def get_phase_times(self) -> dict:
        """Get times for all phases"""
        return {phase: duration for phase, duration in self.phases.items()}

def save_processed_file(content, original_name, template_name):
    """Save processed content to a file and return the file path"""
    # Create a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(original_name).stem.replace(" ", "_")
    filename = f"{safe_name}_{timestamp}.md"
    
    # Save to the output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / filename
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return {
        "path": str(file_path),
        "name": filename,
        "original_name": original_name,
        "content": content,
        "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_template": template_name
    }

def process_and_download(file, llm_provider, output_dir):
    """Process a single file and return its download button"""
    try:
        # Extract text from file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        processor = FileProcessor.get_processor(tmp_path)
        text = processor.extract_text(tmp_path)
        os.unlink(tmp_path)
        
        if not text:
            return None, f"Failed to extract text from {file.name}"
            
        # Process with LLM
        system_prompt = st.session_state.system_prompt
        user_prompt = st.session_state.user_prompt.format(text=text)
        llm_provider.set_prompts(system_prompt, user_prompt)
        result = llm_provider.generate_qa(text)
        
        if not result:
            return None, f"Failed to process {file.name}"
        
        # Save result and create download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{Path(file.name).stem}_{timestamp}.md"
        output_path = output_dir / output_filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
            
        return output_path, None
    except Exception as e:
        return None, str(e)

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
            
        elif provider_name == "OpenAI Compatible":
            if not config.get("api_key") or not config.get("base_url"):
                st.sidebar.warning("API key and base URL are required to fetch available models")
                return []
            provider = OpenAICompatibleProvider(config["api_key"], config["base_url"], config["model"], config.get("api_version"), config["provider_name"])
            models = provider.list_models()
            if not models:
                st.sidebar.warning("No models found. Please check your API credentials.")
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
            # Don't call OpenAI's API for Deepseek
            models = ["deepseek-chat", "deepseek-coder"]  # Default models
            if not models:
                st.sidebar.warning("No models found. Please check your API credentials.")
            return models
            
    except Exception as e:
        st.sidebar.error(f"Error fetching models from {provider_name}: {str(e)}")
        st.sidebar.info("Please configure your provider settings correctly to see available models.")
        return []
    
    return []

def load_compatible_providers() -> dict:
    """Load OpenAI-compatible providers from environment variables"""
    providers = {}
    
    # Find all provider configurations in environment variables
    for key in os.environ:
        if key.startswith('PROVIDER_') and key.endswith('_NAME'):
            provider_prefix = key[:-5]  # Remove '_NAME'
            provider_name = os.getenv(key)
            
            # Get provider configuration
            config = {
                "name": provider_name,
                "api_key": os.getenv(f"{provider_prefix}_API_KEY"),
                "base_url": os.getenv(f"{provider_prefix}_BASE_URL"),
                "model": os.getenv(f"{provider_prefix}_MODEL"),
                "api_version": os.getenv(f"{provider_prefix}_API_VERSION"),
                "provider_id": provider_prefix.replace('PROVIDER_', '').lower()
            }
            
            # Only add provider if required fields are present
            if config["api_key"] and config["base_url"]:
                provider_id = provider_prefix.replace('PROVIDER_', '').lower()
                providers[provider_id] = config
    
    return providers

def load_prompt_template(template_name: str) -> tuple[str, str]:
    """Load system and user prompts from template directory"""
    base_path = os.path.join("Prompts", template_name)
    system_path = os.path.join(base_path, "system.md")
    user_path = os.path.join(base_path, "user.md")
    
    system_prompt = ""
    user_prompt = "{text}"
    
    try:
        if os.path.exists(system_path):
            with open(system_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
    except Exception as e:
        st.error(f"Error loading system prompt: {str(e)}")
    
    try:
        if os.path.exists(user_path):
            with open(user_path, 'r', encoding='utf-8') as f:
                user_prompt = f.read().strip()
                if not user_prompt:  # If user.md is empty, use a default template
                    user_prompt = "Process the following text:\n\n{text}"
    except Exception as e:
        st.error(f"Error loading user prompt: {str(e)}")
    
    return system_prompt, user_prompt

def get_prompt_templates() -> list[str]:
    """Get list of available prompt templates"""
    prompts_dir = "Prompts"
    if not os.path.exists(prompts_dir):
        return []
    return [d for d in os.listdir(prompts_dir) 
            if os.path.isdir(os.path.join(prompts_dir, d))]

# Initialize session state variables
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = "Extract information from the given text."
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = "Extract information from the following text:\n\n{text}\n\nFormat as: ..."
if 'provider_name' not in st.session_state:
    st.session_state.provider_name = "OpenAI"
if 'llm_config' not in st.session_state:
    st.session_state.llm_config = {}
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = "processed_files"
if 'compatible_providers' not in st.session_state:
    st.session_state.compatible_providers = load_compatible_providers()
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = {}

# Create output directory if it doesn't exist
output_base_dir = Path("output")
output_base_dir.mkdir(exist_ok=True)

# Display any stored errors at the top
if st.session_state.errors:
    with st.container():
        st.error("⚠️ Recent Errors:")
        for error in st.session_state.errors[-5:]:  # Show last 5 errors
            st.warning(error)
        if st.button("Clear Errors"):
            st.session_state.errors = []

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # LLM Provider Configuration in a collapsible section
    with st.expander("LLM Provider Settings", expanded=True):
        # LLM Provider Selection
        available_providers = ["OpenAI", "Ollama", "Deepseek", "OpenAI Compatible"]
        provider_name = st.selectbox("Select LLM Provider", available_providers)

        # Provider-specific configuration
        if provider_name == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            llm_config = {"api_key": api_key}
        elif provider_name == "Ollama":
            base_url = st.text_input("Ollama Base URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
            llm_config = {"base_url": base_url}
        elif provider_name == "Deepseek":
            api_key = st.text_input("Deepseek API Key", type="password", value=os.getenv("DEEPSEEK_API_KEY", ""))
            base_url = st.text_input("Deepseek Base URL", value=os.getenv("DEEPSEEK_BASE_URL", ""))
            llm_config = {"api_key": api_key, "base_url": base_url}
        elif provider_name == "OpenAI Compatible":
            api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""))
            base_url = st.text_input("Base URL", value=os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""))
            api_version = st.text_input("API Version (optional)", value=os.getenv("OPENAI_COMPATIBLE_API_VERSION", ""))
            llm_config = {
                "api_key": api_key,
                "base_url": base_url,
                "api_version": api_version,
                "provider_name": provider_name,
                "model": ""  # Will be set after model selection
            }
        
        # Fetch and display available models
        models = get_provider_models(provider_name, llm_config)
        
        # Add search box for models
        model_search = st.text_input("Search Models", key="model_search", placeholder="Type to filter models...")
        
        # Filter models based on search
        if model_search and models:
            filtered_models = [model for model in models if model_search.lower() in model.lower()]
        else:
            filtered_models = models

        selected_model = st.selectbox(
            "Select Model",
            filtered_models,
            index=0 if filtered_models else -1,
            help="Choose the model to use"
        )
        llm_config["model"] = selected_model

    st.divider()
    
    # Prompt Settings in a collapsible section
    with st.expander("Prompt Settings", expanded=True):
        # Add search box for prompts
        prompt_search = st.text_input("Search Prompts", key="prompt_search", placeholder="Search prompt templates...")
        
        # Get and filter prompt templates
        templates = get_prompt_templates()
        if prompt_search and templates:
            filtered_templates = [t for t in templates if prompt_search.lower() in t.lower()]
        else:
            filtered_templates = templates

        if filtered_templates:
            selected_template = st.selectbox(
                "Select Prompt Template",
                filtered_templates,
                key="prompt_template"
            )
            
            if selected_template:
                template_content = load_prompt_template(selected_template)
                if template_content:
                    system_prompt, user_prompt = template_content
                    st.session_state.system_prompt = system_prompt
                    st.session_state.user_prompt = user_prompt
                    st.success(f"Loaded template: {selected_template}")
        else:
            st.warning("No prompt templates found in the Prompts directory")
            # Fallback to custom prompts
            system_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.system_prompt,
                height=100
            )
            user_prompt = st.text_area(
                "User Prompt",
                value=st.session_state.user_prompt,
                height=100
            )
            st.session_state.system_prompt = system_prompt
            st.session_state.user_prompt = user_prompt

    st.info("Your configuration is securely stored for this session only.")
    
    # Add usage instructions at the bottom
    with st.expander("How to Use", expanded=False):
        st.markdown("""
        1. Select an LLM provider and configure its settings
        2. Choose or search for a model
        3. Select a prompt template or create custom prompts
        4. Upload your documents for processing
        """)

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # File upload and processing section
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "xlsx", "xls", "docx", "txt", "csv", "json", "xml", "md"],
        accept_multiple_files=True,
        help="Select one or more files to process"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary", use_container_width=True):
            # Initialize LLM provider
            llm_provider = None
            if provider_name == "OpenAI":
                llm_provider = OpenAIProvider(llm_config["api_key"])
            elif provider_name == "Ollama":
                llm_provider = OllamaProvider(llm_config["base_url"])
            elif provider_name == "Deepseek":
                llm_provider = DeepseekProvider(llm_config["api_key"], llm_config["base_url"])
            elif provider_name == "OpenAI Compatible":
                llm_provider = OpenAICompatibleProvider(
                    llm_config["api_key"],
                    llm_config["base_url"],
                    llm_config["model"],
                    llm_config.get("api_version"),
                    llm_config["provider_name"]
                )
            
            # Process files with progress tracking
            progress = st.progress(0)
            status = st.empty()
            processed_files = []
            
            for i, file in enumerate(uploaded_files):
                status.text(f"Processing {file.name}...")
                progress.progress((i + 1) / len(uploaded_files))
                
                output_path, error = process_and_download(file, llm_provider, output_dir)
                if output_path:
                    processed_files.append((file.name, output_path))
                    # Auto-trigger download
                    with open(output_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            f"Download {file.name} Results",
                            f.read(),
                            file_name=output_path.name,
                            mime="text/markdown",
                            key=f"download_{i}"
                        )
                if error:
                    st.error(f"❌ {error}")
            
            # Clear progress indicators
            progress.empty()
            status.empty()
            
            # Show completion message
            if processed_files:
                st.success("🎉 Processing complete! Downloads should start automatically.")
                
                # Create ZIP for batch download if multiple files
                if len(processed_files) > 1:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for _, path in processed_files:
                            zf.write(path, path.name)
                    
                    st.download_button(
                        "📦 Download All Results (ZIP)",
                        zip_buffer.getvalue(),
                        file_name="processed_documents.zip",
                        mime="application/zip",
                        key="download_all"
                    )
    else:
        st.info("Upload documents to begin processing.")

if __name__ == "__main__":
    main()
