import os
import streamlit as st
from pathlib import Path
from python import PDFProcessor, OpenAIProvider, OllamaProvider, DeepseekProvider, DocumentProcessor, OpenAICompatibleProvider
from dotenv import load_dotenv
import tempfile
import zipfile
from datetime import datetime, timedelta
import io
import tiktoken  # Add tiktoken for token counting
import time

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
        
    def start_phase(self, phase_name: str):
        """Start timing a new phase"""
        self.current_phase = phase_name
        if phase_name not in self.phases:
            self.phases[phase_name] = 0
        self._phase_start = time.time()
    
    def end_phase(self):
        """End timing the current phase"""
        if self.current_phase:
            self.phases[self.current_phase] += time.time() - self._phase_start
            self.current_phase = None
    
    def get_total_time(self) -> float:
        """Get total elapsed time"""
        return time.time() - self.start_time
    
    def get_phase_times(self) -> dict:
        """Get times for all phases"""
        return {phase: duration for phase, duration in self.phases.items()}

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
            models = provider.list_models()
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

# Main content area
def main():
    # Create output directory if it doesn't exist
    output_base_dir = Path("output")
    output_base_dir.mkdir(exist_ok=True)
    
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
            st.error("⚠️ Configuration Error: Please enter your OpenAI API key in the sidebar.")
            return
        elif provider_name == "Deepseek" and (not llm_config["api_key"] or not llm_config["base_url"]):
            st.error("⚠️ Configuration Error: Please enter your Deepseek API key and base URL in the sidebar.")
            return
        elif provider_name == "Ollama" and not llm_config["base_url"]:
            st.error("⚠️ Configuration Error: Please enter the Ollama base URL in the sidebar.")
            return
        
        if st.button("Process Files", type="primary"):
            # Validate prompts
            if not st.session_state.system_prompt:
                st.error("⚠️ Configuration Error: Please configure System Prompt before processing files.")
                return

            if not st.session_state.user_prompt:
                st.session_state.user_prompt = "{text}"
                st.info("ℹ️ User Prompt was empty. Defaulting to '{text}'.")
            elif "{text}" not in st.session_state.user_prompt:
                st.error("⚠️ Configuration Error: User Prompt must contain {text} placeholder for document content.")
                return

            # Create batch directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get prompt template name or use 'custom' if not using a template
            prompt_name = st.session_state.get('selected_prompt_template', 'Custom')
            if prompt_name == 'Custom' and 'last_template' in st.session_state:
                prompt_name = st.session_state.last_template
            prompt_name = prompt_name.lower().replace(' ', '_')
            
            # Create directory structure: output/prompt_name/batch_timestamp/
            batch_dir = output_base_dir / prompt_name / timestamp
            batch_dir.mkdir(parents=True, exist_ok=True)

            # Initialize appropriate provider
            try:
                if provider_name == "OpenAI":
                    llm_provider = OpenAIProvider(llm_config["api_key"], llm_config["model"])
                elif provider_name == "Ollama":
                    llm_provider = OllamaProvider(llm_config["base_url"], llm_config["model"])
                elif provider_name == "Deepseek":
                    llm_provider = DeepseekProvider(llm_config["api_key"], llm_config["base_url"], llm_config["model"])
                elif provider_name == "OpenAI Compatible":
                    llm_provider = OpenAICompatibleProvider(
                        llm_config["api_key"],
                        llm_config["base_url"],
                        llm_config["model"],
                        llm_config.get("api_version"),
                        "openai_compatible"
                    )
                
                # Set custom prompts immediately after initialization
                llm_provider.set_prompts(st.session_state.system_prompt, st.session_state.user_prompt)
                
                processor = DocumentProcessor(llm_provider)
                
                # Store processed files for later use
                processed_files = []
                failed_files = []
                
                # Initialize progress and timing metrics
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create a container for all processing details
                processing_details = st.container()
                
                # Create columns for metrics in the processing details
                with processing_details:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        current_file_time = st.empty()
                    with col2:
                        total_time = st.empty()
                    with col3:
                        remaining_time = st.empty()
                
                # Results container will be shown at the top
                results_container = st.container()
                
                # Process each file
                batch_timer = ProcessingTimer()
                for i, uploaded_file in enumerate(uploaded_files):
                    file_timer = ProcessingTimer()
                    progress = (i) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Update status
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Create a temporary file for the uploaded content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name
                    
                    try:
                        # Extract text
                        file_timer.start_phase("text_extraction")
                        text = processor.process_file(temp_path)
                        file_timer.end_phase()
                        
                        if text:
                            # Check token limit before processing
                            file_timer.start_phase("token_check")
                            within_limit, token_count = check_token_limit(text, llm_config["model"], provider_name)
                            file_timer.end_phase()
                            
                            if not within_limit:
                                error_msg = f"⚠️ Token limit exceeded for {uploaded_file.name}: {token_count} tokens"
                                st.warning(error_msg)
                                st.session_state.errors.append(error_msg)
                                failed_files.append((uploaded_file.name, "Token limit exceeded"))
                                continue
                            
                            # Process with LLM
                            file_timer.start_phase("llm_processing")
                            result = processor.extract_information(text)
                            file_timer.end_phase()
                            
                            if result:
                                # Save to markdown
                                file_timer.start_phase("file_saving")
                                output_filename = f"{Path(uploaded_file.name).stem}.md"
                                output_path = batch_dir / output_filename
                                
                                if processor.format_to_markdown(result, str(output_path)):
                                    processed_files.append(output_path)
                                    # Store processing time and phases
                                    st.session_state.processing_times[uploaded_file.name] = {
                                        'total_time': file_timer.get_total_time(),
                                        'phases': file_timer.get_phase_times()
                                    }
                                    
                                    # Show results first (in the results container)
                                    with results_container:
                                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                                        st.markdown(f"Output saved to: `{output_filename}`")
                                        
                                        # All processing information in one expander
                                        with st.expander("View Processing Information", expanded=False):
                                            st.markdown("#### Current File Processing Phases")
                                            phase_times = file_timer.get_phase_times()
                                            for phase, duration in phase_times.items():
                                                st.text(f"{phase.replace('_', ' ').title()}: {format_time(duration)}")
                                            
                                            st.markdown("#### Processing Times")
                                            st.metric("Current File Total Time", format_time(file_timer.get_total_time()))
                                            st.metric("Total Processing Time", format_time(batch_timer.get_total_time()))
                                            if i > 0:
                                                est_remaining = estimate_remaining_time(i, len(uploaded_files), batch_timer.get_total_time())
                                                st.metric("Estimated Time Remaining", format_time(est_remaining))
                                else:
                                    error_msg = f"Failed to save results for {uploaded_file.name}"
                                    st.error(error_msg)
                                    st.session_state.errors.append(error_msg)
                                    failed_files.append((uploaded_file.name, "Failed to save results"))
                            else:
                                error_msg = f"Failed to process {uploaded_file.name}"
                                st.error(error_msg)
                                st.session_state.errors.append(error_msg)
                                failed_files.append((uploaded_file.name, "Processing failed"))
                        else:
                            error_msg = f"Failed to extract text from {uploaded_file.name}"
                            st.error(error_msg)
                            st.session_state.errors.append(error_msg)
                            failed_files.append((uploaded_file.name, "Text extraction failed"))
                    except Exception as e:
                        error_msg = f"Error processing {uploaded_file.name}: {str(e)}"
                        st.error(error_msg)
                        st.session_state.errors.append(error_msg)
                        failed_files.append((uploaded_file.name, str(e)))
                    finally:
                        # Clean up temporary file
                        os.unlink(temp_path)
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Show final results and summary
                with results_container:
                    st.markdown("### Results")
                    if processed_files:
                        st.markdown(f"Files have been saved to: `{batch_dir}`")
                        
                        with st.expander("View Processing Summary", expanded=False):
                            st.markdown(f"""
                            - ✅ Successfully processed: {len(processed_files)} files
                            - ❌ Failed: {len(failed_files)} files
                            - ⏱️ Total processing time: {format_time(batch_timer.get_total_time())}
                            """)
                            
                            # Show detailed timing statistics
                            st.markdown("#### Processing Times per File")
                            # Calculate and display phase statistics
                            all_phases = set()
                            phase_totals = {}
                            for file_data in st.session_state.processing_times.values():
                                for phase, time in file_data['phases'].items():
                                    all_phases.add(phase)
                                    phase_totals[phase] = phase_totals.get(phase, 0) + time
                            
                            total_time = sum(data['total_time'] for data in st.session_state.processing_times.values())
                            for phase in sorted(all_phases):
                                phase_time = phase_totals[phase]
                                percentage = (phase_time / total_time) * 100
                                st.markdown(f"- **{phase.replace('_', ' ').title()}**:")
                                st.markdown(f"  - Total: {format_time(phase_time)}")
                                st.markdown(f"  - Average: {format_time(phase_time / len(processed_files))}")
                                st.markdown(f"  - Percentage: {percentage:.1f}%")
                        
                        if len(processed_files) > 1:
                            avg_time = sum(data['total_time'] for data in st.session_state.processing_times.values()) / len(processed_files)
                            st.markdown(f"\n**Average processing time per file**: {format_time(avg_time)}")
                
                if failed_files:
                    with st.expander("Show Failed Files"):
                        for file_name, reason in failed_files:
                            st.markdown(f"- **{file_name}**: {reason}")
                
                if processed_files:
                    st.markdown("### Results")
                    st.markdown(f"Files have been saved to: `{batch_dir}`")
                    
                    # Show individual file previews
                    for output_file in processed_files:
                        with st.expander(f"Preview: {output_file.name}"):
                            with open(output_file, "r", encoding="utf-8") as f:
                                content = f.read()
                                st.markdown(content)
                    
                    # Option to download all as zip
                    if len(processed_files) > 1:
                        st.markdown("#### Download All Files")
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                            for output_file in processed_files:
                                zipf.write(output_file, output_file.name)
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            "Download All as ZIP",
                            zip_buffer,
                            file_name="processed_files.zip",
                            mime="application/zip",
                            help="Download all processed files as a ZIP archive"
                        )
            
            except Exception as e:
                error_msg = f"Error initializing provider: {str(e)}"
                st.error(error_msg)
                st.session_state.errors.append(error_msg)
    
    # Add documentation
    with st.expander("📖 Documentation"):
        st.markdown("""
        ### About this app
        
        This application uses various LLM providers to extract information from document files.
        
        **Supported Providers:**
        - OpenAI (GPT-3.5, GPT-4)
        - Ollama (Local models)
        - Deepseek
        - OpenAI Compatible
        
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
        """)

if __name__ == "__main__":
    main()
