import os
import logging
import argparse
from typing import Optional, List, Dict
from pathlib import Path
from abc import ABC, abstractmethod
import requests

import fitz  # PyMuPDF
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self):
        self.system_prompt = "Extract questions and answers from the given text."
        self.user_prompt_template = "Extract all questions and answers from the following text:\n\n{text}\n\nFormat as: Question: ... Answer: ..."
    
    def set_prompts(self, system_prompt: str, user_prompt_template: str):
        """Set custom prompts for the provider"""
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
    
    @abstractmethod
    def generate_qa(self, text: str) -> Optional[str]:
        """Generate Q&A from text using the LLM"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models from the provider"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__()
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise
        self.model = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_qa(self, text: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt_template.format(text=text)}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            return None
    
    def list_models(self) -> List[str]:
        """List available chat models from OpenAI"""
        try:
            models = self.client.models.list()
            # Filter for chat models and include their max context window
            chat_models = []
            for model in models:
                if model.id.startswith(('gpt-4', 'gpt-3.5')):
                    chat_models.append(model.id)
            return sorted(chat_models)
        except Exception as e:
            logger.error(f"Error listing OpenAI models: {str(e)}")
            return []

class OllamaProvider(LLMProvider):
    """Ollama API provider"""
    
    def __init__(self, base_url: str, model: str = "llama2"):
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_qa(self, text: str) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{self.system_prompt}\n\n{self.user_prompt_template.format(text=text)}",
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            logger.error(f"Error in Ollama API call: {str(e)}")
            return None
    
    def list_models(self) -> List[str]:
        """List available models from Ollama local instance"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            if 'models' in data:
                return sorted([model['name'] for model in data['models']])
            else:
                logger.warning("No models found in Ollama response")
                return []
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Is it running?")
            return []
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return []

class DeepseekProvider(LLMProvider):
    """Deepseek API provider"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-chat"):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_qa(self, text: str) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt_template.format(text=text)}
                    ]
                }
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error in Deepseek API call: {str(e)}")
            return None
    
    def list_models(self) -> List[str]:
        """List available models from Deepseek API"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            if 'data' in data:
                return sorted([model['id'] for model in data['data']])
            else:
                logger.warning("No models found in Deepseek response")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Deepseek models: {str(e)}")
            return []

class PDFProcessor:
    def __init__(self, llm_provider: LLMProvider):
        """Initialize PDFProcessor with an LLM provider."""
        self.llm_provider = llm_provider
        
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                # Check file size
                if os.path.getsize(pdf_path) > 10_000_000:  # 10MB limit
                    raise ValueError("PDF file too large (>10MB)")
                    
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return None

    def extract_questions_answers(self, text: str) -> Optional[str]:
        """
        Extract questions and answers from text using the configured LLM provider.
        
        Args:
            text: Input text to process
            
        Returns:
            Formatted string of questions and answers
        """
        return self.llm_provider.generate_qa(text)

    def format_to_markdown(self, questions_answers: str, output_path: str) -> bool:
        """
        Format extracted questions and answers into Markdown.
        
        Args:
            questions_answers: String containing Q&A pairs
            output_path: Path to save the markdown file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as md_file:
                md_file.write("# Extracted Questions and Answers\n\n")
                for qa in questions_answers.split("\n\n"):
                    if "Question:" in qa and "Answer:" in qa:
                        md_file.write(f"## {qa}\n\n")
            return True
        except Exception as e:
            logger.error(f"Error writing markdown file {output_path}: {str(e)}")
            return False

    def get_pdf_files(self, input_path: str) -> List[str]:
        """
        Get list of PDF files from input path.
        
        Args:
            input_path: Path to PDF file or directory
            
        Returns:
            List of PDF file paths
        """
        input_path = Path(input_path)
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            return [str(input_path)]
        elif input_path.is_dir():
            return [str(p) for p in input_path.glob('**/*.pdf')]
        else:
            raise ValueError("Invalid input path. Provide a PDF file or a folder containing PDFs.")

    def process_pdfs(self, input_path: str, output_folder: str) -> None:
        """
        Process PDFs and generate Markdown files.
        
        Args:
            input_path: Path to PDF file or directory
            output_folder: Path to output directory
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            pdf_files = self.get_pdf_files(input_path)
            total_files = len(pdf_files)
            
            if total_files == 0:
                logger.warning("No PDF files found to process")
                return
                
            logger.info(f"Found {total_files} PDF files to process")
            
            for i, pdf_file in enumerate(pdf_files, 1):
                logger.info(f"Processing file {i}/{total_files}: {pdf_file}")
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_file)
                if not text:
                    logger.error(f"Skipping {pdf_file} due to text extraction failure")
                    continue
                
                # Extract Q&A
                questions_answers = self.extract_questions_answers(text)
                if not questions_answers:
                    logger.error(f"Skipping {pdf_file} due to Q&A extraction failure")
                    continue
                
                # Save to markdown
                output_path = os.path.join(
                    output_folder,
                    Path(pdf_file).stem + '.md'
                )
                if self.format_to_markdown(questions_answers, output_path):
                    logger.info(f"Successfully processed {pdf_file} -> {output_path}")
                else:
                    logger.error(f"Failed to save markdown for {pdf_file}")
                    
        except Exception as e:
            logger.error(f"Error processing PDFs: {str(e)}")
            raise

def main():
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(description='Extract Q&A from PDF files using LLM')
    parser.add_argument('input_path', help='Path to PDF file or directory containing PDFs')
    parser.add_argument('output_folder', help='Path to output directory for markdown files')
    parser.add_argument('--env-file', help='Path to .env file', default='.env')
    parser.add_argument('--provider', help='LLM provider (openai, ollama, deepseek)', default='openai')
    parser.add_argument('--base-url', help='Base URL for API (required for Ollama and Deepseek)')
    parser.add_argument('--model', help='Model name for the provider')
    parser.add_argument('--system-prompt', help='Custom system prompt for the provider')
    parser.add_argument('--user-prompt-template', help='Custom user prompt template for the provider')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)
    
    try:
        # Initialize appropriate LLM provider
        if args.provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            provider = OpenAIProvider(api_key, args.model or "gpt-3.5-turbo")
        elif args.provider == 'ollama':
            if not args.base_url:
                raise ValueError("base-url is required for Ollama provider")
            provider = OllamaProvider(args.base_url, args.model or "llama2")
        elif args.provider == 'deepseek':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key or not args.base_url:
                raise ValueError("DEEPSEEK_API_KEY and base-url are required for Deepseek provider")
            provider = DeepseekProvider(api_key, args.base_url, args.model or "deepseek-chat")
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")
        
        if args.system_prompt:
            provider.set_prompts(args.system_prompt, provider.user_prompt_template)
        if args.user_prompt_template:
            provider.set_prompts(provider.system_prompt, args.user_prompt_template)
        
        processor = PDFProcessor(provider)
        processor.process_pdfs(args.input_path, args.output_folder)
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
