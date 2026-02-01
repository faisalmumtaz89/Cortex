"""Dataset preparation utilities for fine-tuning."""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepare and validate datasets for fine-tuning."""
    
    SUPPORTED_FORMATS = ['.jsonl', '.json', '.csv', '.txt']
    MIN_EXAMPLES = 5
    MAX_EXAMPLES = 10000
    
    def validate_dataset(self, file_path: Path) -> Tuple[bool, str, Optional[Path]]:
        """
        Validate and prepare a dataset file.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Tuple of (is_valid, message, processed_path)
        """
        try:
            # Check file exists
            if not file_path.exists():
                return False, f"File not found: {file_path}", None
            
            # Check file extension
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return False, f"Unsupported format. Supported: {', '.join(self.SUPPORTED_FORMATS)}", None
            
            # Process based on format
            if file_path.suffix.lower() == '.jsonl':
                return self._validate_jsonl(file_path)
            elif file_path.suffix.lower() == '.json':
                return self._validate_json(file_path)
            elif file_path.suffix.lower() == '.csv':
                return self._validate_csv(file_path)
            elif file_path.suffix.lower() == '.txt':
                return self._validate_txt(file_path)
            else:
                return False, "Unsupported format", None
                
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False, f"Validation error: {str(e)}", None
    
    def _validate_jsonl(self, file_path: Path) -> Tuple[bool, str, Optional[Path]]:
        """Validate JSONL format dataset."""
        try:
            examples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        return False, f"Invalid JSON at line {line_num}: {e}", None
                    
                    # Check for required fields
                    if 'prompt' in data and 'response' in data:
                        examples.append(data)
                    elif 'text' in data:
                        examples.append(data)
                    elif 'instruction' in data and 'output' in data:
                        # Convert to standard format
                        examples.append({
                            'prompt': data['instruction'],
                            'response': data['output']
                        })
                    else:
                        return False, f"Line {line_num}: Missing required fields (need 'prompt'/'response' or 'text')", None
            
            # Check example count
            if len(examples) < self.MIN_EXAMPLES:
                return False, f"Too few examples ({len(examples)}). Minimum: {self.MIN_EXAMPLES}", None
            
            if len(examples) > self.MAX_EXAMPLES:
                logger.warning(f"Dataset has {len(examples)} examples. Will use first {self.MAX_EXAMPLES}")
                examples = examples[:self.MAX_EXAMPLES]
            
            # Save processed dataset if needed
            processed_path = self._save_processed_dataset(examples, file_path)
            
            return True, f"Valid dataset with {len(examples)} examples", processed_path
            
        except Exception as e:
            return False, f"Error reading JSONL file: {e}", None
    
    def _validate_json(self, file_path: Path) -> Tuple[bool, str, Optional[Path]]:
        """Validate JSON format dataset."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of examples
                for item in data:
                    if isinstance(item, dict):
                        if 'prompt' in item and 'response' in item:
                            examples.append(item)
                        elif 'text' in item:
                            examples.append(item)
            elif isinstance(data, dict):
                # Single example or nested structure
                if 'examples' in data:
                    examples = data['examples']
                elif 'data' in data:
                    examples = data['data']
                elif 'prompt' in data and 'response' in data:
                    examples = [data]
            
            if not examples:
                return False, "No valid examples found in JSON file", None
            
            # Convert to JSONL format
            processed_path = file_path.with_suffix('.jsonl')
            with open(processed_path, 'w', encoding='utf-8') as f:
                for example in examples[:self.MAX_EXAMPLES]:
                    f.write(json.dumps(example) + '\n')
            
            return True, f"Converted JSON to JSONL with {len(examples)} examples", processed_path
            
        except Exception as e:
            return False, f"Error reading JSON file: {e}", None
    
    def _validate_csv(self, file_path: Path) -> Tuple[bool, str, Optional[Path]]:
        """Validate CSV format dataset."""
        try:
            examples = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Look for prompt/response columns
                    if 'prompt' in row and 'response' in row:
                        examples.append({
                            'prompt': row['prompt'],
                            'response': row['response']
                        })
                    elif 'instruction' in row and 'output' in row:
                        examples.append({
                            'prompt': row['instruction'],
                            'response': row['output']
                        })
                    elif 'question' in row and 'answer' in row:
                        examples.append({
                            'prompt': row['question'],
                            'response': row['answer']
                        })
                    elif 'text' in row:
                        examples.append({'text': row['text']})
            
            if not examples:
                return False, "No valid columns found (need 'prompt'/'response' or similar)", None
            
            # Convert to JSONL
            processed_path = file_path.with_suffix('.jsonl')
            with open(processed_path, 'w', encoding='utf-8') as f:
                for example in examples[:self.MAX_EXAMPLES]:
                    f.write(json.dumps(example) + '\n')
            
            return True, f"Converted CSV to JSONL with {len(examples)} examples", processed_path
            
        except Exception as e:
            return False, f"Error reading CSV file: {e}", None
    
    def _validate_txt(self, file_path: Path) -> Tuple[bool, str, Optional[Path]]:
        """Validate plain text format dataset."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            examples = []
            
            # Try to parse as Q&A format
            qa_pattern = r'(?:Q:|Question:)\s*(.*?)\s*(?:A:|Answer:)\s*(.*?)(?=(?:Q:|Question:)|$)'
            qa_matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if qa_matches:
                for question, answer in qa_matches:
                    examples.append({
                        'prompt': question.strip(),
                        'response': answer.strip()
                    })
            else:
                # Try to parse as conversation format
                conv_pattern = r'(?:User|Human):\s*(.*?)\s*(?:Assistant|AI|Bot):\s*(.*?)(?=(?:User|Human):|$)'
                conv_matches = re.findall(conv_pattern, content, re.DOTALL | re.IGNORECASE)
                
                if conv_matches:
                    for user_msg, assistant_msg in conv_matches:
                        examples.append({
                            'prompt': user_msg.strip(),
                            'response': assistant_msg.strip()
                        })
                else:
                    # Treat as single text block
                    # Split into chunks if too long
                    chunks = self._split_text_into_chunks(content, max_length=500)
                    for chunk in chunks[:self.MAX_EXAMPLES]:
                        examples.append({'text': chunk})
            
            if not examples:
                return False, "Could not parse text file into examples", None
            
            # Convert to JSONL
            processed_path = file_path.with_suffix('.jsonl')
            with open(processed_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
            
            return True, f"Parsed text file into {len(examples)} examples", processed_path
            
        except Exception as e:
            return False, f"Error reading text file: {e}", None
    
    def _split_text_into_chunks(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks of approximately max_length."""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 1 < max_length:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _save_processed_dataset(self, examples: List[Dict], original_path: Path) -> Path:
        """Save processed dataset if modifications were made."""
        # If already in correct format, return original path
        if all('prompt' in ex and 'response' in ex for ex in examples):
            return original_path
        
        # Otherwise, save processed version
        processed_dir = Path.home() / ".cortex" / "processed_datasets"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        processed_path = processed_dir / f"{original_path.stem}_processed.jsonl"
        
        with open(processed_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        return processed_path
    
    def create_sample_dataset(self, domain: str = "general") -> Path:
        """Create a sample dataset for testing."""
        samples = {
            "general": [
                {"prompt": "What is machine learning?", 
                 "response": "Machine learning is a type of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
                {"prompt": "Explain neural networks", 
                 "response": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information."},
                {"prompt": "What is deep learning?", 
                 "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn from large amounts of data."},
                {"prompt": "Define natural language processing", 
                 "response": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language."},
                {"prompt": "What is computer vision?", 
                 "response": "Computer vision is a field of AI that trains computers to interpret and understand visual information from the world."},
            ],
            "coding": [
                {"prompt": "How do I reverse a string in Python?", 
                 "response": "You can reverse a string in Python using slicing: `reversed_string = original_string[::-1]`"},
                {"prompt": "What is a list comprehension?", 
                 "response": "A list comprehension is a concise way to create lists in Python: `[expression for item in iterable if condition]`"},
                {"prompt": "Explain recursion", 
                 "response": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem."},
                {"prompt": "What is object-oriented programming?", 
                 "response": "Object-oriented programming (OOP) is a programming paradigm that organizes code into objects containing data and methods."},
                {"prompt": "How do I handle exceptions in Python?", 
                 "response": "Use try-except blocks: `try: risky_code() except Exception as e: handle_error(e)`"},
            ],
            "creative": [
                {"prompt": "Write a haiku about coding", 
                 "response": "Lines of logic flow,\nBugs emerge, then disappear,\nCode compiles at last."},
                {"prompt": "Create a metaphor for machine learning", 
                 "response": "Machine learning is like teaching a child to recognize patterns - showing many examples until they can identify new ones on their own."},
                {"prompt": "Describe a sunset poetically", 
                 "response": "The sun paints the sky in hues of amber and rose, a masterpiece that fades into the embrace of twilight."},
                {"prompt": "Write a short story opening", 
                 "response": "The old lighthouse keeper had seen many storms, but none quite like the one approaching that November evening."},
                {"prompt": "Create a motivational quote", 
                 "response": "Every line of code you write today is a step toward the solution you'll celebrate tomorrow."},
            ]
        }
        
        # Get samples for the specified domain
        domain_samples = samples.get(domain, samples["general"])
        
        # Save to file
        sample_dir = Path.home() / ".cortex" / "sample_datasets"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        sample_path = sample_dir / f"sample_{domain}.jsonl"
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            for sample in domain_samples:
                f.write(json.dumps(sample) + '\n')
        
        return sample_path