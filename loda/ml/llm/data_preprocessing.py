"""
Data preprocessing utilities for LLM training on OEIS sequences and LODA programs.

This module handles:
1. Extracting sequence descriptions from LODA program comments
2. Pairing natural language descriptions with LODA code
3. Creating training datasets for sequence-to-sequence models
4. Tokenization and data formatting for transformer models
"""

import os
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from loda.lang import Program
from loda.oeis import ProgramCache, Sequence


@dataclass
class TrainingExample:
    """A single training example pairing natural language with LODA code."""
    sequence_id: str
    description: str
    loda_code: str
    terms: Optional[List[int]] = None


class DataPreprocessor:
    """Handles preprocessing of OEIS programs for LLM training."""
    
    def __init__(self, programs_dir: str):
        """Initialize with path to OEIS programs directory."""
        self.programs_dir = programs_dir
        self.program_cache = ProgramCache(programs_dir)
        
    def extract_description_from_program(self, program_text: str) -> Optional[str]:
        """
        Extract the natural language description from a LODA program.
        
        LODA programs typically start with comments like:
        ; A000045: Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.
        
        Args:
            program_text: The full LODA program as text
            
        Returns:
            The description string or None if no description found
        """
        lines = program_text.strip().split('\n')
        
        for line in lines:
            # Look for OEIS description lines (start with ; A######:)
            match = re.match(r';\s*A\d{6}:\s*(.+)', line)
            if match:
                description = match.group(1).strip()
                # Clean up common artifacts
                description = description.rstrip('.')
                # Remove mathematical notation that might be confusing
                # Keep it simple for initial training
                return description
                
        return None
    
    def extract_terms_from_program(self, program_text: str) -> Optional[List[int]]:
        """
        Extract the sequence terms from a LODA program comment.
        
        Args:
            program_text: The full LODA program as text
            
        Returns:
            List of sequence terms or None if not found
        """
        lines = program_text.strip().split('\n')
        
        for line in lines:
            # Look for lines with comma-separated numbers (sequence terms)
            if line.startswith(';') and ',' in line:
                # Extract numbers from the line
                numbers_str = line[1:].strip()  # Remove the ';'
                # Skip if it looks like it contains non-numeric content
                if ':' in numbers_str or any(c.isalpha() for c in numbers_str):
                    continue
                    
                try:
                    terms = [int(x.strip()) for x in numbers_str.split(',') if x.strip()]
                    if len(terms) >= 5:  # Reasonable number of terms
                        return terms
                except ValueError:
                    continue
                    
        return None
    
    def clean_loda_code(self, program_text: str) -> str:
        """
        Clean LODA code by removing comments and normalizing format.
        
        Args:
            program_text: Raw LODA program text
            
        Returns:
            Cleaned LODA code suitable for training
        """
        lines = program_text.strip().split('\n')
        code_lines = []
        
        for line in lines:
            # Skip comment lines
            if line.strip().startswith(';'):
                continue
            # Skip empty lines
            if not line.strip():
                continue
            # Add the code line
            code_lines.append(line.strip())
        
        return '\n'.join(code_lines)
    
    def create_training_examples(self, max_examples: int = -1) -> List[TrainingExample]:
        """
        Create training examples from all available LODA programs.
        
        Args:
            max_examples: Maximum number of examples to create (-1 for all)
            
        Returns:
            List of TrainingExample objects
        """
        examples = []
        program_ids = self.program_cache.all_ids()
        
        if max_examples > 0:
            program_ids = program_ids[:max_examples]
        
        print(f"Processing {len(program_ids)} programs...")
        
        for i, program_id in enumerate(program_ids):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(program_ids)} programs")
                
            try:
                # Read the program file
                program_path = self.program_cache._get_path(program_id)
                if not os.path.exists(program_path):
                    continue
                    
                with open(program_path, 'r') as f:
                    program_text = f.read()
                
                # Extract description
                description = self.extract_description_from_program(program_text)
                if not description:
                    continue
                
                # Extract terms (optional)
                terms = self.extract_terms_from_program(program_text)
                
                # Clean the LODA code
                clean_code = self.clean_loda_code(program_text)
                if not clean_code:
                    continue
                
                # Validate that the code parses correctly
                try:
                    Program.parse(clean_code)
                except Exception:
                    continue  # Skip programs that don't parse
                
                example = TrainingExample(
                    sequence_id=program_id,
                    description=description,
                    loda_code=clean_code,
                    terms=terms
                )
                examples.append(example)
                
            except Exception as e:
                print(f"Error processing {program_id}: {e}")
                continue
        
        print(f"Created {len(examples)} training examples")
        return examples
    
    def augment_descriptions(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """
        Augment training examples with variations of descriptions.
        
        This can help make the model more robust to different phrasings.
        
        Args:
            examples: List of original training examples
            
        Returns:
            Augmented list with additional variations
        """
        augmented = list(examples)  # Start with originals
        
        for example in examples:
            desc = example.description
            
            # Create variations
            variations = []
            
            # Add "sequence of" prefix if not present
            if not desc.lower().startswith(('sequence', 'the sequence')):
                variations.append(f"Sequence of {desc.lower()}")
            
            # Add "Generate" prefix
            variations.append(f"Generate {desc.lower()}")
            
            # Add "Compute" prefix
            variations.append(f"Compute {desc.lower()}")
            
            # Remove mathematical symbols for simpler versions
            simple_desc = re.sub(r'[()=+\-*/^]', ' ', desc)
            simple_desc = re.sub(r'\s+', ' ', simple_desc).strip()
            if simple_desc != desc and simple_desc:
                variations.append(simple_desc)
            
            # Create new examples for each variation
            for variation in variations:
                augmented_example = TrainingExample(
                    sequence_id=example.sequence_id + "_aug",
                    description=variation,
                    loda_code=example.loda_code,
                    terms=example.terms
                )
                augmented.append(augmented_example)
        
        return augmented
    
    def save_dataset(self, examples: List[TrainingExample], output_file: str):
        """
        Save training examples to a file for later use.
        
        Args:
            examples: List of training examples
            output_file: Path to output file
        """
        import json
        
        data = []
        for example in examples:
            data.append({
                'sequence_id': example.sequence_id,
                'description': example.description,
                'loda_code': example.loda_code,
                'terms': example.terms
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(examples)} examples to {output_file}")
    
    def load_dataset(self, input_file: str) -> List[TrainingExample]:
        """
        Load training examples from a file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of TrainingExample objects
        """
        import json
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            example = TrainingExample(
                sequence_id=item['sequence_id'],
                description=item['description'],
                loda_code=item['loda_code'],
                terms=item.get('terms')
            )
            examples.append(example)
        
        print(f"Loaded {len(examples)} examples from {input_file}")
        return examples


def create_dataset(programs_dir: str, output_file: str, max_examples: int = -1, augment: bool = True):
    """
    Convenience function to create and save a training dataset.
    
    Args:
        programs_dir: Path to OEIS programs directory
        output_file: Path to save the dataset
        max_examples: Maximum number of examples (-1 for all)
        augment: Whether to augment with description variations
    """
    preprocessor = DataPreprocessor(programs_dir)
    examples = preprocessor.create_training_examples(max_examples)
    
    if augment:
        examples = preprocessor.augment_descriptions(examples)
    
    preprocessor.save_dataset(examples, output_file)
    return examples


if __name__ == "__main__":
    # Example usage
    programs_dir = "programs/oeis"
    dataset = create_dataset(programs_dir, "loda_training_data.json", max_examples=1000)
    print(f"Created dataset with {len(dataset)} examples")