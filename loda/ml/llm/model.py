"""
Transformer-based model for natural language to LODA code generation.

This module implements an encoder-decoder transformer architecture using Hugging Face
transformers, specifically designed for sequence-to-sequence tasks like converting
natural language descriptions to LODA assembly code.
"""

import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    PreTrainedTokenizer,
    PreTrainedModel
)
from typing import List, Dict, Optional, Tuple
import json
import os


class LodaTokenizer:
    """Custom tokenizer for LODA assembly language."""
    
    def __init__(self):
        """Initialize LODA tokenizer with vocabulary."""
        # LODA operations
        self.operations = [
            'mov', 'add', 'sub', 'mul', 'div', 'dif', 'mod', 'pow', 'gcd', 'bin',
            'cmp', 'min', 'max', 'lpb', 'lpe', 'nop', 'cal', 'seq', 'trn', 'clr'
        ]
        
        # Common operand patterns
        self.operand_patterns = [
            # Direct memory references
            '$0', '$1', '$2', '$3', '$4', '$5', '$6', '$7', '$8', '$9', '$10',
            # Indirect memory references  
            '$$1', '$$2', '$$3', '$$4', '$$5',
            # Common constants
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '-1'
        ]
        
        # Special tokens
        self.special_tokens = ['<pad>', '<unk>', '<s>', '</s>', '<mask>']
        
        # Build vocabulary
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
        
        # Add operations
        for token in self.operations:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
        
        # Add operand patterns
        for token in self.operand_patterns:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
        
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab['<pad>']
        self.unk_token_id = self.vocab['<unk>']
        self.bos_token_id = self.vocab['<s>']
        self.eos_token_id = self.vocab['</s>']
    
    def tokenize_loda_code(self, code: str) -> List[str]:
        """
        Tokenize LODA assembly code.
        
        Args:
            code: LODA assembly code as string
            
        Returns:
            List of tokens
        """
        lines = code.strip().split('\n')
        tokens = ['<s>']  # Start token
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split on whitespace and comma
            parts = line.replace(',', ' ').split()
            
            for part in parts:
                part = part.strip()
                if part in self.vocab:
                    tokens.append(part)
                else:
                    # Try to handle unknown operands
                    if part.startswith('$') and part[1:].isdigit():
                        # Direct memory reference
                        if part in self.vocab:
                            tokens.append(part)
                        else:
                            tokens.append('<unk>')
                    elif part.startswith('$$') and part[2:].isdigit():
                        # Indirect memory reference
                        if part in self.vocab:
                            tokens.append(part)
                        else:
                            tokens.append('<unk>')
                    elif part.lstrip('-').isdigit():
                        # Numeric constant
                        if part in self.vocab:
                            tokens.append(part)
                        else:
                            tokens.append('<unk>')
                    else:
                        tokens.append('<unk>')
        
        tokens.append('</s>')  # End token
        return tokens
    
    def encode_loda_code(self, code: str) -> List[int]:
        """
        Encode LODA code to token IDs.
        
        Args:
            code: LODA assembly code
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize_loda_code(code)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def decode_loda_code(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to LODA code.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            LODA assembly code as string
        """
        tokens = [self.reverse_vocab.get(id, '<unk>') for id in token_ids]
        
        # Filter out special tokens
        filtered_tokens = []
        for token in tokens:
            if token in ['<s>', '</s>', '<pad>']:
                continue
            if token == '<unk>':
                continue
            filtered_tokens.append(token)
        
        # Reconstruct LODA code
        code_lines = []
        i = 0
        
        while i < len(filtered_tokens):
            if i + 2 < len(filtered_tokens):
                # Try to form operation: op target source
                op = filtered_tokens[i]
                if op in self.operations and i + 2 < len(filtered_tokens):
                    target = filtered_tokens[i + 1]
                    source = filtered_tokens[i + 2]
                    code_lines.append(f"{op} {target},{source}")
                    i += 3
                elif op in self.operations and i + 1 < len(filtered_tokens):
                    # Single operand operation
                    target = filtered_tokens[i + 1]
                    code_lines.append(f"{op} {target}")
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return '\n'.join(code_lines)


class LodaT5Model(nn.Module):
    """
    T5-based model for natural language to LODA code generation.
    """
    
    def __init__(self, model_name: str = "t5-small", loda_vocab_size: Optional[int] = None):
        """
        Initialize the model.
        
        Args:
            model_name: Base T5 model to use
            loda_vocab_size: Size of LODA vocabulary (if extending tokenizer)
        """
        super().__init__()
        
        # Load base T5 model and tokenizer
        self.text_tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Initialize LODA tokenizer
        self.loda_tokenizer = LodaTokenizer()
        
        # If we need to extend the vocabulary
        if loda_vocab_size and loda_vocab_size > self.loda_tokenizer.vocab_size:
            # Could extend vocabulary here if needed
            pass
    
    def prepare_input(self, descriptions: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepare natural language descriptions for input.
        
        Args:
            descriptions: List of natural language descriptions
            
        Returns:
            Dictionary with input tensors
        """
        # Add task prefix for T5
        prefixed_descriptions = [f"translate to loda: {desc}" for desc in descriptions]
        
        # Tokenize with T5 tokenizer
        encoded = self.text_tokenizer(
            prefixed_descriptions,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return encoded
    
    def prepare_target(self, loda_codes: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepare LODA codes as targets.
        
        Args:
            loda_codes: List of LODA assembly codes
            
        Returns:
            Dictionary with target tensors
        """
        # For T5, we need to encode targets using the text tokenizer as well
        # We'll create a custom format that represents LODA code
        
        # Convert LODA to a text representation that T5 can understand
        text_loda_codes = []
        for code in loda_codes:
            # Convert LODA code to a more text-like format
            text_code = self.loda_to_text_format(code)
            text_loda_codes.append(text_code)
        
        encoded = self.text_tokenizer(
            text_loda_codes,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        return encoded
    
    def loda_to_text_format(self, code: str) -> str:
        """
        Convert LODA code to a text format suitable for T5.
        
        This creates a more natural language representation of LODA code.
        
        Args:
            code: LODA assembly code
            
        Returns:
            Text representation of the code
        """
        lines = code.strip().split('\n')
        text_parts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse the line and convert to text
            parts = line.replace(',', ' ').split()
            if len(parts) >= 3:
                op, target, source = parts[0], parts[1], parts[2]
                text_parts.append(f"{op} {target} {source}")
            elif len(parts) >= 2:
                op, target = parts[0], parts[1]
                text_parts.append(f"{op} {target}")
            else:
                text_parts.append(line)
        
        return " | ".join(text_parts)
    
    def text_format_to_loda(self, text_code: str) -> str:
        """
        Convert text format back to LODA code.
        
        Args:
            text_code: Text representation of LODA code
            
        Returns:
            LODA assembly code
        """
        parts = text_code.split(" | ")
        loda_lines = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            tokens = part.split()
            if len(tokens) >= 3:
                op, target, source = tokens[0], tokens[1], tokens[2]
                loda_lines.append(f"{op} {target},{source}")
            elif len(tokens) >= 2:
                op, target = tokens[0], tokens[1]
                loda_lines.append(f"{op} {target}")
            else:
                loda_lines.append(part)
        
        return '\n'.join(loda_lines)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (for training)
            
        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, descriptions: List[str], max_length: int = 256, num_beams: int = 4) -> List[str]:
        """
        Generate LODA code from natural language descriptions.
        
        Args:
            descriptions: List of natural language descriptions
            max_length: Maximum length of generated sequences
            num_beams: Number of beams for beam search
            
        Returns:
            List of generated LODA codes
        """
        # Prepare input
        inputs = self.prepare_input(descriptions)
        
        # Generate with the model
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode generated sequences
        generated_texts = self.text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Convert from text format back to LODA
        loda_codes = [self.text_format_to_loda(text) for text in generated_texts]
        
        return loda_codes
    
    def save_model(self, save_path: str):
        """
        Save the model and tokenizers.
        
        Args:
            save_path: Directory to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save T5 model and tokenizer
        self.model.save_pretrained(save_path)
        self.text_tokenizer.save_pretrained(save_path)
        
        # Save LODA tokenizer
        loda_tokenizer_path = os.path.join(save_path, "loda_tokenizer.json")
        with open(loda_tokenizer_path, 'w') as f:
            json.dump({
                'vocab': self.loda_tokenizer.vocab,
                'reverse_vocab': {str(k): v for k, v in self.loda_tokenizer.reverse_vocab.items()}
            }, f, indent=2)
    
    @classmethod
    def load_model(cls, load_path: str):
        """
        Load a saved model.
        
        Args:
            load_path: Directory containing the saved model
            
        Returns:
            Loaded LodaT5Model instance
        """
        # Load T5 model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained(load_path)
        text_tokenizer = T5Tokenizer.from_pretrained(load_path)
        
        # Create model instance
        loda_model = cls()
        loda_model.model = model
        loda_model.text_tokenizer = text_tokenizer
        
        # Load LODA tokenizer if it exists
        loda_tokenizer_path = os.path.join(load_path, "loda_tokenizer.json")
        if os.path.exists(loda_tokenizer_path):
            with open(loda_tokenizer_path, 'r') as f:
                tokenizer_data = json.load(f)
            
            loda_model.loda_tokenizer.vocab = tokenizer_data['vocab']
            loda_model.loda_tokenizer.reverse_vocab = {
                int(k): v for k, v in tokenizer_data['reverse_vocab'].items()
            }
        
        return loda_model


def create_model(model_name: str = "t5-small") -> LodaT5Model:
    """
    Create a new LodaT5Model.
    
    Args:
        model_name: Base T5 model to use
        
    Returns:
        New LodaT5Model instance
    """
    return LodaT5Model(model_name)