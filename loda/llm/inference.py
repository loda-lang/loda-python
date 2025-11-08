"""
Inference and evaluation utilities for the LODA LLM.

This module provides:
1. Text-to-LODA code generation
2. Model evaluation metrics
3. Program validation and testing
4. Utilities for interactive usage
"""

import os
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from loda.lang import Program
from loda.runtime import Interpreter, Evaluator
from loda.oeis import Sequence
from .model import LodaT5Model
from .data_preprocessing import TrainingExample


@dataclass
class GenerationResult:
    """Result of code generation."""
    description: str
    generated_code: str
    is_valid: bool
    error_message: Optional[str] = None
    generated_sequence: Optional[List[int]] = None
    generation_time: float = 0.0


class LodaGenerator:
    """Generator class for creating LODA code from natural language."""
    
    def __init__(self, model: LodaT5Model, max_length: int = 256, num_beams: int = 4):
        """
        Initialize the generator.
        
        Args:
            model: Trained LodaT5Model
            max_length: Maximum length of generated code
            num_beams: Number of beams for beam search
        """
        self.model = model
        self.max_length = max_length
        self.num_beams = num_beams
    
    def generate(self, description: str, num_samples: int = 1) -> List[GenerationResult]:
        """
        Generate LODA code from a natural language description.
        
        Args:
            description: Natural language description of the sequence
            num_samples: Number of code samples to generate
            
        Returns:
            List of GenerationResult objects
        """
        start_time = time.time()
        
        # Generate multiple samples
        descriptions = [description] * num_samples
        generated_codes = self.model.generate(
            descriptions, 
            max_length=self.max_length,
            num_beams=self.num_beams
        )
        
        generation_time = time.time() - start_time
        
        results = []
        for code in generated_codes:
            result = self._validate_and_evaluate_code(description, code)
            result.generation_time = generation_time / num_samples
            results.append(result)
        
        return results
    
    def _validate_and_evaluate_code(self, description: str, code: str) -> GenerationResult:
        """
        Validate and evaluate generated LODA code.
        
        Args:
            description: Original description
            code: Generated LODA code
            
        Returns:
            GenerationResult with validation info
        """
        result = GenerationResult(
            description=description,
            generated_code=code,
            is_valid=False
        )
        
        try:
            # Try to parse the program
            program = Program(code)
            
            # Try to evaluate it for a few terms
            interpreter = Interpreter(max_memory=100, max_stack=10, max_steps=10000)
            evaluator = Evaluator(program, interpreter)
            
            sequence_terms = []
            for i in range(10):  # Generate first 10 terms
                try:
                    term = evaluator(i)
                    sequence_terms.append(term)
                except Exception:
                    break  # Stop if evaluation fails
            
            if len(sequence_terms) >= 3:  # At least 3 terms generated
                result.is_valid = True
                result.generated_sequence = sequence_terms
            else:
                result.error_message = "Could not generate sufficient sequence terms"
        
        except Exception as e:
            result.error_message = f"Program validation failed: {str(e)}"
        
        return result
    
    def generate_interactive(self):
        """Interactive mode for generating LODA code."""
        print("LODA Code Generator - Interactive Mode")
        print("Enter natural language descriptions to generate LODA code.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                description = input("Description: ").strip()
                
                if description.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not description:
                    continue
                
                print("Generating code...")
                results = self.generate(description, num_samples=1)
                
                for i, result in enumerate(results):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Generated in {result.generation_time:.2f}s")
                    print(f"Valid: {result.is_valid}")
                    
                    if result.error_message:
                        print(f"Error: {result.error_message}")
                    
                    print("Generated LODA code:")
                    print(result.generated_code)
                    
                    if result.generated_sequence:
                        print(f"Sequence terms: {result.generated_sequence}")
                    
                    print("-" * 50)
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


class LodaEvaluator:
    """Evaluator for assessing model performance."""
    
    def __init__(self, model: LodaT5Model):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained LodaT5Model to evaluate
        """
        self.model = model
        self.generator = LodaGenerator(model)
    
    def evaluate_examples(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """
        Evaluate the model on test examples.
        
        Args:
            test_examples: List of test examples
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating on {len(test_examples)} examples...")
        
        total_examples = len(test_examples)
        valid_programs = 0
        exact_matches = 0
        sequence_matches = 0
        total_generation_time = 0
        
        results = []
        
        for i, example in enumerate(test_examples):
            if i % 10 == 0:
                print(f"Progress: {i}/{total_examples}")
            
            # Generate code
            generation_results = self.generator.generate(example.description, num_samples=1)
            
            if generation_results:
                result = generation_results[0]
                results.append(result)
                
                total_generation_time += result.generation_time
                
                if result.is_valid:
                    valid_programs += 1
                    
                    # Check for exact match
                    if self._normalize_code(result.generated_code) == self._normalize_code(example.loda_code):
                        exact_matches += 1
                    
                    # Check for sequence match (if we have expected terms)
                    if (example.terms and result.generated_sequence and 
                        len(result.generated_sequence) >= 3 and
                        result.generated_sequence[:3] == example.terms[:3]):
                        sequence_matches += 1
        
        # Calculate metrics
        metrics = {
            'total_examples': total_examples,
            'valid_program_rate': valid_programs / total_examples if total_examples > 0 else 0,
            'exact_match_rate': exact_matches / total_examples if total_examples > 0 else 0,
            'sequence_match_rate': sequence_matches / total_examples if total_examples > 0 else 0,
            'avg_generation_time': total_generation_time / total_examples if total_examples > 0 else 0,
            'valid_programs': valid_programs,
            'exact_matches': exact_matches,
            'sequence_matches': sequence_matches
        }
        
        return metrics, results
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove extra whitespace and normalize format
        lines = []
        for line in code.strip().split('\n'):
            line = line.strip()
            if line:
                lines.append(line)
        return '\n'.join(lines)
    
    def print_evaluation_report(self, metrics: Dict[str, float], results: List[GenerationResult]):
        """Print a detailed evaluation report."""
        print("\n" + "="*60)
        print("LODA LLM EVALUATION REPORT")
        print("="*60)
        
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Valid Programs: {metrics['valid_programs']} ({metrics['valid_program_rate']:.1%})")
        print(f"Exact Matches: {metrics['exact_matches']} ({metrics['exact_match_rate']:.1%})")
        print(f"Sequence Matches: {metrics['sequence_matches']} ({metrics['sequence_match_rate']:.1%})")
        print(f"Avg Generation Time: {metrics['avg_generation_time']:.2f}s")
        
        # Show some example results
        print("\n" + "-"*60)
        print("SAMPLE RESULTS")
        print("-"*60)
        
        # Show successful examples
        successful = [r for r in results if r.is_valid]
        if successful:
            print("\nSuccessful generations:")
            for i, result in enumerate(successful[:3]):  # Show first 3
                print(f"\n{i+1}. Description: {result.description}")
                print(f"   Generated: {result.generated_code.replace(chr(10), '; ')}")
                if result.generated_sequence:
                    print(f"   Sequence: {result.generated_sequence}")
        
        # Show failed examples
        failed = [r for r in results if not r.is_valid]
        if failed:
            print(f"\nFailed generations ({len(failed)} total):")
            for i, result in enumerate(failed[:3]):  # Show first 3
                print(f"\n{i+1}. Description: {result.description}")
                print(f"   Error: {result.error_message}")
                print(f"   Generated: {result.generated_code.replace(chr(10), '; ')}")


def load_model_for_inference(model_path: str) -> LodaGenerator:
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        LodaGenerator instance ready for inference
    """
    model = LodaT5Model.load_model(model_path)
    return LodaGenerator(model)


def evaluate_model(model_path: str, test_data_path: str):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to the saved model
        test_data_path: Path to test data JSON file
    """
    # Load model
    print("Loading model...")
    model = LodaT5Model.load_model(model_path)
    evaluator = LodaEvaluator(model)
    
    # Load test data
    print("Loading test data...")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    test_examples = []
    for item in test_data:
        example = TrainingExample(
            sequence_id=item['sequence_id'],
            description=item['description'],
            loda_code=item['loda_code'],
            terms=item.get('terms')
        )
        test_examples.append(example)
    
    # Evaluate
    metrics, results = evaluator.evaluate_examples(test_examples)
    evaluator.print_evaluation_report(metrics, results)
    
    return metrics, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LODA LLM Inference and Evaluation")
    parser.add_argument("--mode", choices=["interactive", "evaluate"], required=True,
                        help="Mode to run in")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--test_data", type=str,
                        help="Path to test data (for evaluate mode)")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        generator = load_model_for_inference(args.model_path)
        generator.generate_interactive()
    
    elif args.mode == "evaluate":
        if not args.test_data:
            print("Test data path is required for evaluate mode")
            exit(1)
        evaluate_model(args.model_path, args.test_data)