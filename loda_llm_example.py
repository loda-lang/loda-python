#!/usr/bin/env python3
"""
Example script demonstrating LODA LLM usage.

This script shows how to:
1. Create training data from OEIS programs
2. Train an LLM model
3. Generate LODA code from natural language
4. Evaluate model performance

Run with: python loda_llm_example.py
"""

import os
import sys
import tempfile
from loda.llm import (
    create_dataset, 
    train_loda_llm, 
    LodaGenerator,
    LodaEvaluator
)


def main():
    print("LODA LLM Example")
    print("=" * 50)
    
    # Check if programs directory exists
    programs_dir = "programs/oeis"
    if not os.path.exists(programs_dir):
        print(f"Error: Programs directory '{programs_dir}' not found.")
        print("Please ensure you have the OEIS programs directory.")
        return 1
    
    # Create temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Step 1: Create training dataset (small sample for demo)
        print("\n1. Creating training dataset...")
        dataset_file = os.path.join(temp_dir, "training_data.json")
        
        try:
            examples = create_dataset(
                programs_dir=programs_dir,
                output_file=dataset_file,
                max_examples=100,  # Small sample for quick demo
                augment=True
            )
            print(f"Created {len(examples)} training examples")
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return 1
        
        # Step 2: Train a small model (for demonstration)
        print("\n2. Training LLM model...")
        model_dir = os.path.join(temp_dir, "model")
        
        try:
            model = train_loda_llm(
                programs_dir=programs_dir,
                output_dir=model_dir,
                model_name="t5-small",  # Small model for quick training
                max_examples=50,  # Very small for demo
                num_epochs=1,  # Single epoch for demo
                batch_size=4,
                learning_rate=1e-4
            )
            print("Training completed!")
            
        except Exception as e:
            print(f"Error training model: {e}")
            print("Note: This requires PyTorch and transformers to be installed.")
            print("Install with: pip install torch transformers")
            return 1
        
        # Step 3: Generate code from natural language
        print("\n3. Generating LODA code...")
        
        try:
            generator = LodaGenerator(model)
            
            test_descriptions = [
                "Fibonacci numbers",
                "Powers of 2",
                "Square numbers",
                "Natural numbers",
                "Factorial numbers"
            ]
            
            for description in test_descriptions:
                print(f"\nDescription: {description}")
                results = generator.generate(description, num_samples=1)
                
                if results:
                    result = results[0]
                    print(f"Generated in {result.generation_time:.2f}s")
                    print(f"Valid: {result.is_valid}")
                    
                    if result.error_message:
                        print(f"Error: {result.error_message}")
                    
                    print("Generated code:")
                    for line in result.generated_code.split('\n'):
                        if line.strip():
                            print(f"  {line}")
                    
                    if result.generated_sequence:
                        print(f"Sequence: {result.generated_sequence}")
                
                print("-" * 40)
                
        except Exception as e:
            print(f"Error generating code: {e}")
            return 1
        
        # Step 4: Demonstrate evaluation (if we have test data)
        print("\n4. Model evaluation...")
        
        try:
            evaluator = LodaEvaluator(model)
            
            # Use a subset of the training data as test data for demo
            from loda.llm.data_preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor(programs_dir)
            test_examples = preprocessor.create_training_examples(max_examples=10)
            
            if test_examples:
                metrics, eval_results = evaluator.evaluate_examples(test_examples)
                
                print(f"Evaluation Results:")
                print(f"  Total examples: {metrics['total_examples']}")
                print(f"  Valid programs: {metrics['valid_programs']} ({metrics['valid_program_rate']:.1%})")
                print(f"  Exact matches: {metrics['exact_matches']} ({metrics['exact_match_rate']:.1%})")
                print(f"  Sequence matches: {metrics['sequence_matches']} ({metrics['sequence_match_rate']:.1%})")
                print(f"  Avg generation time: {metrics['avg_generation_time']:.2f}s")
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("To use the LLM in your own code:")
    print("1. Train a model: train_loda_llm('programs/oeis', 'my_model')")
    print("2. Load for inference: generator = LodaGenerator.load_model('my_model')")
    print("3. Generate code: results = generator.generate('your description')")
    print("\nCommand line usage:")
    print("- Train: python -m loda.llm.trainer --programs_dir programs/oeis")
    print("- Interactive: python -m loda.llm.inference --mode interactive --model_path my_model")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)