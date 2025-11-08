"""
Large Language Model (LLM) implementation for natural language to LODA code generation.

This module provides functionality to train transformer-based models that can understand
natural language descriptions of integer sequences (like OEIS sequences) and generate
corresponding LODA assembly programs.

Key components:
- Data preprocessing for OEIS sequence descriptions and LODA programs
- Transformer-based encoder-decoder architecture
- Training pipeline with proper tokenization
- Inference utilities for code generation
- Evaluation metrics for generated programs

Example usage:
>>> from loda.ml.llm import LodaT5Model, LodaGenerator, train_loda_llm
>>> 
>>> # Train a model
>>> model = train_loda_llm("programs/oeis", "trained_model")
>>> 
>>> # Generate code
>>> generator = LodaGenerator(model)
>>> results = generator.generate("Fibonacci numbers")
>>> print(results[0].generated_code)
"""

# Import main classes for easy access
# Handle optional dependencies gracefully
try:
    from .model import LodaT5Model, LodaTokenizer
    from .trainer import LodaTrainer, train_loda_llm
    from .inference import LodaGenerator, LodaEvaluator, GenerationResult
    _llm_available = True
except ImportError:
    _llm_available = False
    # Create placeholder classes
    class _MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LLM functionality requires additional dependencies. "
                "Install with: pip install torch transformers datasets tqdm"
            )
    
    LodaT5Model = _MissingDependency
    LodaTokenizer = _MissingDependency
    LodaTrainer = _MissingDependency
    train_loda_llm = _MissingDependency
    LodaGenerator = _MissingDependency
    LodaEvaluator = _MissingDependency
    GenerationResult = _MissingDependency

# Data preprocessing doesn't require PyTorch/transformers
from .data_preprocessing import DataPreprocessor, TrainingExample, create_dataset

__all__ = [
    'LodaT5Model',
    'LodaTokenizer', 
    'LodaTrainer',
    'train_loda_llm',
    'LodaGenerator',
    'LodaEvaluator',
    'GenerationResult',
    'DataPreprocessor',
    'TrainingExample',
    'create_dataset',
    '_llm_available'
]