# LODA LLM: Natural Language to Assembly Code Generation

This module extends the LODA Python project with Large Language Model (LLM) capabilities for generating LODA assembly code from natural language descriptions of integer sequences.

## Overview

The LODA LLM system can understand descriptions like "Fibonacci numbers" or "squares of positive integers" and generate corresponding LODA assembly programs that compute these sequences.

### Key Features

- **Transformer-based Architecture**: Uses T5 encoder-decoder model for sequence-to-sequence translation
- **OEIS Integration**: Trained on 145,000+ OEIS sequence descriptions and LODA programs  
- **Robust Preprocessing**: Extracts and augments training data from existing LODA programs
- **Comprehensive Evaluation**: Validates generated programs and evaluates sequence correctness
- **Interactive Interface**: Command-line tool for real-time code generation

## Architecture

```
Natural Language → T5 Encoder → Hidden Representation → T5 Decoder → LODA Code
     ↓                                                                    ↓
"Fibonacci numbers"                                              "mov $1,$0\n..."
```

### Components

1. **Data Preprocessing** (`data_preprocessing.py`)
   - Extracts sequence descriptions from LODA program comments
   - Creates training pairs of (description, LODA code)
   - Augments data with description variations
   - Handles data cleaning and validation

2. **Model Architecture** (`model.py`)
   - T5-based encoder-decoder transformer
   - Custom LODA tokenizer for assembly syntax
   - Text format conversion for T5 compatibility
   - Model saving/loading utilities

3. **Training Pipeline** (`trainer.py`)
   - PyTorch training loop with proper batching
   - Learning rate scheduling and gradient clipping
   - Validation and checkpointing
   - Support for different T5 model sizes

4. **Inference & Evaluation** (`inference.py`)
   - Code generation from natural language
   - Program validation and sequence testing
   - Evaluation metrics (validity, accuracy)
   - Interactive generation interface

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The new LLM dependencies include:
   - `torch>=1.9.0` - PyTorch for deep learning
   - `transformers>=4.20.0` - Hugging Face transformers (T5)
   - `datasets>=2.0.0` - Data loading utilities
   - `tqdm>=4.62.0` - Progress bars
   - `scikit-learn>=1.0.0` - Evaluation metrics

## Usage

### 1. Prepare Training Data

```python
from loda.ml.llm.data_preprocessing import create_dataset

# Create training dataset from OEIS programs
dataset = create_dataset(
    programs_dir="programs/oeis",
    output_file="loda_training_data.json",
    max_examples=10000,  # Use subset for faster training
    augment=True  # Create description variations
)
```

### 2. Train the Model

```python
from loda.ml.llm.trainer import train_loda_llm

# Train the model
model = train_loda_llm(
    programs_dir="programs/oeis",
    output_dir="trained_model",
    model_name="t5-small",  # or "t5-base", "t5-large"
    max_examples=10000,
    num_epochs=3,
    batch_size=8
)
```

Command line training:
```bash
python -m loda.ml.llm.trainer \
    --programs_dir programs/oeis \
    --output_dir trained_model \
    --max_examples 10000 \
    --num_epochs 3
```

### 3. Generate Code

```python
from loda.ml.llm.inference import load_model_for_inference

# Load trained model
generator = load_model_for_inference("trained_model")

# Generate code
results = generator.generate("Fibonacci numbers")
for result in results:
    print(f"Generated: {result.generated_code}")
    print(f"Valid: {result.is_valid}")
    if result.generated_sequence:
        print(f"Sequence: {result.generated_sequence}")
```

Interactive mode:
```bash
python -m loda.ml.llm.inference --mode interactive --model_path trained_model
```

### 4. Evaluate Performance

```python
from loda.ml.llm.inference import evaluate_model

# Evaluate on test set
metrics, results = evaluate_model("trained_model", "test_data.json")
print(f"Valid program rate: {metrics['valid_program_rate']:.1%}")
print(f"Sequence match rate: {metrics['sequence_match_rate']:.1%}")
```

## Training Data Format

Training examples are JSON objects with the following structure:

```json
{
  "sequence_id": "A000045",
  "description": "Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1",
  "loda_code": "mov $1,$0\nmov $4,1\nlpb $0\n...",
  "terms": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
}
```

## Model Configuration

### Supported T5 Models

- `t5-small` (60M parameters) - Fast training, good for experimentation
- `t5-base` (220M parameters) - Better quality, moderate resource requirements  
- `t5-large` (770M parameters) - Best quality, high resource requirements

### Training Parameters

```python
# Recommended settings for different use cases

# Quick experimentation
train_loda_llm(
    model_name="t5-small",
    max_examples=1000,
    batch_size=16,
    num_epochs=1,
    learning_rate=1e-4
)

# Production training
train_loda_llm(
    model_name="t5-base", 
    max_examples=-1,  # Use all data
    batch_size=8,
    num_epochs=5,
    learning_rate=5e-5
)
```

## Evaluation Metrics

The system provides several evaluation metrics:

- **Valid Program Rate**: Percentage of generated programs that parse and execute
- **Exact Match Rate**: Percentage matching the target program exactly
- **Sequence Match Rate**: Percentage generating correct sequence terms
- **Generation Time**: Average time to generate code

## Implementation Details

### LODA Tokenization

The system uses a custom tokenizer designed for LODA assembly:

```python
# LODA operations
operations = ['mov', 'add', 'sub', 'mul', 'div', 'lpb', 'lpe', ...]

# Memory operands  
operands = ['$0', '$1', '$2', '$$1', '$$2', ...]

# Constants
constants = ['0', '1', '2', '-1', ...]
```

### Text Format Conversion

Since T5 expects text input/output, LODA code is converted to a text representation:

```
Original LODA:     mov $1,$0
                   add $1,5
                   
Text format:       mov $1 $0 | add $1 5
```

### Data Augmentation

Training descriptions are augmented to improve robustness:

```
Original: "Fibonacci numbers"
Augmented: 
- "Sequence of fibonacci numbers"
- "Generate fibonacci numbers"  
- "Compute fibonacci numbers"
```

## Performance Considerations

### Memory Usage

- T5-small: ~2GB GPU memory for training
- T5-base: ~8GB GPU memory for training
- T5-large: ~16GB GPU memory for training

### Training Time

Approximate training times (on V100 GPU):
- 1,000 examples: 10-30 minutes
- 10,000 examples: 2-6 hours  
- 100,000+ examples: 1-3 days

### Generation Speed

- T5-small: ~0.1-0.5 seconds per program
- T5-base: ~0.2-1.0 seconds per program
- T5-large: ~0.5-2.0 seconds per program

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use smaller model
2. **Poor generation quality**: Train longer or use larger model
3. **Invalid programs**: Check training data quality and augmentation

### Model Selection

Choose model size based on your requirements:

| Use Case | Model | Trade-offs |
|----------|-------|------------|
| Research/Experimentation | t5-small | Fast, lower quality |
| Production/Demo | t5-base | Balanced speed/quality |
| Best Results | t5-large | Slow, highest quality |

## Extending the System

### Custom Training Data

Add new training examples:

```python
from loda.ml.llm.data_preprocessing import TrainingExample

custom_example = TrainingExample(
    sequence_id="custom_001",
    description="Powers of 2", 
    loda_code="mov $1,1\nlpb $0\n  mul $1,2\n  sub $0,1\nlpe\nmov $0,$1",
    terms=[1, 2, 4, 8, 16, 32]
)
```

### Fine-tuning

Fine-tune on specific sequence types:

```python
# Load pre-trained model
model = LodaT5Model.load_model("base_model")

# Train on specialized data
train_loda_llm(
    programs_dir="specialized_programs",
    model=model,  # Start from pre-trained
    learning_rate=1e-5,  # Lower learning rate
    num_epochs=1
)
```

## Future Improvements

- **Better tokenization**: Domain-specific vocabulary
- **Program synthesis**: Multi-step reasoning
- **Verification**: Formal correctness checking  
- **Interactive refinement**: Human-in-the-loop generation
- **Specialized architectures**: CodeBERT, CodeT5+ integration

---

For more information, see the LODA project documentation and the individual module docstrings.