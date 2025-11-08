# GitHub Copilot Instructions for LODA Python Project

## Project Overview

This is a Python implementation of LODA - an assembly language designed for integer sequences. The project enables reading, writing, evaluating LODA programs and searching for matches in the OEIS database.

## Core Concepts

### LODA Assembly Language
- **Memory Model**: Integer memory cells accessed by index, cell 0 contains input/output
- **Operand Types**: 
  - Constants: `5`, `-3`
  - Direct memory: `$1`, `$2` (value at memory location)
  - Indirect memory: `$$1` (value at location pointed to by $1)
- **Operations**: `mov`, `add`, `sub`, `mul`, `div`, `dif`, `mod`, `pow`, `gcd`, `bin`, `cmp`, `min`, `max`, `lpb`, `lpe`
- **Loops**: `lpb $n` starts loop, `lpe` ends loop (counter-based termination)

## Source Code Structure

### Core Language (`loda/lang/`)
- **`operand.py`**: `Operand` class with types CONSTANT, DIRECT, INDIRECT
- **`operation.py`**: `Operation` class representing single assembly instructions
- **`program.py`**: `Program` class containing list of operations, handles parsing

### Runtime System (`loda/runtime/`)
- **`interpreter.py`**: `Interpreter` executes programs with memory management and resource limits
- **`evaluator.py`**: `Evaluator` high-level interface for generating integer sequences
- **`operations.py`**: Implementation of all arithmetic operations

### OEIS Integration (`loda/oeis/`)
- **`sequence.py`**: `Sequence` class with OEIS metadata and b-file loading
- **`program_cache.py`**: `ProgramCache` manages filesystem loading/caching
- **`prefix_index.py`**: `PrefixIndex` enables sequence matching by prefix patterns

### Utilities (`loda/ml/`)
- **`util.py`**: Token conversion utilities (program ↔ tokens, merging)

### Mining (`loda/mine/`)
- **`miner.py`**: `Miner` searches for programs matching OEIS sequences

## Coding Guidelines

### When working with Programs:
```python
# Always handle parsing errors
try:
    program = Program.parse(program_text)
except Exception as e:
    # Handle parse error
    
# Use resource limits for evaluation
interpreter = Interpreter(max_memory=1000, max_stack=10, max_steps=100000)
```

### When working with Operands:
```python
# Check operand types before operations
if operand.type == OperandType.CONSTANT:
    value = operand.value
elif operand.type == OperandType.DIRECT:
    value = memory[operand.value]
elif operand.type == OperandType.INDIRECT:
    value = memory[memory[operand.value]]
```

### When working with sequences:
```python
# Always specify term count and handle evaluation errors
evaluator = Evaluator(program, interpreter)
try:
    terms = [evaluator(i) for i in range(num_terms)]
except Exception:
    # Handle evaluation error (infinite loop, overflow, etc.)
```

## Common Patterns

### Program Evaluation Pattern:
```python
program = Program.parse(program_text)
interpreter = Interpreter()
evaluator = Evaluator(program, interpreter)
sequence_terms = []
for i in range(10):
    try:
        term = evaluator(i)
        sequence_terms.append(term)
    except Exception:
        break
```

### Caching Pattern:
```python
# Use caches for performance
program_cache = ProgramCache("path/to/programs")
program = program_cache.get_program(sequence_id)
```

### Token Conversion Pattern:
```python
# Token conversion utilities
from loda.ml.util import program_to_tokens, tokens_to_program

tokens, vocab = program_to_tokens(program)
reconstructed = tokens_to_program(tokens)
```

## Testing Conventions

- Use CSV files in `tests/operations/` for operation test cases
- Sample programs go in `tests/programs/`
- Unit tests follow `test_*.py` naming convention
- Test both valid and invalid inputs for robustness

## Resource Management

Always set appropriate limits:
- `max_memory`: Prevent excessive memory usage
- `max_steps`: Prevent infinite loops  
- `max_stack`: Prevent stack overflow in nested loops
- Handle `MemoryError`, `RuntimeError`, and `TimeoutError`

## File Naming and Organization

- Programs: `A######.asm` format (OEIS sequence numbers)
- B-files: `b######.txt` format for sequence terms
- Use relative paths from project root

## Integration Points

- OEIS database integration via sequence IDs
- File system caching for performance
- CSV parsing for test data

Remember: LODA programs are deterministic and should produce consistent integer sequences. Always validate generated programs before use.
