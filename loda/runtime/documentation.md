You can evaluate LODA programs to an integer sequence as follows:

```python
from loda.lang import Program
from loda.runtime import Interpreter

with open("fibonacci.asm", "r") as file:
    program = Program(file.read())
    interpreter = Interpreter()
    print(interpreter.eval_to_seq(program, num_terms=10))
```

The result is a pair, where the first entry is the list of computed Fibonacci numbers:

```python
([0, 1, 1, 2, 3, 5, 8, 13, 21, 34], 305)
```
