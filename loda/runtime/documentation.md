You can evaluate LODA programs to an integer sequence as follows:

```python
from loda.lang import Program
from loda.runtime import Interpreter

with open("fibonacci.asm", "r") as file:
    program = Program(file.read())
    interpreter = Interpreter()
    evaluator = Evaluator(program, interpreter)
    for _ in range(10):
        print(evaluator())
```
