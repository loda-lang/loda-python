"""Evaluate programs to integer sequences."""

from loda.lang import Program
from .interpreter import Interpreter


class Evaluator:

    num_total_steps: int
    """Total number of executed interpreter steps"""

    def __init__(self, program: Program, interpreter: Interpreter, eval_to_steps=False):
        """
        Evaluate a program to an integer sequence.

        Args:
            program: Program that should be evaluated.
            interpreter: Interpreter to be used.
            eval_to_steps: Flag indicating whether the number of execution steps should
                be returned instead of the actual evaluation result.

        ## Example
        >>> Initialize evaluator:
        >>> fibonacci = Program(...)
        >>> interpreter = Interpreter()
        >>> evaluator = Evaluator(fibonacci, interpreter)
        >>>
        >>> # Evaluate first ten terms:
        >>> for _ in range(10)
        >>>     print(evaluator())
        """
        self.__program = program
        self.__interpreter = interpreter
        self.__eval_to_steps = eval_to_steps
        self.__memory = {}
        self.__argument = 0
        self.num_total_steps = 0

    def __call__(self):
        self.__memory.clear()
        self.__memory[0] = self.__argument
        num_steps = self.__interpreter.run(self.__program, self.__memory)
        self.num_total_steps += num_steps
        self.__argument += 1
        return num_steps if self.__eval_to_steps else self.__memory[0]
