import os.path

from loda.lang import Program
from loda.oeis import ProgramCache, Sequence
from loda.runtime import Evaluator, Interpreter
from loda.mine import Miner
from loda.ml.keras.program_generation_rnn import load_model, train_model, Generator


class SampleLODA:

    def __init__(self):
        # Initialize LODA programs cache using *.asm files from tests folder.
        programs_dir = os.path.join("tests", "programs", "oeis")
        # programs_dir = os.path.expanduser("~/loda/programs/oeis")
        self.program_cache = ProgramCache(programs_dir)
        self.interpreter = Interpreter(self.program_cache)

    def print_program(self):
        # Load the LODA program for the Fibonacci numbers (A000045.asm).
        # See also the integer sequence entry at https://oeis.org/A000045.
        program = self.program_cache.get(45)  # numeric version of A000045
        print(program)

    def eval_program_to_seq(self):
        # Evaluate the program to an integer sequence.
        # This time we load it from a file.
        with open("fibonacci.asm", "r") as file:
            program = Program(file.read())
            evaluator = Evaluator(program, self.interpreter)
            for _ in range(10):
                print(evaluator())


if __name__ == "__main__":
    sample = SampleLODA()
    sample.print_program()
    sample.eval_program_to_seq()
