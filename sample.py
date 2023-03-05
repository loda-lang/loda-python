import os.path

from loda.lang import Program
from loda.oeis import ProgramCache
from loda.runtime import Interpreter
from loda.ml import Model

import tensorflow as tf


class SampleLODA:

    def __init__(self):
        # Initialize LODA programs cache using *.asm files from tests folder
        program_dir = os.path.join('tests', 'programs', 'oeis')
        self.program_cache = ProgramCache(program_dir)

    def print_program(self):
        # Load the LODA program for the prime numbers (A000040.asm)
        # See also the integer sequence entry at https://oeis.org/A000040
        program = self.program_cache.get(40)  # numeric version of A000040
        print(program)

    def eval_program_to_seq(self):
        # Evaluate the program to an integer sequence
        program = self.program_cache.get(40)  # numeric version of A000040
        interpreter = Interpreter(program_cache=self.program_cache)
        sequence, _ = interpreter.eval_to_seq(program, num_terms=20)
        print("Evaluated to sequence: {}\n".format(sequence))

    def ml_model(self):
        program = self.program_cache.get(40)
        model = Model(program)
        print("Program to Tokens: {}\n".format(model.tokens))
        print("Vocabulary: {}\n".format(model.vocab))
        print("Tokens to IDs: {}\n".format(model.ids))
        programs = model.ids_to_programs(model.ids)
        print("Tokens to Program: {}".format(programs[0]))


if __name__ == "__main__":
    sample = SampleLODA()
    sample.print_program()
    sample.eval_program_to_seq()
    sample.ml_model()
