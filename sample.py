import os.path

from loda.lang import Program
from loda.oeis import ProgramCache
from loda.runtime import Interpreter
from loda.ml import KerasModel
from loda.ml.encoding import *

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

    def keras_model(self):
        model = KerasModel(self.program_cache)
        print("Model Tokens: {}\n".format(model.tokens))
        print("Vocabulary: {}\n".format(model.vocab))
        print("Model IDs: {}\n".format(model.ids))
        for input, label in model.split_dataset.take(3):
            print("Input:", model.ids_to_tokens_str(input))
            print("Label:", model.ids_to_tokens_str(label), "\n")
        for input_example_batch, _ in model.prefetch_dataset.take(1):
            example_batch_predictions = model(input_example_batch)
            print(example_batch_predictions.shape,
                  "# (batch_size, sequence_length, vocab_size)")
        model.summary()


if __name__ == "__main__":
    sample = SampleLODA()
    sample.print_program()
    sample.eval_program_to_seq()
    sample.keras_model()
