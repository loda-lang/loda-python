import os.path

from loda.lang import Program
from loda.oeis import ProgramCache
from loda.runtime import Interpreter
from loda.ml import program_to_tokens, tokens_to_program, merge_programs

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

    def programs_to_tensor(self):
        program = self.program_cache.get(40)
        # Convert to tokens and vocabulary
        tokens, vocab = program_to_tokens(program)
        print("Program to Tokens: {}\n".format(tokens))
        print("Vocabulary: {}\n".format(vocab))
        # Vectorize tokens (convert to IDs)
        ids_from_tokens = tf.keras.layers.StringLookup(
            vocabulary=list(vocab), mask_token=None)
        ids = ids_from_tokens(tokens)
        print("Tokens to IDs: {}\n".format(ids))
        # Convert IDs back to tokens
        tokens_from_ids = tf.keras.layers.StringLookup(
            vocabulary=ids_from_tokens.get_vocabulary(), invert=True, mask_token=None)
        tokens = [t.numpy().decode("utf-8") for t in tokens_from_ids(ids)]
        print("IDs to Tokens: {}\n".format(tokens))
        # Convert tokens back to programs
        program = tokens_to_program(tokens)
        print("Tokens to Program: {}".format(program))


if __name__ == "__main__":
    sample = SampleLODA()
    sample.print_program()
    sample.eval_program_to_seq()
    sample.programs_to_tensor()
