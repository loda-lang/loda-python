from loda.lang import Program
from loda.oeis import ProgramCache
from .encoding import *

import tensorflow as tf


class Model:
    def __init__(self, program_or_program_cache):
        if isinstance(program_or_program_cache, Program):
            program = program_or_program_cache
        elif isinstance(program_or_program_cache, ProgramCache):
            # Merge all programs into one program
            program = merge_programs(program_or_program_cache)
        else:
            raise ValueError("invalid argument")

        # Convert to tokens and vocabulary
        # TODO: do we really need to store them as memebers?
        self.tokens, self.vocab = program_to_tokens(program)

        # Initialize TF lookup layers
        self.tokens_to_ids = tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab), mask_token=None)
        self.ids_to_tokens = tf.keras.layers.StringLookup(
            vocabulary=self.tokens_to_ids.get_vocabulary(), invert=True, mask_token=None)

        # Convert the tokens to IDs
        # TODO: do we really need to store them as memebers?
        self.ids = self.tokens_to_ids(self.tokens)

    def ids_to_programs(self, ids) -> list[Program]:
        tokens = [t.numpy().decode("utf-8") for t in self.ids_to_tokens(ids)]
        return split_program(tokens_to_program(tokens))
