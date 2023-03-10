from loda.lang import Program
from loda.oeis import ProgramCache
from .encoding import merge_programs, program_to_tokens, split_program, tokens_to_program

import tensorflow as tf

# TODO: inherit from tf.keras.Model


class KerasModel:

    num_ops_per_batch: int

    def __init__(self, program_or_program_cache, num_ops_per_batch: int = 3):
        if isinstance(program_or_program_cache, Program):
            program = program_or_program_cache
        elif isinstance(program_or_program_cache, ProgramCache):
            # Merge all programs into one program
            program = merge_programs(
                program_or_program_cache, num_ops_per_batch)
        else:
            raise ValueError("invalid argument")
        self.num_ops_per_batch = num_ops_per_batch

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
        self.ids_dataset = tf.data.Dataset.from_tensor_slices(self.ids)
        # one operation is encoded using three tokens
        self.ids_batches = self.ids_dataset.batch(
            3 * self.num_ops_per_batch, drop_remainder=True)

    def ids_to_tokens_str(self, ids) -> list[str]:
        return [t.numpy().decode("utf-8") for t in self.ids_to_tokens(ids)]

    def ids_to_programs(self, ids) -> list[Program]:
        return split_program(tokens_to_program(self.ids_to_tokens_str(ids)))
