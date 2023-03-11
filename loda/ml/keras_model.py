from loda.lang import Program
from loda.oeis import ProgramCache
from .encoding import merge_programs, program_to_tokens, split_program, tokens_to_program

import tensorflow as tf


# TODO: inherit from tf.keras.Model
class KerasModel:

    num_ops_per_sample: int

    def __init__(self, program_or_program_cache, num_ops_per_sample: int = 3,
                 batch_size: int = 64, buffer_size: int = 10000):
        if isinstance(program_or_program_cache, Program):
            program = program_or_program_cache
        elif isinstance(program_or_program_cache, ProgramCache):
            # Merge all programs into one program
            program = merge_programs(
                program_or_program_cache, num_ops_per_sample)
        else:
            raise ValueError("invalid argument")
        self.num_ops_per_sample = num_ops_per_sample

        # Convert to tokens and vocabulary
        # TODO: do we really need to store them as memebers?
        self.tokens, self.vocab = program_to_tokens(program)

        # Initialize TF lookup layers
        self.tokens_to_ids = tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab), mask_token=None)
        self.ids_to_tokens = tf.keras.layers.StringLookup(
            vocabulary=self.tokens_to_ids.get_vocabulary(), invert=True, mask_token=None)

        # Convert the tokens to IDs
        # TODO: do we really need to store them as members?
        self.ids = self.tokens_to_ids(self.tokens)
        self.slice_dataset = tf.data.Dataset.from_tensor_slices(self.ids)

        # One operation is encoded using three tokens
        # plus one token because we split into input/output
        self.batch_dataset = self.slice_dataset.batch(
            3 * self.num_ops_per_sample + 1, drop_remainder=True)

        self.split_dataset = self.batch_dataset.map(self.__split_input_label)

        self.prefetch_dataset = (self.split_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    def ids_to_tokens_str(self, ids) -> list[str]:
        return [t.numpy().decode("utf-8") for t in self.ids_to_tokens(ids)]

    def ids_to_programs(self, ids) -> list[Program]:
        return split_program(tokens_to_program(self.ids_to_tokens_str(ids)))

    def __split_input_label(self, sample: list):
        input = sample[:-1]
        label = sample[1:]
        return input, label
